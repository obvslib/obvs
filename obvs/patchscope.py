# Implementation of the patchscopes framework: https://arxiv.org/abs/2401.06102
# Patchscopes takes a representation like so:
# - (S, i, M, ℓ) corresponds to the source from which the original hidden representation is drawn.
#   - S is the source input sequence.
#   - i is the position within that sequence.
#           NB: We extend the method to allow a range of positions
#   - M is the original model that processes the sequence.
#   - ℓ is the layer in model M from which the hidden representation is taken.
#
# and patches it to a target context like so:
# - (T, i*, f, M*, ℓ*) defines the target context for the intervention (patching operation).
#   - T is the target prompt, which can be different from the source prompt S or the same.
#   - i* is the position in the target prompt that will receive the patched representation.
#           NB: We extend the method to allow a range of positions
#   - f is the mapping function that operates on the hidden representation to possibly transform
#       it before it is patched into the target context. It can be a simple identity function or a more complex transformation.
#   - M* is the model (which could be the same as M or different) in which the patching operation is performed.
#   - ℓ* is the layer in the target model M* where the hidden representation h̅ᵢˡ* will be replaced with the patched version.
#
# The simplest patchscope is defined by the following parameters:
# - S = T
# - i = i*
# - M = M*
# - ℓ = ℓ*
# - f = identity function
# this be indistinguishable from a forward pass.
#
# The most simple one that does something interesting is the logit lens, where:
# - ℓ = range(L*)
# - ℓ* = L*
# Meaning, we take the hidden representation from each layer of the source model and patch it into the final layer of the target model.

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import einops
import torch
from nnsight import LanguageModel
from tqdm import tqdm

from obvs.logging import logger
from obvs.patchscope_base import PatchscopeBase


@dataclass
class SourceContext:
    """
    Source context for the patchscope
    """

    _prompt: str | torch.Tensor = field(init=False, repr=False, default="<|endoftext|>")
    _text_prompt: str = field(init=False, repr=False)
    _soft_prompt: torch.Tensor | None = field(init=False, repr=False)

    prompt: str | torch.Tensor
    position: Sequence[int] | None = None
    layer: int = -1
    head: Sequence[int] | None = None
    model_name: str = "gpt2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # This overrides the `prompt` field
    # See https://florimond.dev/en/posts/2018/10/reconciling-dataclasses-and-properties-in-python
    @property
    def prompt(self) -> str | torch.Tensor:
        """
        The prompt
        """
        return self._prompt

    @prompt.setter
    def prompt(self, value: str | torch.Tensor | None):
        if isinstance(value, property):
            # initial value not specified, use default
            value = SourceContext._prompt

        if value is None:
            value = "<|endoftext|>"

        if isinstance(value, torch.Tensor):
            if value.dim() == 2:
                # Add batch dimension
                value = torch.unsqueeze(value, 0)

            if value.dim() != 3:
                raise ValueError(
                    f"Soft prompt must have shape [tokens_len, d_model] or [batch, tokens_len, d_model]. But prompt.shape is {value.shape}",
                )

            self._text_prompt = " ".join("_" * value.shape[1])
            self._soft_prompt = value
        elif isinstance(value, str):
            self._text_prompt = value
            self._soft_prompt = None
        else:
            raise ValueError(f"Unknown prompt type: {type(value)}")

        self._prompt = value

    @property
    def text_prompt(self) -> str:
        """
        The text prompt input or generated from soft prompt
        """
        return self._text_prompt

    @property
    def soft_prompt(self) -> torch.Tensor | None:
        """
        The soft prompt input or None
        """
        return self._soft_prompt


@dataclass
class TargetContext(SourceContext):
    """
    Target context for the patchscope
    Parameters identical to the source context, with the addition of
    a mapping function and max_new_tokens to control generation length
    """

    mapping_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
    max_new_tokens: int = 10

    @staticmethod
    def from_source(
        source: SourceContext,
        mapping_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
        max_new_tokens: int = 10,
    ) -> TargetContext:
        """
        Construct a target context from the source context
        """

        return TargetContext(
            prompt=source.prompt,
            position=source.position,
            model_name=source.model_name,
            layer=source.layer,
            head=source.head,
            mapping_function=mapping_function or (lambda x: x),
            max_new_tokens=max_new_tokens,
            device=source.device,
        )


class ModelLoader:
    @staticmethod
    def load(model_name: str, device: str) -> LanguageModel:
        if "mamba" in model_name:
            # pylint: disable=import-outside-toplevel
            # We import here because MambaInterp depends on some GPU libs that might not be installed.
            from nnsight.models.Mamba import MambaInterp

            logger.info(f"Loading Mamba model: {model_name}")
            return MambaInterp(model_name, device=device)
        else:
            logger.info(f"Loading NNsight LanguagModel: {model_name}")
            return LanguageModel(model_name, device_map=device)


class Patchscope(PatchscopeBase):
    REMOTE: bool = False

    def __init__(self, source: SourceContext, target: TargetContext) -> None:
        self.source = source
        self.target = target
        logger.info(f"Patchscope initialize with source:\n{source}\nand target:\n{target}")

        self.source_model = ModelLoader.load(self.source.model_name, device=self.source.device)

        if (
            self.source.model_name == self.target.model_name
            and self.source.device == self.target.device
        ):
            self.target_model = self.source_model
        else:
            self.target_model = ModelLoader.load(self.target.model_name, device=self.target.device)

        self.tokenizer = self.source_model.tokenizer

        (
            self.source_base_name,
            self.source_layer_name,
            self.source_attn_name,
            self.source_head_name,
        ) = self.get_model_specifics(self.source.model_name)
        (
            self.target_base_name,
            self.target_layer_name,
            self.target_attn_name,
            self.target_head_name,
        ) = self.get_model_specifics(self.target.model_name)

        self._source_hidden_state = None
        self.source_output = None

        self._mapped_hidden_state = None

        self._target_outputs: list[torch.Tensor] = []

    def source_forward_pass(self) -> None:
        """
        Get the source representation.

        We use the 'trace' context so we can add the REMOTE option.

        For each architecture, you need to know the name of the layers.
        """
        with self.source_model.trace(self.source.text_prompt, remote=self.REMOTE):
            if self.source.soft_prompt is not None:
                # TODO: validate this with non GPT2 & GPTJ models
                self.source_model.transformer.wte.output = self.source.soft_prompt

            self._source_hidden_state = self.manipulate_source().detach().save()
            self.source_output = self.source_model.lm_head.output[0].detach().save()

        self._source_hidden_state = self._source_hidden_state.value

    def manipulate_source(self) -> torch.Tensor:
        """
        Get the hidden state from the source representation.

        NB: This is seperated out from the source_forward_pass method to allow for batching.
        """

        # get the specified layer
        layer = getattr(getattr(self.source_model, self.source_base_name), self.source_layer_name)[
            self.source.layer
        ]

        # if a head index is given, need to access the ATTN and HEAD components
        if self.source.head is not None:
            attn = getattr(layer, self.source_attn_name)
            # TODO may not be .input for other models
            head_act = getattr(attn, self.source_head_name).input[0][0]

            # need to reshape the output into the specific heads
            head_act = einops.rearrange(
                head_act,
                "batch pos (n_head d_head) -> batch pos n_head d_head",
                n_head=attn.num_heads,
                d_head=attn.head_dim,
            )
            return head_act[:, self._source_position, self.source.head, :]

        return layer.output[0][:, self._source_position, :]

    def map(self) -> None:
        """
        Apply the mapping function to the source representation
        """
        self._mapped_hidden_state = self.target.mapping_function(self._source_hidden_state)

    def target_forward_pass(self) -> None:
        """
        Patch the target representation.
        In order to support multi-token generation,
        we save the output for max_new_tokens iterations.

        We use a the 'generate' context which support remote operation and multi-token generation

        For each architecture, you need to know the name of the layers.
        """
        with self.target_model.generate(
            self.target.text_prompt,
            remote=self.REMOTE,
            max_new_tokens=self.target.max_new_tokens,
        ) as _:
            if self.target.soft_prompt is not None:
                # Not sure if this works with mamba and other models
                self.target_model.transformer.wte.output = self.target.soft_prompt

            self.manipulate_target()

    def manipulate_target(self) -> None:
        # get the specified layer
        layer = getattr(getattr(self.target_model, self.target_base_name), self.target_layer_name)[
            self.target.layer
        ]

        # if a head index is given, need to access the ATTN and HEAD components
        if self.target.head is not None:
            attn = getattr(layer, self.target_attn_name)
            # TODO may not be .input for other models
            concat_head_act = getattr(attn, self.target_head_name).input[0][0]

            # need to reshape the output of head into the specific heads
            split_head_act = einops.rearrange(
                concat_head_act,
                "batch pos (n_head d_head) -> batch pos n_head d_head",
                n_head=attn.num_heads,
                d_head=attn.head_dim,
            )

            # check if the dimensions of the mapped_hidden_state and target head activations match
            target_act = split_head_act[:, self._target_position, self.target.head, :]
            if self._mapped_hidden_state.shape != target_act.shape:
                raise ValueError(
                    f"Cannot set activation of head {self.target.head} in target model with shape"
                    f" {list(target_act.shape)} to patched activation of source model with shape"
                    f" {list(self._mapped_hidden_state.shape)}!",
                )
            split_head_act[
                :,
                self._target_position,
                self.target.head,
                :,
            ] = self._mapped_hidden_state
        else:
            layer.output[0][:, self._target_position, :] = self._mapped_hidden_state

        self._target_outputs.append(self.target_model.lm_head.output[0].save())
        for _ in range(self.target.max_new_tokens - 1):
            self._target_outputs.append(self.target_model.lm_head.next().output[0].save())

    def check_patchscope_setup(self) -> bool:
        """Check if patchscope is correctly set-up before running"""

        # head can be int or None, patchscope run is only possible if they have the same type
        if not isinstance(self.source.head, type(self.target.head)):
            logger.error(
                f"Cannot run patchscope with source head attribute: {self.source.head} and target head attribute: {self.target.head}. Both need to be of the same type.",
            )
            return False

        # currently, accessing single head activations is only supported for GPT2LMHead models
        if (
            self.source.head is not None
            and "gpt2" not in self.source.model_name
            or self.target.head is not None
            and "gpt2" not in self.target.model_name
        ):
            raise NotImplementedError(
                "Accessing single head activations is currently only"
                " implemented for GPT2-style models",
            )

        return True

    def run(self) -> None:
        """
        Run the patchscope
        """

        # check before running
        if not self.check_patchscope_setup():
            raise ValueError("Cannot run patchscope with the provided arguments")

        self.clear()
        self.source_forward_pass()
        self.map()
        self.target_forward_pass()

    def clear(self) -> None:
        """
        Clear the outputs and the cache
        """
        self._target_outputs = []
        if hasattr(self, "source_output"):
            del self.source_output
        if hasattr(self, "_source_hidden_state"):
            del self._source_hidden_state
        if hasattr(self, "_mapped_hidden_state"):
            del self._mapped_hidden_state
        torch.cuda.empty_cache()

    def over(
        self,
        source_layers: Sequence[int],
        target_layers: Sequence[int],
    ) -> list[torch.Tensor]:
        """
        Run the patchscope over the specified set of layers.

        :param source_layers: A list of layer indices or a range of layer indices.
        :param target_layers: A list of layer indices or a range of layer indices.
        :return: A source_layers x target_layers x max_new_tokens list of outputs.
        """
        logger.info("Running sets.")
        for i in source_layers:
            self.source.layer = i
            for j in target_layers:
                self.target.layer = j
                logger.info(f"Running Source Layer-{i}, Target Layer-{j}")
                self.run()
                logger.info(self.full_output())
                logger.info("Saving last token outputs")
                # Output sizes are too large. For now, we only need the last character of the first output.
                yield self._target_outputs[0][-1, :]

    def over_pairs(
        self,
        source_layers: Sequence[int],
        target_layers: Sequence[int],
    ) -> list[torch.Tensor]:
        """
        Run the patchscope over the specified set of layers in pairs
        :param source_layers: A list of layer indices or a range of layer indices.
        :param target_layers: A list of layer indices or a range of layer indices.
        :return: A source_layers x target_layers x max_new_tokens list of outputs.
        """
        logger.info("Running pairs.")
        for i, j in tqdm(zip(source_layers, target_layers)):
            self.source.layer = i
            self.target.layer = j
            logger.info(f"Running Source Layer-{i}, Target Layer-{j}")
            self.run()
            logger.info(self.full_output())
            logger.info("Saving last token outputs")
            # Output sizes are too large. For now, we only need the last character of the first output.
            logger.info(self._target_outputs[0].shape)
            yield self._target_outputs[0][-1, :]
