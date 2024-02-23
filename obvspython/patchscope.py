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
from dataclasses import dataclass

import torch
from nnsight import LanguageModel

from obvspython.patchscope_base import PatchscopeBase
from obvspython.logging import logger


@dataclass
class SourceContext:
    """
    Source context for the patchscope
    """

    prompt: Sequence[str] = "<|endoftext|>"
    position: Sequence[int] | None = None
    layer: int = -1
    model_name: str = "gpt2"
    device: str = "cuda:0"

    def __repr__(self) -> str:
        return (
            f"SourceContext(prompt={self.prompt}, position={self.position}, "
            f"model_name={self.model_name}, layer={self.layer}, device={self.device})"
        )


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
        return TargetContext(
            prompt=source.prompt,
            position=source.position,
            model_name=source.model_name,
            layer=source.layer,
            mapping_function=mapping_function or (lambda x: x),
            max_new_tokens=max_new_tokens,
            device=source.device,
        )

    def __repr__(self) -> str:
        return (
            f"TargetContext(prompt={self.prompt}, position={self.position}, "
            f"model_name={self.model_name}, layer={self.layer}, device={self.device}, "
            f"max_new_tokens={self.max_new_tokens}, "
            f"mapping_function={self.mapping_function})"
        )


class ModelLoader:
    @staticmethod
    def load(model_name: str, device: str) -> LanguageModel:
        if 'mamba' in model_name:
            # We import here because MambaInterp depends on some GPU libs that might not be installed.
            from nnsight.models.Mamba import MambaInterp
            logger.info(f"Loading Mamba model: {model_name}")
            return MambaInterp(model_name, device=device)
        else:
            logger.info(f"Loading NNsight LanguagModel: {model_name}")
            return LanguageModel(model_name, device_map=device)

    @staticmethod
    def generation_kwargs(model_name: str, max_new_tokens: int) -> dict:
        if "mamba" not in model_name:
            return {"max_new_tokens": max_new_tokens}
        else:
            return {"max_length": max_new_tokens}


class Patchscope(PatchscopeBase):
    REMOTE: bool = False

    def __init__(self, source: SourceContext, target: TargetContext) -> None:
        self.source = source
        self.target = target
        logger.info(f"Patchscope initialize with source:\n{source}\nand target:\n{target}")

        self.source_model = ModelLoader.load(self.source.model_name, device=self.source.device)

        if self.source.model_name == self.target.model_name and self.source.device == self.target.device:
            self.target_model = self.source_model
        else:
            self.target_model = ModelLoader.load(self.target.model_name, device=self.target.device)

        self.generation_kwargs = ModelLoader.generation_kwargs(self.target.model_name, self.target.max_new_tokens)

        self.tokenizer = self.source_model.tokenizer
        self.init_positions()

        self.MODEL_SOURCE, self.LAYER_SOURCE = self.get_model_specifics(self.source.model_name)
        self.MODEL_TARGET, self.LAYER_TARGET = self.get_model_specifics(self.target.model_name)

        self._target_outputs: list[torch.Tensor] = []

    def source_forward_pass(self) -> None:
        """
        Get the source representation.

        We use the 'trace' context so we can add the REMOTE option.

        For each architecture, you need to know the name of the layers.
        """
        with self.source_model.trace(self.source.prompt, remote=self.REMOTE) as _:
            self._source_hidden_state = self.manipulate_source().save()

    def manipulate_source(self) -> torch.Tensor:
        """
        Get the hidden state from the source representation.

        NB: This is seperated out from the source_forward_pass method to allow for batching.
        """
        return getattr(getattr(self.source_model, self.MODEL_SOURCE), self.LAYER_SOURCE)[
            self.source.layer
        ].output[0][:, self.source.position, :]

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
            self.target.prompt,
            remote=self.REMOTE,
            **self.generation_kwargs,
        ) as _:
            self.manipulate_target()

    def manipulate_target(self) -> None:
        (
            getattr(getattr(self.target_model, self.MODEL_TARGET), self.LAYER_TARGET)[
                self.target.layer
            ].output[0][:, self.target.position, :]
        ) = self._mapped_hidden_state

        self._target_outputs.append(self.target_model.lm_head.output[0].save())
        for _ in range(self.target.max_new_tokens - 1):
            self._target_outputs.append(self.target_model.lm_head.next().output[0].save())

    def run(self) -> None:
        """
        Run the patchscope
        """
        self._target_outputs = []
        self.source_forward_pass()
        self.map()
        self.target_forward_pass()
