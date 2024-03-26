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

import torch

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from nnsight import LanguageModel
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from obvs.logging import logger


@dataclass
class SourceContext:
    """
    Source context for the patchscope
    """
    # Either text prompt or a soft prompt (aka token embeddings of size [pos, dmodel])
    prompt: str | torch.Tensor | None = None
    position: Sequence[int] | None = None
    layer: int = -1
    model_name: str = "gpt2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        if self.prompt is None:
            self.prompt = "<|endoftext|>"

        # TODO: validation doesn't work after initialization. Maybe create a descriptor
        if self._is_soft_prompt() and self.prompt.dim() != 2:
            raise ValueError(f"soft prompt must have shape [pos, dmodel]. prompt.shape = {self.prompt.shape}")

    @property
    def text_prompt(self) -> str:
        """
        The text prompt input or generated from soft prompt
        """
        if self._is_soft_prompt():
            tokens_count = self.prompt.shape[0]

            # Works with GPT2 & GPTJ, not sure about other models
            return " ".join("_" * tokens_count)

        return self.prompt

    @property
    def soft_prompt(self) -> torch.Tensor | None:
        """
        The soft prompt input or None
        """
        return self.prompt if self._is_soft_prompt() else None

    def _is_soft_prompt(self) -> bool:
        return isinstance(self.prompt, torch.Tensor)



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


class ModelLoader:
    @staticmethod
    def load(model_name: str, device: str) -> LanguageModel:
        if "mamba" in model_name:
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
            return {"max_new_tokens": max_new_tokens}


class PatchscopeBase(ABC):
    """
    A base class with lots of helper functions
    """

    source: SourceContext
    target: TargetContext
    source_model: LanguageModel
    target_model: LanguageModel
    tokenizer: PreTrainedTokenizer

    MODEL_TARGET: str
    LAYER_TARGET: str

    _target_outputs: list[torch.Tensor]

    def get_model_specifics(self, model_name):
        """
        Get the model specific attributes.
        The following works for gpt2, llama2 and mistral models.
        """
        if "gpt" in model_name:
            return "transformer", "h"
        if "mamba" in model_name:
            return "backbone", "layers"
        return "model", "layers"

    @abstractmethod
    def source_forward_pass(self) -> None:
        pass

    @abstractmethod
    def map(self) -> None:
        pass

    @abstractmethod
    def target_forward_pass(self) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    @property
    def source_token_ids(self) -> list[int]:
        """
        Return the source tokens
        """
        return self.tokenizer.encode(self.source.text_prompt)

    @property
    def target_token_ids(self) -> list[int]:
        """
        Return the target tokens
        """
        return self.tokenizer.encode(self.target.text_prompt)

    @property
    def source_tokens(self) -> list[str]:
        """
        Return the input to the source model
        """
        return [self.tokenizer.decode(token) for token in self.source_token_ids]

    @property
    def target_tokens(self) -> list[str]:
        """
        Return the input to the target model
        """
        return [self.tokenizer.decode(token) for token in self.target_token_ids]

    def init_positions(self, force: bool=False) -> None:
        if self.source.position is None or force:
            # If no position is specified, take them all
            self.source.position = range(len(self.source_token_ids))

        if self.target.position is None or force:
            self.target.position = range(len(self.target_token_ids))

    def top_k_tokens(self, k: int=10) -> list[str]:
        """
        Return the top k tokens from the target model
        """
        token_ids = self._target_outputs[0].value[self.target.position, :].topk(k).indices.tolist()
        return [self.tokenizer.decode(token_id) for token_id in token_ids]

    def top_k_logits(self, k: int=10) -> list[int]:
        """
        Return the top k logits from the target model
        """
        return self._target_outputs[0].value[self.target.position, :].topk(k).values.tolist()

    def top_k_probs(self, k: int=10) -> list[float]:
        """
        Return the top k probabilities from the target model
        """
        # FIXME: broken, returns a list of [1.0]
        logits = self.top_k_logits(k)
        return [torch.nn.functional.softmax(torch.tensor(logit), dim=-1).item() for logit in logits]

    def logits(self) -> torch.Tensor:
        """
        Return the logits from the target model (size [pos, d_vocab])
        """
        return self._target_outputs[0].value[:, :]

    def probabilities(self) -> torch.Tensor:
        """
        Return the probabilities from the target model (size [pos, d_vocab])
        """
        return torch.softmax(self.logits(), dim=-1)

    def output(self) -> list[str]:
        """
        Return the generated output from the target model
        """
        token_ids = self.logits().argmax(dim=-1)
        return [self.tokenizer.decode(token_id) for token_id in token_ids]

    def _output_token_ids(self) -> list[int]:
        tensors_list = [tensor_proxy.value for tensor_proxy in self._target_outputs]
        tokens = torch.cat(tensors_list, dim=0)
        return tokens.argmax(dim=-1).tolist()

    def llama_output(self) -> list[str]:
        """
        For llama, if you don't decode them all together, they don't add the spaces.
        """
        tokens = self._output_token_ids()
        return self.tokenizer.decode(tokens)

    def full_output_tokens(self) -> list[str]:
        """
        Return the generated output from the target model
        This is a bit hacky. Its not super well supported. I have to concatenate all the inputs and add the input tokens to them.
        """
        token_ids = self._output_token_ids()

        input_token_ids = self.tokenizer.encode(self.target.text_prompt)
        token_ids.insert(0, " ")
        token_ids[: len(input_token_ids)] = input_token_ids
        return [self.tokenizer.decode(token_id) for token_id in token_ids]

    def full_output(self) -> str:
        """
        Return the generated output from the target model
        This is a bit hacky. Its not super well supported.
        I have to concatenate all the inputs and add the input tokens to them.
        """
        return "".join(self.full_output_tokens())

    def find_in_source(self, substring: str) -> int:
        """
        Find the position of the substring tokens in the source prompt

        Note: only works if substring's tokenization happens to match that of the source prompt's tokenization
        """
        position, _ = self.source_position_tokens(substring)
        return position

    def source_position_tokens(self, substring: str) -> tuple[int, list[int]]:
        """
        Find the position of a substring in the source prompt, and return the substring tokenized

        NB: The try: except block handles the difference between gpt2 and llama
        tokenization. Perhaps this can be better dealt with a seperate tokenizer
        class that handles the differences between the tokenizers. There are a
        few subtleties there, and tokenizing properly is important for getting
        the best out of your model.
        """
        if substring not in self.source.text_prompt:
            raise ValueError(f"Substring {substring} could not be found in {self.source.text_prompt}")

        try:
            token_ids = self.tokenizer.encode(substring, add_special_tokens=False)
            return self.source_token_ids.index(token_ids[0]), token_ids
        except ValueError:
            token_ids = self.tokenizer.encode(" " + substring, add_special_tokens=False)
            return self.source_token_ids.index(token_ids[0]), token_ids

    def find_in_target(self, substring: str) -> int:
        """
        Find the position of the substring tokens in the target prompt

        Note: only works if substring's tokenization happens to match that of the target prompt's tokenization
        """
        position, _ = self.target_position_tokens(substring)
        return position

    def target_position_tokens(self, substring) -> tuple[int, list[int]]:
        """
        Find the position of a substring in the target prompt, and return the substring tokenized

        NB: The try: except block handles the difference between gpt2 and llama
        tokenization. Perhaps this can be better dealt with a seperate tokenizer
        class that handles the differences between the tokenizers. There are a
        few subtleties there, and tokenizing properly is important for getting
        the best out of your model.
        """
        if substring not in self.target.text_prompt:
            raise ValueError(f"Substring {substring} could not be found in {self.target.text_prompt}")

        try:
            token_ids = self.tokenizer.encode(substring, add_special_tokens=False)
            return self.target_token_ids.index(token_ids[0]), token_ids
        except ValueError:
            token_ids = self.tokenizer.encode(" " + substring, add_special_tokens=False)
            return self.target_token_ids.index(token_ids[0]), token_ids

    @property
    def n_layers(self) -> int:
        return self.n_layers_target

    @property
    def n_layers_source(self) -> int:
        return len(getattr(getattr(self.source_model, self.MODEL_TARGET), self.LAYER_TARGET))

    @property
    def n_layers_target(self) -> int:
        return len(getattr(getattr(self.target_model, self.MODEL_TARGET), self.LAYER_TARGET))

    def compute_precision_at_1(self, estimated_probs: torch.Tensor, true_token_index):
        """
        Compute Precision@1 metric. From the outputs of the target (patched) model
        (estimated_probs) against the output of the source model, aka the 'true' token.
        Args:
        - estimated_probs: The estimated probabilities for each token as a torch.Tensor.
        - true_token_index: The index of the true token in the vocabulary.
        Returns:
        - precision_at_1: Precision@1 metric result.

        This is the evaluation method of the token identity from patchscopes: https://arxiv.org/abs/2401.06102
        Its used for running an evaluation over large datasets.
        """
        predicted_token_index = torch.argmax(estimated_probs)
        precision_at_1 = 1 if predicted_token_index == true_token_index else 0
        return precision_at_1

    def compute_surprisal(self, estimated_probs: torch.Tensor, true_token_index):
        """
        Compute Surprisal metric. From the outputs of the target (patched) model
        (estimated_probs) against the output of the source model, aka the 'true' token.
        Args:
        - estimated_probs: The estimated probabilities for each token as a torch.Tensor.
        - true_token_index: The index of the true token in the vocabulary.
        Returns:
        - surprisal: Surprisal metric result.
        """
        # For now, just compute the surprisal of the first element. We'll need to improve this.
        if isinstance(true_token_index, list):
            true_token_index = true_token_index[0]
        # To avoid log(0) issues, add a small constant to the probabilities
        estimated_probs = torch.clamp(estimated_probs, min=1e-12)
        surprisal = -torch.log(estimated_probs[true_token_index])
        return surprisal.item()


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

        self.generation_kwargs = ModelLoader.generation_kwargs(
            self.target.model_name,
            self.target.max_new_tokens,
        )

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
        with self.source_model.trace(self.source.text_prompt, remote=self.REMOTE) as _:
            if self.source.soft_prompt is not None:
                # TODO: validate this with non GPT2 & GPTJ models
                self.source_model.transformer.wte.output = self.source.soft_prompt

            self._source_hidden_state = self.manipulate_source().save()
            self.source_output = self.source_model.lm_head.output[0].save()

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
            self.target.text_prompt,
            remote=self.REMOTE,
            **self.generation_kwargs,
        ) as _:
            if self.target.soft_prompt is not None:
                # Not sure if this works with mamba and other models
                self.target_model.transformer.wte.output = self.source.soft_prompt

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
