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
from typing import Any

import torch
from nnsight import LanguageModel
from nnsight.contexts import Invoker

from obvspython.patchscope_base import PatchscopeBase


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

    def __repr__(self):
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
    ):
        return TargetContext(
            prompt=source.prompt,
            position=source.position,
            model_name=source.model_name,
            layer=source.layer,
            mapping_function=mapping_function or (lambda x: x),
            max_new_tokens=max_new_tokens,
            device=source.device,
        )

    def __repr__(self):
        return (
            f"TargetContext(prompt={self.prompt}, position={self.position}, "
            f"model_name={self.model_name}, layer={self.layer}, device={self.device}, "
            f"max_new_tokens={self.max_new_tokens}, "
            f"mapping_function={self.mapping_function})"
        )


@dataclass
class Patchscope(PatchscopeBase):
    source: SourceContext
    target: TargetContext
    source_model: LanguageModel = field(init=False)
    target_model: LanguageModel = field(init=False)

    tokenizer: Any = field(init=False)

    REMOTE: bool = False

    _source_hidden_state: torch.Tensor = field(init=False)
    _mapped_hidden_state: torch.Tensor = field(init=False)
    _target_outputs: list[torch.Tensor] = field(init=False, default_factory=list)

    def __post_init__(self):
        # Load models
        self.source_model = LanguageModel(self.source.model_name, device_map=self.source.device)
        if (
            self.source.model_name == self.target.model_name
            and self.source.device == self.target.device
        ):
            self.target_model = self.source_model
        else:
            self.target_model = LanguageModel(self.target.model_name, device_map=self.target.device)

        self.tokenizer = self.source_model.tokenizer
        self.init_positions()

    def source_forward_pass(self) -> None:
        """
        Get the source representation.
        """
        self._source_hidden_state = self._source_forward_pass(self.source)

    def _source_forward_pass(self, source: SourceContext):
        """
        Get the requested hidden states from the foward pass of the model.

        We use the 'forward' context so we can add the REMOTE option.

        For each architecture, you need to know the name of the layers.
        """
        with self.source_model.forward(remote=self.REMOTE) as runner:
            with runner.invoke(source.prompt) as _:
                if "gpt2" in self.source.model_name:
                    return self._gpt_source_invoker(source)
                elif "lama" in self.source.model_name:
                    return self._llama2_source_invoker(source)
                else:
                    raise ValueError(f"Model {self.source.model_name} not supported")

    def _gpt_source_invoker(self, source: SourceContext):
        """
        Get the hidden state from any GPT2 model
        """
        return (
            self.source_model.transformer.h[source.layer].output[0][:, source.position, :]
        ).save()

    def _llama2_source_invoker(self, source: SourceContext):
        """
        Get the hidden state from any Llama2 model
        """
        return (
            self.source_model.model.layers[source.layer].output[0][:, source.position, :]
        ).save()

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
            remote=self.REMOTE,
            max_new_tokens=self.target.max_new_tokens,
        ) as runner:
            with runner.invoke(self.target.prompt) as invoker:
                if "gpt2" in self.source.model_name:
                    self._gpt_target_invoker(invoker)
                elif "lama" in self.source.model_name:
                    self._llama2_target_invoker(invoker)
                else:
                    raise ValueError(f"Model {self.target.model_name} not supported")

    def _gpt_target_invoker(self, invoker: Invoker.Invoker):
        """
        Patch the target representation for GPT2 models and save the output
        """
        (
            self.target_model.transformer.h[self.target.layer].output[0][:, self.target.position, :]
        ) = self._mapped_hidden_state.value

        for generation in range(self.target.max_new_tokens):
            self._target_outputs.append(self.target_model.lm_head.output[0].save())
            invoker.next()

    def _llama2_target_invoker(self, invoker: Invoker.Invoker):
        """
        Patch the target representation for Llama2 models and save the output
        """
        (
            self.target_model.model.layers[self.target.layer].output[0][:, self.target.position, :]
        ) = self._mapped_hidden_state.value

        for generation in range(self.target.max_new_tokens):
            self._target_outputs.append(self.target_model.lm_head.output[0].save())
            invoker.next()

    def run(self) -> None:
        """
        Run the patchscope
        """
        self._target_outputs = []
        self.source_forward_pass()
        self.map()
        self.target_forward_pass()
