"""
lenses.py
Implementation of some widely-known lenses in the Patchscope framework
"""

from pathlib import Path
from typing import Optional, Sequence

from obvs.patchscope import SourceContext, TargetContext, Patchscope
from obvs.vis import plot_surprisal
from obvs.logging import logger

import numpy as np
import torch


class TokenIdentity(Patchscope):
    """ Implementation of token identiy patchscope.
        The logit-lens is defined in the patchscope framework as follows:
        Target prompt is an identity prompt:
            T = "bat is bat; 135 is 135; hello is hello; black is black; shoe is shoe; x is"
        Model is the same as the source model (NB, this can be changed):
            M = M*  (source model = target model)
        Target layer is equal to the source layer:
            l* = l* (target layer = last layer)
        Source position is specified by the user, target position is -1:
            i = i*  (source position = target position)
        Mapping is the identity function:
            f = id  (mapping = identity function)
    """
    IDENTITIY_PROMPT = "bat is bat; 135 is 135; hello is hello; black is black; shoe is shoe; x is"

    def __init__(
            self,
            source_prompt: str,
            model_name: str = "gpt2",
            source_phrase: Optional[str] = None,
            device: Optional[str] = None,
            target_prompt: Optional[str] = None,
            filename: Optional[str] = None
    ):
        logger.info(f"Starting token identity patchscope with source: {source_prompt} and target: {target_prompt}")
        if filename:
            logger.info(f"Saving to file: {filename}")
            self.filename = Path(filename).expanduser()
        else:
            self.filename = None
        self.model_name = model_name
        self.prompt = source_prompt

        # Source finds the device automatically, but we can override it
        source_context = SourceContext(prompt=source_prompt, model_name=model_name, position=-1)
        source_context.device = device or source_context.device

        # Target context is the same as the source context in most cases
        target_context = TargetContext.from_source(source_context, max_new_tokens=1)
        target_context.prompt = target_prompt or self.IDENTITIY_PROMPT

        # Setup our patchscope
        self.patchscope = Patchscope(source=source_context, target=target_context)

        # We find the source position and the expected output This uses the model tokenizer,
        # so it has to be done after the patchscope is created
        if source_phrase:
            self.patchscope.source.position = self.patchscope.find_in_source(source_phrase)

    def run(self, layers: Optional[Sequence[int]] = None):
        self.layers = layers or list(range(self.patchscope.n_layers))
        self.outputs = self.patchscope.over_pairs(self.layers, self.layers)

        return self

    def compute_surprisal(self, word):
        self.expected = word
        if isinstance(word, str):
            if not word.startswith(" ") and "gpt" in self.patchscope.model_name:
                # Note to devs: we probably want some tokenizer helpers for this kind of thing
                logger.warning("Target should probably start with a space!")
            target = self.patchscope.tokenizer.encode(word)
        else:
            target = word
        logger.info(f"Computing surprisal of target tokens: {target} from word {word}")
        self.surprisal = np.zeros(len(self.layers))
        for i, output in enumerate(self.outputs):
            probs = torch.softmax(output[0], dim=-1)
            self.surprisal[i] = self.patchscope.compute_surprisal(probs[-1], target)
        logger.info("Done")

        if self.filename:
            np.save(self.filename.with_suffix(".npy"), self.surprisal)

        return self

    def visualize(self):
        if self.surprisal is None:
            raise ValueError("You need to compute the surprisal values first.")

        # Visualize the surprisal values
        self.fig = plot_surprisal(
            self.layers,
            self.surprisal,
            title=f"Token Identity: Surprisal by Layer {self.model_name} Prompt: {self.prompt[-30:]}, Target: {self.expected}",
        )

        if self.filename:
            self.fig.write_image(self.filename.with_suffix(".png"))
        self.fig.show()


class ExtendedTokenIdentity(Patchscope):
    """ Implementation of extended token identiy patchscope.
        In order to support mutli-token phrases, we can extend the token identity patchscope.
        We can also loosen the constraint that the source and target layers are the same to
        check which are the best pairings.

        We also allow for patching into a specified part of the target phrase, instead of the end.
    """

    def __init__(
            self,
            source_prompt: str,
            model_name: str,
            source_phrase: str,
            device: Optional[str] = None,
            target_prompt: Optional[str] = None,
            target_phrase: Optional[str] = None,
    ):
        pass

    def run(self, source_layers: Optional[Sequence[int]] = None, target_layers: Optional[Sequence[int]] = None):
        self.source_layers = source_layers or list(range(self.patchscope.n_layers))
        self.target_layers = target_layers or list(range(self.patchscope.n_layers))

        self.outputs = self.patchscope.over_pairs(self.source_layers, self.target_layers)

        return self
