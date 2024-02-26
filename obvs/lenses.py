"""
lenses.py
Implementation of some widely-known lenses in the Patchscope framework
"""

from pathlib import Path
from typing import Optional, Sequence

from obvs.patchscope import SourceContext, TargetContext, Patchscope
from obvs.vis import plot_surprisal, create_heatmap
from obvs.logging import logger

import numpy as np
import torch
import gc


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
            self._filename = Path(filename).expanduser()
        else:
            self._filename = None
        self.model_name = model_name

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

    @property
    def prompt(self):
        return self.patchscope.source.prompt

    @prompt.setter
    def prompt(self, value):
        self.patchscope.source.prompt = value

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        self._filename = Path(value).expanduser()

    def run(self, source_layers: Optional[Sequence[int]] = None, target_layers: Optional[Sequence[int]] = None):
        self.source_layers = source_layers or list(range(self.patchscope.n_layers_source))
        if target_layers:
            self.target_layers = target_layers
            # If there are two sets of layers, run over all of them in nested for loops
            self.outputs = list(self.patchscope.over(self.source_layers, self.target_layers))
        else:
            # Otherwise, run over the same set of layers
            self.outputs = list(self.patchscope.over_pairs(self.source_layers, self.source_layers))

        return self

    def compute_surprisal(self, word: Optional[str] = None):
        target = self._target_word(word)
        self.prepare_data_array()

        logger.info(f"Computing surprisal of target tokens: {target} from word {word}")

        gc.collect()

        if hasattr(self, "source_layers") and hasattr(self, "target_layers"):
            for i, source_layer in enumerate(self.source_layers):
                for j, target_layer in enumerate(self.target_layers):
                    probs = torch.softmax(self.outputs[i * len(self.target_layers) + j], dim=-1)
                    self.surprisal[i, j] = self.patchscope.compute_surprisal(probs, target)
                    self.precision_at_1[i, j] = self.patchscope.compute_precision_at_1(probs, target)
        elif hasattr(self, "source_layers"):
            for i, output in enumerate(self.outputs):
                probs = torch.softmax(output, dim=-1)
                self.surprisal[i] = self.patchscope.compute_surprisal(probs, target)
                self.precision_at_1[i] = self.patchscope.compute_precision_at_1(probs, target)
        logger.info("Done")

        self.save_to_file()

        return self

    def run_and_compute(
            self,
            source_layers: Optional[Sequence[int]] = None,
            target_layers: Optional[Sequence[int]] = None,
            word: Optional[str] = None
    ):
        """
        For larger models, saving the outputs for every layer eats up the GPU memoery. This method
        runs the patchscope and computes the surprisal in one go, saving memory.
        """
        self.source_layers = source_layers or list(range(self.patchscope.n_layers_source))
        if target_layers:
            self.target_layers = target_layers
            # If there are two sets of layers, run over all of them in nested for loops
            self.outputs = self.patchscope.over(self.source_layers, self.target_layers)
        else:
            # Otherwise, run over the same set of layers
            self.outputs = self.patchscope.over_pairs(self.source_layers, self.source_layers)

        self.prepare_data_array()

        # Get the first output to initialize the state of the patchscope
        if hasattr(self, "source_layers") and hasattr(self, "target_layers"):
            for i, source_layer in enumerate(self.source_layers):
                for j, target_layer in enumerate(self.target_layers):
                    self.nextloop(next(self.outputs), word, i, j)
        elif hasattr(self, "source_layers"):
            for i, output in enumerate(self.outputs):
                self.nextloop(next(self.outputs), word, i, None)

        return self

    def nextloop(self, output, word, i=None, j=None):
        target = self._target_word(word)

        logger.info(f"Computing surprisal of target tokens: {target} from word {word}")

        gc.collect()

        probs = torch.softmax(output, dim=-1)
        if i is not None and j is not None:
            self.surprisal[i, j] = self.patchscope.compute_surprisal(probs, target)
            self.precision_at_1[i, j] = self.patchscope.compute_precision_at_1(probs, target)
        else:
            self.surprisal[i] = self.patchscope.compute_surprisal(probs, target)
            self.precision_at_1[i] = self.patchscope.compute_precision_at_1(probs, target)
        logger.info("Done")

        self.save_to_file()

    def _target_word(self, word):
        if isinstance(word, str):
            if not word.startswith(" ") and "gpt" in self.patchscope.model_name:
                # Note to devs: we probably want some tokenizer helpers for this kind of thing
                logger.warning("Target should probably start with a space!")
            target = self.patchscope.tokenizer.encode(word)
        else:
            # Otherwise, we find the next token from the source output:
            target = self.patchscope.source_output[-1].argmax(dim=-1).item()
        return target

    def prepare_data_array(self):
        if hasattr(self, "source_layers") and hasattr(self, "target_layers"):
            self.surprisal = np.zeros((len(self.source_layers), len(self.target_layers)))
            self.precision_at_1 = np.zeros((len(self.source_layers), len(self.target_layers)))
        elif hasattr(self, "source_layers"):
            self.surprisal = np.zeros(len(self.source_layers))
            self.precision_at_1 = np.zeros(len(self.source_layers))

    def save_to_file(self):
        if self.filename:
            surprisal_file = Path(self.filename.stem + "_surprisal").with_suffix(".npy")
            np.save(surprisal_file, self.surprisal)
            precision_file = Path(self.filename.stem + "_precision_at_1").with_suffix(".npy")
            np.save(precision_file, self.precision_at_1)

    def visualize(self, show: bool = True):
        if not hasattr(self, "surprisal"):
            raise ValueError("You need to compute the surprisal values first.")

        # Visualize the surprisal values
        if hasattr(self, "source_layers") and hasattr(self, "target_layers"):
            self.fig = create_heatmap(
                self.source_layers,
                self.target_layers,
                self.surprisal,
                title=f"Token Identity: Surprisal by Layer {self.model_name} Prompt: {self.prompt[-30:]}",
            )
        elif hasattr(self, "source_layers"):
            self.fig = plot_surprisal(
                self.source_layers,
                self.surprisal,
                title=f"Token Identity: Surprisal by Layer {self.model_name} Prompt: {self.prompt[-30:]}",
            )

        if self.filename:
            self.fig.write_image(self.filename.with_suffix(".png"))
        if show:
            self.fig.show()
        return self


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
