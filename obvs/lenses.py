"""
lenses.py
Implementation of some widely-known lenses in the Patchscope framework
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from obvs.logging import logger
from obvs.metrics import PrecisionAtKMetric, SurprisalMetric
from obvs.patchscope import Patchscope, SourceContext, TargetContext
from obvs.vis import create_heatmap, plot_surprisal


class TokenIdentity:
    """Implementation of token identiy patchscope.
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
        source_phrase: str | None = None,
        device: str | None = None,
        target_prompt: str | None = None,
        filename: str | None = None,
    ):
        logger.info(
            f"Starting token identity patchscope with source: {source_prompt} and target: {target_prompt}",
        )
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
        self._patchscope = Patchscope(source=source_context, target=target_context)

        # We find the source position and the expected output This uses the model tokenizer,
        # so it has to be done after the patchscope is created
        if source_phrase:
            self._patchscope.source.position = self._patchscope.find_in_source(source_phrase)

    @property
    def prompt(self):
        return self._patchscope.source.prompt

    @prompt.setter
    def prompt(self, value):
        self._patchscope.source.prompt = value

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        self._filename = Path(value).expanduser()

    def run(
        self,
        source_layers: Sequence[int] | None = None,
        target_layers: Sequence[int] | None = None,
    ):
        self.source_layers = source_layers or list(range(self._patchscope.n_layers_source))
        if target_layers:
            self.target_layers = target_layers
            # If there are two sets of layers, run over all of them in nested for loops
            self.outputs = list(self._patchscope.over(self.source_layers, self.target_layers))
        else:
            # Otherwise, run over the same set of layers
            self.outputs = list(self._patchscope.over_pairs(self.source_layers, self.source_layers))

        return self

    def compute_surprisal(self, word: str | None = None):
        target = self._target_word(word)
        self.prepare_data_array()

        logger.info(f"Computing surprisal of target tokens: {target} from word {word}")

        if hasattr(self, "source_layers") and hasattr(self, "target_layers"):
            for i, source_layer in enumerate(self.source_layers):
                for j, target_layer in enumerate(self.target_layers):
                    logits = self.outputs[i * len(self.target_layers) + j]
                    self.surprisal[i, j] = SurprisalMetric.batch(logits, target)
                    self.precision_at_1[i, j] = PrecisionAtKMetric.batch(logits, target, 1)
        elif hasattr(self, "source_layers"):
            for i, output in enumerate(self.outputs):
                self.surprisal[i] = SurprisalMetric.batch(output, target)
                self.precision_at_1[i] = PrecisionAtKMetric.batch(output, target, 1)
        logger.info("Done")

        self.save_to_file()

        return self

    def run_and_compute(
        self,
        source_layers: Sequence[int] | None = None,
        target_layers: Sequence[int] | None = None,
        word: str | None = None,
    ):
        """
        For larger models, saving the outputs for every layer eats up the GPU memoery. This method
        runs the patchscope and computes the surprisal in one go, saving memory.
        """
        self.source_layers = source_layers or list(range(self._patchscope.n_layers_source))
        if target_layers:
            self.target_layers = target_layers
            # If there are two sets of layers, run over all of them in nested for loops
            self.outputs = self._patchscope.over(self.source_layers, self.target_layers)
        else:
            # Otherwise, run over the same set of layers
            self.outputs = self._patchscope.over_pairs(self.source_layers, self.source_layers)

        self.prepare_data_array()

        if hasattr(self, "source_layers") and hasattr(self, "target_layers"):
            for i, source_layer in enumerate(self.source_layers):
                for j, target_layer in enumerate(self.target_layers):
                    self._nextloop(next(self.outputs), word, i, j)
        elif hasattr(self, "source_layers"):
            for i, output in enumerate(self.outputs):
                self._nextloop(output, word, i, None)

        return self

    def _nextloop(self, output, word, i=None, j=None):
        target = self._target_word(word)

        logger.info(f"Computing surprisal of target tokens: {target} from word {word}")

        if i is not None and j is not None:
            self.surprisal[i, j] = SurprisalMetric.batch(output, target)
            self.precision_at_1[i, j] = PrecisionAtKMetric.batch(output, target, 1)
        else:
            self.surprisal[i] = SurprisalMetric.batch(output, target)
            self.precision_at_1[i] = PrecisionAtKMetric.batch(output, target, 1)
        logger.info("Done")

        self.save_to_file()

    def _target_word(self, word):
        if isinstance(word, str):
            if not word.startswith(" ") and "gpt" in self._patchscope.model_name:
                # Note to devs: we probably want some tokenizer helpers for this kind of thing
                logger.warning("Target should probably start with a space!")
            target = self._patchscope.tokenizer.encode(word)
        else:
            # Otherwise, we find the next token from the source output:
            target = self._patchscope.source_output[-1].argmax(dim=-1).item()
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
