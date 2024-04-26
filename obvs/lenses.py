"""
lenses.py

Implementation of some widely-known lenses in the Patchscope framework
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from plotly.graph_objects import Figure

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
            for i, _source_layer in enumerate(self.source_layers):
                for j, _target_layer in enumerate(self.target_layers):
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
            for i, _source_layer in enumerate(self.source_layers):
                for j, _target_layer in enumerate(self.target_layers):
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


class BaseLogitLens:
    """Parent class for LogitLenses.
    Patchscope and classic logit-lens are run differently,
    but share the same visualization.
    """

    def __init__(self, model: str, prompt: str, device: str, layers: list[int], substring: str):
        """Constructor. Setup a Patchscope object with Source and Target context.
            The target context is equal to the source context, apart from the layer.

        Args:
            model (str): Name of the model. Must be a valid name for huggingface transformers
                package
            prompt (str): The prompt to be analyzed
            device (str): Device on which the model should be run: e.g. cpu, auto
            layers (list[int]): Indices of Transformer Layers for which the lens should be applied
            substring (str): Substring of the prompt for which the top prediction and logits
                should be calculated
        """

        self.model_name = model

        # create SourceContext, leave position and layer as default for now
        source_context = SourceContext(prompt=prompt, model_name=model, device=device)

        # create TargetContext from SourceContext, as they are mostly equal
        target_context = TargetContext.from_source(source_context)
        # for the logit-lens, the target layer is always the last layer
        target_context.layer = -1

        # create Patchscope object
        self.patchscope = Patchscope(source_context, target_context)
        start_pos, substring_tokens = self.patchscope.source_position_tokens(substring)
        self.start_pos = start_pos
        self.layers = layers
        self.substring_tokens = substring_tokens
        self.data = {}

    def visualize(self, kind: str = "top_logits_preds", file_name: str = "") -> Figure:
        """Visualize the logit lens results in one of the following ways:
                top_logits_preds: Heatmap with the top predicted tokens and their logits
        Args:
              kind (str): The kind of visualization
              file_name (str): If provided, save figure to a file with the given name
                               (in the current path)
        Returns (plotly.graph_objects.Figure):
            The created figure
        """

        if not self.data:
            logger.error("You need to call .run() before .visualize()!")
            return Figure()

        if kind == "top_logits_preds":
            # get the top logits and corresponding tokens for each layer and token position
            top_logits, top_pred_idcs = torch.max(self.data["logits"], dim=-1)

            # create NxM list of strings from the top predictions
            top_preds = []

            # loop over the layer dimension in top_preds, get a list of predictions for
            # each position associated with that layer
            for i in range(top_pred_idcs.shape[0]):
                top_preds.append(self.patchscope.tokenizer.batch_decode(top_pred_idcs[i]))
            x_ticks = [
                f"{self.patchscope.tokenizer.decode(tok)}" for tok in self.data["substring_tokens"]
            ]
            y_ticks = [
                f"{self.patchscope.source_base_name}_{self.patchscope.source_layer_name}{i}"
                for i in self.data["layers"]
            ]

            # create a heatmap with the top logits and predicted tokens
            fig = create_heatmap(
                x_ticks,
                y_ticks,
                top_logits,
                cell_annotations=top_preds,
                title="Top predicted token and its logit",
            )
        if file_name:
            fig.write_html(f'{file_name.replace(".html", "")}.html')
        return fig

    def compute_surprisal_at_position(self):
        """
        Compute the surprisal for a specific position across all layers.
        """
        if "logits" not in self.data:
            raise ValueError("Logits data not found. Please run the lens first.")

        # Assuming target_token_ids is known and corresponds to the actual token ID at target_position
        # For this example, we assume a single target token ID for simplicity.
        # In a real scenario, this would be dynamic or calculated based on input sequence.
        target_token_id = (
            self.target_token_id
        )  # This should be set or calculated based on your specific use case.
        target_position = self.target_token_position
        logits_at_position = self.data["logits"][
            :,
            target_position,
            :,
        ]  # Shape: (n_layers, d_vocab)
        probabilities_at_position = torch.softmax(torch.tensor(logits_at_position), dim=-1)

        # Probability of the actual next token at the given position, for all layers
        actual_token_probabilities = probabilities_at_position[:, target_token_id]

        # Surprisal calculation: negative log probability
        surprisals = -torch.log(actual_token_probabilities)

        return surprisals.numpy()  # Convert to numpy array for convenience


class PatchscopeLogitLens(BaseLogitLens):
    """Implementation of logit-lens in patchscope framework.
    The logit-lens is defined in the patchscope framework as follows:
    S = T   (source prompt = target prompt)
    M = M*  (source model = target model)
    l* = L* (target layer = last layer)
    i = i*  (source position = target position)
    f = id  (mapping = identity function)

    The source layer l and position i can vary.

        In words: The logit-lens maps the hidden state at position i of layer l of the model M
            to the last layer of that same model. It is equal to taking the hidden state and
            applying unembed to it.
    """

    def __init__(self, model: str, prompt: str, device: str, layers: list[int], substring: str):
        """
        substring (str): Substring of the prompt for which the top prediction and logits should be calculated.
        layers (list[int]): Indices of Transformer Layers for which the lens should be applied
        """

        super().__init__(model, prompt, device, layers, substring)
        self.data["logits"] = torch.zeros(
            len(self.layers),
            len(self.substring_tokens),
            self.patchscope.source_model.lm_head.out_features,  # not vocab size, as gpt-j embedding dimension different to tokenizer vocab size
        )

    def run(self, position: int):
        """Run the logit lens for each layer in layers, for a specific position in the prompt.

        Args:
            position (int): Position in the prompt for which the lens should be applied
        """
        # get starting position and tokens of substring
        assert position < len(self.substring_tokens), "Position out of bounds!"

        # loop over each layer and token in substring
        for i, layer in enumerate(self.layers):
            self.patchscope.source.layer = layer
            self.patchscope.source.position = self.start_pos + position
            self.patchscope.target.position = self.start_pos + position
            self.patchscope.run()

            self.data["logits"][i, position, :] = self.patchscope.logits()[
                self.start_pos + position
            ].to("cpu")

            # empty CUDA cache to avoid filling of GPU memory
            torch.cuda.empty_cache()

        # detach logits, save tokens from substring and layer indices
        self.data["logits"] = self.data["logits"].detach()
        self.data["substring_tokens"] = self.substring_tokens
        self.data["layers"] = self.layers


class ClassicLogitLens(BaseLogitLens):
    """Implementation of LogitLens in standard fashion.
    Run a forward pass on the model and apply the final layer norm and unembed to the output of
    a specific layer to get the logits of that layer.
    For convenience, use methods from the Patchscope class.
    """

    def run(self, substring: str, layers: list[int]):
        """Run the logit lens for each layer in layers and each token in substring.

        Args:
            substring (str): Substring of the prompt for which the top prediction and logits
                should be calculated.
            layers (list[int]): Indices of Transformer Layers for which the lens should be applied
        """

        # get starting position and tokens of substring
        start_pos, substring_tokens = self.patchscope.source_position_tokens(substring)

        # initialize tensor for logits

        self.data["logits"] = torch.zeros(
            len(layers),
            len(substring_tokens),
            self.patchscope.source_model.lm_head.out_features,  # not vocab size, as gpt-j embedding dimension different to tokenizer vocab size
        )

        # loop over all layers
        for i, layer in enumerate(layers):
            # with one forward pass, we can get the logits of every position
            with self.patchscope.source_model.trace(self.patchscope.source.prompt) as _:
                # get the appropriate sub-module and block from source_model
                sub_mod = getattr(self.patchscope.source_model, self.patchscope.source_base_name)
                block = getattr(sub_mod, self.patchscope.source_layer_name)

                # get hidden state after specified layer
                hidden = block[layer].output[0]

                # apply final layer norm and unembedding to hidden state
                ln_f_out = sub_mod.ln_f(hidden)
                logits = self.patchscope.source_model.lm_head(ln_f_out).save()

            # loop over all tokens in substring and get the corresponding logits
            for j in range(len(substring_tokens)):
                self.data["logits"][i, j, :] = logits[0, start_pos + j, :].to("cpu")

            # empty CDUA cache to avoid filling of GPU memory
            torch.cuda.empty_cache()

        # detach logits, save tokens from substring and layer indices
        self.data["logits"] = self.data["logits"].detach()
        self.data["substring_tokens"] = substring_tokens
        self.data["layers"] = layers
