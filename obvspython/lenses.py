"""
lenses.py

Implementation of some widely-known lenses
"""

from typing import List, Tuple
import torch
from plotly.graph_objects import Figure
from obvspython.logging import logger
from obvspython.vis import create_annotated_heatmap
from obvspython.patchscope import SourceContext, TargetContext, Patchscope


class BaseLogitLens:
    """ Parent class for LogitLenses.
        Patchscope and classic logit-lens are run differently,
        but share the same visualization.
     """

    def __init__(self, model: str, prompt: str, device: str):
        """ Constructor. Setup a Patchscope object with Source and Target context.
            The target context is equal to the source context, apart from the layer.

        Args:
            model (str): Name of the model. Must be a valid name for huggingface transformers
                package
            prompt (str): The prompt to be analyzed
            device (str): Device on which the model should be run: e.g. cpu, auto
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
        self.data = {}

    def visualize(self, kind: str = 'top_logits_preds', file_name: str = '') -> Figure:
        """ Visualize the logit lens results in one of the following ways:
                top_logits_preds: Heatmap with the top predicted tokens and their logits
        Args:
              kind (str): The kind of visualization
              file_name (str): If provided, save figure to a file with the given name
                               (in the current path)
        Returns (plotly.graph_objects.Figure):
            The created figure
        """

        if not self.data:
            logger.error('You need to call .run() before .visualize()!')
            return Figure()

        if kind == 'top_logits_preds':

            logits = []
            preds = []

            # loop over all layers
            for i, layer in enumerate(self.data['layers']):
                layer_logits = []
                layer_preds = []

                # loop over every token in substring
                for j in range(len(self.data['substring_tokens'])):

                    # get the top prediction and logit
                    top_logit, top_pred_idx = torch.max(self.data['logits'][(i, j)], dim=0)
                    # convert the top_pred_idx to a word
                    top_pred = self.patchscope.tokenizer.decode(top_pred_idx)

                    layer_logits.append(top_logit.item())
                    layer_preds.append(top_pred)

                logits.append(layer_logits)
                preds.append(layer_preds)

            x_ticks = [f'{self.patchscope.tokenizer.decode(tok)}'
                       for tok in self.data['substring_tokens']]
            y_ticks = [f'{self.patchscope.MODEL_SOURCE}_{self.patchscope.LAYER_SOURCE}{i}'
                       for i in self.data['layers']]

            # create a heatmap with the top logits and predicted tokens
            fig = create_annotated_heatmap(logits, preds, x_ticks, y_ticks,
                                           title='Top predicted token and its logit')

        if file_name:
            fig.write_html(f'{file_name.replace(".html", "")}.html')
        return fig


class PatchscopeLogitLens(BaseLogitLens):
    """ Implementation of logit-lens in patchscope framework.
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

    def run(self, substring: str, layers: List[int]):
        """ Run the logit lens for each layer in layers and each token in substring.

        Args:
            substring (str): Substring of the prompt for which the top prediction and logits
                should be calculated.
            layers (List[int]): Indices of Transformer Layers for which the lens should be applied
        """

        # get starting position and tokens of substring
        start_pos, substring_tokens = self.patchscope.source_position_tokens(substring)

        self.data['logits'] = {}

        # loop over each layer and token in substring
        for i, layer in enumerate(layers):
            for j in range(len(substring_tokens)):

                self.patchscope.source.layer = layer
                self.patchscope.source.position = start_pos + j
                self.patchscope.target.position = start_pos + j
                self.patchscope.run()

                self.data['logits'][(i, j)] = self.patchscope.logits()[start_pos + j]
        self.data['substring_tokens'] = substring_tokens
        self.data['layers'] = layers


class ClassicLogitLens(BaseLogitLens):
    """ Implementation of LogitLens in standard fashion.
        Run a forward pass on the model and multiply the output of a specific layer
        with the final layer norm and unembed to get the logits of that layer.
        For convenience, use methods from the Patchscope class.
    """

    def run(self, substring: str, layers: List[int]):
        """ Run the logit lens for each layer in layers and each token in substring.

        Args:
            substring (str): Substring of the prompt for which the top prediction and logits
                should be calculated.
            layers (List[int]): Indices of Transformer Layers for which the lens should be applied
        """

        # get starting position and tokens of substring
        start_pos, substring_tokens = self.patchscope.source_position_tokens(substring)

        self.data['logits'] = {}

        # loop over all layers
        for i, layer in enumerate(layers):

            # with one forward pass, we can get the logits of every position
            with self.patchscope.source_model.trace(self.patchscope.source.prompt) as _:

                # get the appropriate sub-module and block from source_model
                sub_mod = getattr(self.patchscope.source_model, self.patchscope.MODEL_SOURCE)
                block = getattr(sub_mod, self.patchscope.LAYER_SOURCE)

                # get hidden state after specified layer
                hidden = block[layer].output[0]

                # apply final layer norm and unembedding to hidden state
                ln_f_out = sub_mod.ln_f(hidden)
                logits = self.patchscope.source_model.lm_head(ln_f_out).save()

            # loop over all tokens in substring and get the corresponding logits
            for j in range(len(substring_tokens)):
                self.data['logits'][(i, j)] = logits[:, start_pos + j, :]
        self.data['substring_tokens'] = substring_tokens
        self.data['layers'] = layers
