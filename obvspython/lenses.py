"""
lenses.py

Implementation of some widely-known lenses in the Patchscope framework
"""

from typing import List
import torch
from obvspython.logging import logger
from obvspython.vis import create_annotated_heatmap
from obvspython.patchscope import SourceContext, TargetContext, Patchscope


class LogitLens(Patchscope):
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
        target_context.layer = -1  # for the logit-lens, the target layer is always the last layer

        # create Patchscope object
        self.patchscope = Patchscope(source_context, target_context)

    def create_top_token_logit_pred_heatmap(self, substring: str, layers: List[int]):
        """ Create a heatmap of the top predicted token and its logit for each layer in layers
            and each token in substring.
            Based on the plot from:
            https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

        Args:
            substring (str): Substring of the prompt for which the top prediction and logits
                should be calculated.
            layers (List[int]): Indices of Transformer Layers for which the lens should be applied
        """

        # get the starting position of the substring in the prompt
        try:
            start_pos, substring_tokens = self.patchscope.source_position_tokens(substring)
        except ValueError:
            logger.error('The substring "%s" could not be found in the prompt "%s"', substring,
                         self.source.prompt)
            return

        logits = []
        preds = []
        x_ticks = [f'{self.patchscope.tokenizer.decode(tok)}' for tok in substring_tokens]
        y_ticks = [f'{self.patchscope.MODEL_SOURCE}_{self.patchscope.LAYER_SOURCE}{i}'
                   for i in layers]

        # for each token in the substring,
        for i, layer in enumerate(layers):
            layer_logits = []
            layer_preds = []
            for j in range(len(substring_tokens)):

                self.patchscope.source.layer = layer
                self.patchscope.source.position = start_pos + j
                self.patchscope.target.position = start_pos + j
                self.patchscope.run()

                # get the top prediction and logit
                top_logit, top_pred_idx = torch.max(self.patchscope.logits()[start_pos + j], dim=0)
                # convert the top_pred_idx to a word
                top_pred = self.patchscope.tokenizer.decode(top_pred_idx)

                layer_logits.append(top_logit.item())
                layer_preds.append(top_pred)

            logits.append(layer_logits)
            preds.append(layer_preds)

        # create a heatmap with the top logits and predicted tokens
        fig = create_annotated_heatmap(logits, preds, x_ticks, y_ticks,
                                       title='Top predicted token and its logit')
        fig.write_html(f'{self.model_name}_logitlens_top_logits_preds.html')