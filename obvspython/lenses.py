"""
lenses.py

Implementation of some widely-known lenses in the Patchscope framework
"""

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

        # create SourceContext, leave position and layer as default for now
        source_context = SourceContext(prompt=prompt, model_name=model, device=device)

        # create TargetContext from SourceContext, as they are mostly equal
        target_context = TargetContext.from_source(source_context)
        target_context.layer = -1  # for the logit-lens, the target layer is always the last layer

        # create Patchscope object
        self.patchscope = Patchscope(source_context, target_context)


