from dataclasses import dataclass, field
from typing import Callable, Sequence, Optional

from transformer_lens import HookedTransformer, ActivationCache
import torch



@dataclass
class SourceContext:
    """
    Source context for the patchscope
    """
    prompt: Sequence[str] = ""
    position: int = 0
    model_name: str = "gpt2"
    layer: int = 0
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
    Parameters identical to the source context, with the addition of a mapping function
    """
    mapping_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x

    @staticmethod
    def from_source(
            source: SourceContext,
            mapping_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ):
        return TargetContext(
            prompt=source.prompt,
            position=source.position,
            model_name=source.model_name,
            layer=source.layer,
            mapping_function=mapping_function or (lambda x: x),
            device=source.device
        )

    def __repr__(self):
        return (
            f"TargetContext(prompt={self.prompt}, position={self.position}, "
            f"model_name={self.model_name}, layer={self.layer}, device={self.device}, "
            f"mapping_function={self.mapping_function})"
        )


@dataclass
class Patchscope:
    source: SourceContext
    target: TargetContext
    source_model: HookedTransformer = field(init=False)
    target_model: HookedTransformer = field(init=False)

    batch_size: int = 0

    _source_hidden_state: torch.Tensor = field(init=False)
    _source_cache: ActivationCache = field(init=False)
    _target_logits: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.source_model = HookedTransformer.from_pretrained(self.source.model_name)
        self.target_model = HookedTransformer.from_pretrained(self.target.model_name)

        # If the source or target use '-1' for the layer, we need to find the number
        # of layers in the model and set the layer to the last layer
        if self.source.layer == -1:
            self.source.layer = self.source_model.cfg.n_layers - 1
        if self.target.layer == -1:
            self.target.layer = self.target_model.cfg.n_layers - 1

    def source_forward_pass(self):
        """
        Run the source model on the prompt and cache all the activations
        """
        _, source_cache = self.source_model.run_with_cache(self.source.prompt)
        return source_cache

    def map(self):
        """
        Apply the mapping function to the source representation
        """
        self._source_hidden_state = self.target.mapping_function(self._source_hidden_state)

    def target_forward_pass(self):
        """
        Run the target model with the mapped source representation
        """
        #def hook_fn(
        #    target_activations: Float[Tensor, '...'],
        #    hook: HookPoint
        #) -> Float[Tensor, '...']:
        #    target_activations[self.batch_size, self.target.position, :] = self._source_hidden_state
        #    return target_activations

        #self.target_model.add_hook(
        #    get_act_name("resid_pre", self.target.layer),
        #    hook_fn,
        #)

        return self.target_model.forward(self.target.prompt, return_type="logits")


def main():

    prompt = "John and Mary are walking together. John said to"

    # Setup source and target context with the simplest configuration
    source_context = SourceContext(
        prompt=prompt,  # Example input text
        model_name="gpt2",
        position=-1,  # Last token (assuming single input)
        layer=0,  # Logit lense will run on each layer.
        device="cpu"
    )

    target_context = TargetContext.from_source(source_context)
    target_context.layer = -1  # Last layer (logit lens)

    print(source_context)
    print(target_context)

    patchscope = Patchscope(source=source_context, target=target_context)
    patchscope.source_forward_pass()
    patchscope.target_forward_pass()


if __name__ == '__main__':
    main()