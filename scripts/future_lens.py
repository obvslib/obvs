from __future__ import annotations

import os

import torch

from obvs.patchscope import Patchscope, SourceContext, TargetContext

MODEL_NAME = "EleutherAI/gpt-j-6b"
PREFIX_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "processed",
    "gptj_soft_prefix.pt",
)
DEVICE = "auto"


def future_lens():
    """
    Simple script to generate the future lens at a single position.
    """

    soft_prompt = torch.load(PREFIX_PATH).to(torch.float32).detach()

    source = SourceContext(
        prompt="Marty McFly from",
        layer=-1,
        position=-1,
        model_name=MODEL_NAME,
        device=DEVICE,
    )

    target = TargetContext(
        prompt=soft_prompt[None, :],
        layer=-1,
        position=-1,
        model_name=MODEL_NAME,
        device=DEVICE,
        max_new_tokens=4,
    )

    # Might need GPU to load gptj
    patchscope = Patchscope(source, target)

    patchscope.run()

    # Expect to see "Back to the future" in the output
    print(patchscope.full_output_tokens())


if __name__ == "__main__":
    future_lens()
