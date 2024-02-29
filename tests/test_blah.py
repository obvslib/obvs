from __future__ import annotations

import torch

from obvs.patchscope import ModelLoader, Patchscope, SourceContext, TargetContext


MODEL="EleutherAI/gpt-j-6B"
DEVICE="cuda:0"

def run(layers, positions):
    outputs = dict()
    for layer in layers:
        for pos in positions: # marty mcfly from has 5 tokens
            print(f"layer = {layer}, pos = {pos}")
            source_context = SourceContext(
                device=DEVICE,
                prompt="Marty McFly from",
                model_name=MODEL,
                position=pos,
                layer=layer,
            )

            target_context = TargetContext(
                device=DEVICE,
                embedding=torch.load('data/processed/gptj_soft_prompt.pt'),
                model_name=MODEL,
                max_new_tokens=4,
                position=-1,
                layer=layer,
            )

            patchscope = Patchscope(source_context, target_context)

            patchscope.run()

            output = patchscope.full_output_words()
            print(f"{(layer, pos)}: {output}")

            outputs[(layer, pos)] = output

    for k, v in sorted(outputs.items()):
        print(f"{k}: {v}")
