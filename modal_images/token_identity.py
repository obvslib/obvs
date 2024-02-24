from __future__ import annotations

import time

import numpy as np
import torch
from tqdm import tqdm

from obvspython.vis import create_heatmap, plot_surprisal

from modal import Stub, gpu, method
from modal_images.gemma import image as gemma_image
from modal_images.mistral import image as mistral_image


# Define the model names for LLaMA-2, Mistral, and GPT-2
model_names = {
    "llamatiny": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "llama": "meta-llama/Llama-2-13b-hf",
    "gpt2": "gpt2",
    "mamba": "MrGonao/delphi-mamba-100k",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "gptj": "EleutherAI/gpt-j-6B",
    "gemma2": "google/gemma-2b",
    "gemma7": "google/gemma-7b"
}

images = {
    "gemma2": gemma_image,
    "gemma7": gemma_image,
    "mistral": mistral_image,
    "gpt2": gemma_image,
}


stub = Stub(image=gemma_image, name="token_identity")


@stub.cls(
    gpu=gpu.A100(memory=80, count=1),
    timeout=60 * 60 * 2,
    container_idle_timeout=60 * 5,
)
class Runner:
    def setup(self, prompt, model, word):
        from obvspython.patchscope import Patchscope, SourceContext, TargetContext

        print("Starting Setup")

        # Setup source and target context with the simplest configuration
        source_context = SourceContext(
            prompt=prompt,  # Example input text
            model_name=model,  # Model name
            position=-1,
            device="cuda",
        )

        target_context = TargetContext.from_source(source_context)
        target_context.prompt = (
            "bat is bat; 135 is 135; hello is hello; black is black; shoe is shoe; x is"
        )
        target_context.max_new_tokens = 1
        self.patchscope = Patchscope(source=source_context, target=target_context)

        try:
            self.patchscope.source.position, target_tokens = self.patchscope.source_position_tokens(word)
            self.patchscope.target.position, _ = self.patchscope.target_position_tokens("X")

            assert (
                self.patchscope.source_words[self.patchscope.source.position].strip() == word
            ), self.patchscope.source_words[self.patchscope.source.position]
            assert (
                self.patchscope.target_words[self.patchscope.target.position].strip() == "X"
            ), self.patchscope.target_words[self.patchscope.target.position]

        except ValueError:
            self.target_tokens = self.patchscope.tokenizer.encode(" boat", add_special_tokens=False)
            self.patchscope.source.position = -1
            self.patchscope.target.position = -1

        self.values = np.zeros((self.patchscope.n_layers_source, self.patchscope.n_layers_target))

        print("Setup complete")

    @method()
    def run(self, prompt, model, word):
        self.setup(prompt, model, word)
        print("Running")
        source_layers = list(range(self.patchscope.n_layers))
        target_layers = list(range(self.patchscope.n_layers))

        outputs = self.patchscope.over_pairs(source_layers, target_layers)

        outputs = self.patchscope.over(source_layers, target_layers)

        print("Computing surprisal")
        target_output = 0

        # for i in source_layers:
        #     # Get the output of the run
        #     probs = torch.softmax(outputs[i][target_output], dim=-1)
        #     self.values[i] = self.patchscope.compute_surprisal(probs[-1], self.target_tokens)

        for i in source_layers:
            for j in target_layers:
                # Get the output of the run
                probs = torch.softmax(outputs[i][j][target_output].value, dim=-1)
                self.values[i, j] = self.patchscope.compute_surprisal(probs[-1], self.target_tokens)

        print("Done")
        print(type(self.values))
        print(type(self.values[0]))
        print(type(self.values[0][0]))
        return source_layers, target_layers, self.values


@stub.local_entrypoint()
def main(
    word: str = "boat",
    model: str = "gemma7",
    prompt: str = "if its on the road, its a car. if its in the air, its a plane. if its on the sea, its a",
):
    print(f"Generating definition for word: {word} using model: {model}")
    if model in model_names:
        model = model_names[model]

    model_name = model.replace("/", "-")
    filename = f"{model_name}_{word}"

    start = time.time()
    runner = Runner()
    source_layers, target_layers, values = runner.run.remote(prompt, model, word)
    print(f"Elapsed time: {time.time() - start:.2f}s. Layers: {source_layers}, {target_layers}")

    # Save the values to a file
    np.save(f"scripts/{filename}.npy", values)

    # fig = plot_surprisal(source_layers, [value[0] for value in values], title=f"Token Identity: Surprisal by Layer {model_name}")
    # fig.write_image(f"scripts/{filename}.png")
    # fig.show()

    fig = create_heatmap(source_layers, target_layers, values, title=f"Token Identity: Surprisal by Layer {model_name}")
    # Save as png
    fig.write_image(f"scripts/{filename}.png")
    fig.show()
