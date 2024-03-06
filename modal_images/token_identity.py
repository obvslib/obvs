from __future__ import annotations

from pathlib import Path

from modal import Secret, Stub, gpu, method

from modal_images.gemma import image as gemma_image
from modal_images.mistral import image as mistral_image
from obvs.vis import create_heatmap, plot_surprisal

images = {
    "gemma2": gemma_image,
    "gemma7": gemma_image,
    "mistral": mistral_image,
    "gpt2": gemma_image,
}


model_names = {
    "llamatiny": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "llama": "meta-llama/Llama-2-13b-hf",
    "gpt2": "gpt2",
    "mamba": "MrGonao/delphi-mamba-100k",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "gptj": "EleutherAI/gpt-j-6B",
    "gemma2": "google/gemma-2b",
    "gemma7": "google/gemma-7b",
}


stub = Stub(
    image=mistral_image,
    name="token_identity",
    secrets=[Secret.from_name("my-huggingface-secret")],
)


@stub.cls(
    gpu=gpu.A100(memory=80, count=1),
    # gpu=gpu.A10G(count=1),   # 24 GB
    # gpu=gpu.T4(count=1),  # 16 GB
    # cpu=1,
    timeout=60 * 30,
    container_idle_timeout=60 * 5,
)
class Runner:
    def setup_ti(self, model_name):
        from obvs.lenses import TokenIdentity

        self.ti = TokenIdentity("", model_name)

    @method()
    def run(self, model_name, prompt, full):
        if not hasattr(self, "ti"):
            print("Setting up TokenIdentity")
            self.setup_ti(model_name)
        self.ti._patchscope.source.prompt = prompt
        source_layers = range(self.ti._patchscope.n_layers_source)
        target_layers = range(self.ti._patchscope.n_layers_target)
        self.ti.run(
            source_layers=source_layers,
            target_layers=target_layers if full else None,
        ).compute_surprisal()
        return self.ti.surprisal, self.ti.source_layers


@stub.local_entrypoint()
def main(model_name, n_samples=5, full=False):
    n_samples = int(n_samples)
    full = bool(full)
    import os

    from datasets import load_dataset

    token = os.environ["HUGGINGFACE_TOKEN"].strip()
    dataset = load_dataset(
        "oscar-corpus/OSCAR-2201",
        "en",
        split="train",
        streaming=True,
        token=token,
    )
    shuffled_dataset = dataset.shuffle(seed=42, buffer_size=n_samples)

    samples = []
    for example in shuffled_dataset.take(n_samples):
        samples.append(example["text"])

    # Trim the samples to the first 300 characters
    samples = [sample[:1000] for sample in samples]

    # Make sure it ends on a space
    samples = [sample[: sample.rfind(" ")] for sample in samples]

    # Strip the spaces
    samples = [sample.strip() for sample in samples]

    filename = Path(f"{model_name}_surprisal_{n_samples}_full").expanduser()

    surprisals = []
    runner = Runner()
    for prompt in samples:
        try:
            surprisal, layers = runner.run.remote(model_names[model_name], prompt, full)
            surprisals.append(surprisal)
            if full:
                fig = create_heatmap(
                    layers,
                    layers,
                    surprisal,
                    title=f"{model_name} Surprisal of the first 1000 characters of a random sample from the OSCAR corpus",
                )
                fig.show()
                fig.write_html(filename.with_suffix(".html"))
            else:
                fig = plot_surprisal(
                    layers,
                    surprisal,
                    title=f"{model_name} Surprisal of the first 1000 characters of a random sample from the OSCAR corpus",
                )
                fig.show()
                fig.write_html(filename.with_suffix(".html"))
        except IndexError as e:
            print(e)
            import sys

            sys.exit(1)
        except Exception as e:
            print(e)
            break

    # Average the surprisals, calculate the standard deviation and plot with plotly
    import numpy as np

    if len(surprisals[0].shape) == 1:
        # Its a single set of layers
        mean_surprisal = np.mean(surprisals, axis=0)
        std_surprisal = np.std(surprisals, axis=0)

        fig = plot_surprisal(
            layers,
            mean_surprisal,
            std_surprisal,
            f"{model_name} Surprisal of the first 1000 characters of {n_samples} random samples from the OSCAR corpus",
        )
        fig.write_html(f"mean_surprisal_heatmap_{model_name}_{n_samples}_samples.html")
        fig.show()

    elif len(surprisals[0].shape) == 2:
        # Its a set of layers for each token, meaning a heatmap. We dont botther with the std
        mean_surprisal = np.mean(surprisals, axis=0)

        fig = create_heatmap(
            layers,
            layers,
            mean_surprisal,
            f"{model_name} Surprisal of the first 1000 characters of {n_samples} random samples from the OSCAR corpus",
        )
        fig.write_html(f"mean_surprisal_heatmap_{model_name}_{n_samples}_samples.html")
        fig.show()
