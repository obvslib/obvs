from modal import Stub, gpu, method, Secret
from modal_images.gemma import image as gemma_image
from modal_images.mistral import image as mistral_image

from datasets import load_dataset
from obvs.vis import create_heatmap, plot_surprisal

dataset = load_dataset('oscar-corpus/OSCAR-2201', 'en', split='train', streaming=True)
shuffled_dataset = dataset.shuffle(seed=42, buffer_size=10_000)

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
    "gemma": "google/gemma-2b"
}


stub = Stub(image=mistral_image, name="token_identity", secrets=[Secret.from_name("my-huggingface-secret")])


@stub.cls(
    gpu=gpu.A100(memory=40, count=1),
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
        self.ti.filename = f"{'full' if full else ''}token_identity_{model_name}_{prompt.replace(' ', '')[:10]}"
        self.ti.patchscope.source.prompt = prompt
        source_layers = range(self.ti.patchscope.n_layers_source)
        target_layers = range(self.ti.patchscope.n_layers_target)
        self.ti.run(
            source_layers=source_layers,
            target_layers=target_layers if full else None
        ).compute_surprisal()
        return self.ti.surprisal, self.ti.source_layers


@stub.local_entrypoint()
def main(model_name, n_samples=5, full=False):
    samples = []
    for example in shuffled_dataset.take(n_samples):
        samples.append(example['text'])

    # Trim the samples to the first 300 characters
    samples = [sample[:1000] for sample in samples]

    # Make sure it ends on a space
    samples = [sample[:sample.rfind(' ')] for sample in samples]

    # Strip the spaces
    samples = [sample.strip() for sample in samples]

    surprisals = []
    runner = Runner()
    for prompt in samples:
        try:
            surprisal, layers = runner.run.remote(model_names[model_name], prompt)
            surprisals.append(surprisal)
            if full:
                create_heatmap(
                    layers,
                    layers,
                    surprisal,
                    title=f"{model_name} Surprisal of the first 1000 characters of a random sample from the OSCAR corpus"
                ).show()
            else:
                plot_surprisal(
                    layers,
                    surprisal,
                    title=f"{model_name} Surprisal of the first 1000 characters of a random sample from the OSCAR corpus"
                ).show()
        except Exception as e:
            print(e)
            break

    # Average the surprisals, calculate the standard deviation and plot with plotly
    import numpy as np

    if len(surprisals[0].shape) == 1:
        # Its a single set of layers
        mean_surprisal = np.mean(surprisals, axis=0)
        std_surprisal = np.std(surprisals, axis=0)

        fig = plot_surprisal(ti.source_layers, mean_surprisal, std_surprisal, f"{model_name} Surprisal of the first 1000 characters of {n} random samples from the OSCAR corpus")
        fig.write_html(f"mean_surprisal_heatmap_{model_names[model_name]}_{len(samples)}_samples.html")
        fig.show()

    elif len(surprisals[0].shape) == 2:
        # Its a set of layers for each token, meaning a heatmap. We dont botther with the std
        mean_surprisal = np.mean(surprisals, axis=0)

        fig = create_heatmap(ti.source_layers, ti.target_layers, mean_surprisal, f"{model_name} Surprisal of the first 1000 characters of {n} random samples from the OSCAR corpus")
        fig.write_html(f"mean_surprisal_heatmap_{model_names[model_name]}_{len(samples)}_samples.html")
        fig.show()
