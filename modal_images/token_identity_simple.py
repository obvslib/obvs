from modal import Stub, gpu, method, Secret
from modal_images.gemma import image as gemma_image
from modal_images.mistral import image as mistral_image

from obvs.vis import plot_surprisal


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


def get_samples():
    from datasets import load_dataset
    dataset = load_dataset('oscar-corpus/OSCAR-2201', 'en', split='train', streaming=True)
    shuffled_dataset = dataset.shuffle(seed=42, buffer_size=10_000)

    samples = []
    for example in shuffled_dataset.take(10):
        samples.append(example['text'])

    # Trim the samples to the first 300 characters
    samples = [sample[:1000] for sample in samples]

    # Make sure it ends on a space
    samples = [sample[:sample.rfind(' ')] for sample in samples]

    # Strip the spaces
    samples = [sample.strip() for sample in samples]

    return samples


@stub.cls(
    gpu=gpu.A100(memory=40, count=1),
    timeout=60 * 30,
    container_idle_timeout=60 * 5,
)
class Runner:
    def setup_ti(self, model_name):
        from obvs.lenses import TokenIdentity
        self.ti = TokenIdentity("", model_name)

    def show_gpu_memory(self):
        import torch
        print(torch.cuda.memory_summary())

    def clear_gpu_memory(self):
        import gc
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    @method()
    def run(self, model_name, prompt):
        self.show_gpu_memory()
        self.clear_gpu_memory()
        self.show_gpu_memory()
        if not hasattr(self, "ti"):
            print("Setting up TokenIdentity")
            self.setup_ti(model_name)
        self.ti.patchscope.source.prompt = prompt
        self.ti.run().compute_surprisal()
        return self.ti.surprisal, self.ti.layers


@stub.local_entrypoint()
def main():
    prompts = get_samples()
    surprisals = []
    runner = Runner()
    for prompt in prompts:
        try:
            surprisal, layers = runner.run.remote(model_names["mistral"], prompt)
            surprisals.append(surprisal)
            plot_surprisal(
                layers, surprisal, title=f"Token Identity: Surprisal by Layer {model_names['mistral']} Prompt: {prompt[-30:]}"
            ).show()
        except Exception as e:
            print(e)
            break

    # Average the surprisals, calculate the standard deviation and plot with plotly
    import numpy as np

    mean_surprisal = np.mean(surprisals, axis=0)
    std_surprisal = np.std(surprisals, axis=0)

    plot_surprisal(
        layers,
        mean_surprisal,
        std=std_surprisal,
        title=f"Token Identity: Surprisal by Layer {model_names['mistral']}",
    )
