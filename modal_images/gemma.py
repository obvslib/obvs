from modal import Image, Stub, Secret


# Define the model names for LLaMA-2, Mistral, and GPT-2
model_names = {
    "llamatiny": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "llama": "meta-llama/Llama-2-13b-hf",
    "gpt2": "gpt2",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "gptj": "EleutherAI/gpt-j-6B",
    "gemma2": "google/gemma-2b",
    "gemma7": "google/gemma-7b",
}


def download_model():
    """
    Download the model and move the cache
    """
    import os
    token = os.environ["HUGGINGFACE_TOKEN"].strip()
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache
    snapshot_download(model_names["gemma7"], token=token, revision="main")
    snapshot_download(model_names["gemma2"], token=token, revision="main")
    move_cache()


image = (
    Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git")
    .pip_install(
        "git+https://github.com/fergusfettes/obvs.git@modal",
        "hf_transfer"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model,
        secrets=[Secret.from_name("my-huggingface-secret")],
    )
)


stub = Stub(image=image, name="gptj")


@stub.local_entrypoint()
def main():
    print("building the model...")
