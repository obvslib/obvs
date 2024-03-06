from __future__ import annotations

from datasets import load_dataset

from obvs.lenses import TokenIdentity
from obvs.vis import create_heatmap, plot_surprisal

dataset = load_dataset("oscar-corpus/OSCAR-2201", "en", split="train", streaming=True)
shuffled_dataset = dataset.shuffle(seed=42, buffer_size=50)

model_names = {
    "llamatiny": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "llama": "meta-llama/Llama-2-13b-hf",
    "gpt2": "gpt2",
    "mamba": "MrGonao/delphi-mamba-100k",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "gptj": "EleutherAI/gpt-j-6B",
    "gemma": "google/gemma-2b",
}


prompts = [
    # "bat is bat; 135 is 135; hello is hello; black is black; shoe is shoe; x is",
    "bat is god; 135 is 5; hello is shoe; black is not; six is old; x is",
    "bat ➡️ bat; 135 ➡️ 135; hello ➡️ hello; black ➡️ black; shoe ➡️ shoe; x ➡️",
    "bat is bat, 135 is 135, hello is hello, black is black, shoe is shoe, x is",
    "bat, shoe, 135, hello, black, fire, tree, z, 1, new, true, x",
    "x is",
    "it is",
    "the",
    "hello! here is a something I'm trying to decode, can you help me? x",
    "hello! here is a something I'm trying to decode, can you help me? x is",
    "hello! here is a something I'm trying to decode, it is normally written like so: x",
    "hello! here is a something I'm trying to understand, it is normally written like so: x",
    "Hello! Here is a something I'm trying to understand, it is normally written like so: x",
    "Hello! Here is a something I'm trying to understand, it is normally written like so: 'x",
    "I found a few files on an old hard drive. Here are some fragments: x",
    "I found a few files on an old hard drive. Here are some fragments: x is",
    "I found a few files on an old hard drive. Here are some fragments: 'x",
    "Someone sent me this. Do you understand it? x",
    "Someone sent me this. Do you understand it? x is",
    "Someone sent me this. Do you understand it? 'x",
    "Imaging someone saying such a random thing! (I'll tell you what they said: 'x",
    "Imaging someone writing such a random thing! (I'll tell you what they said: 'x",
]


multiprompts = [
    "Love: i love you 10, i like you 9, i hate you 1, i don't care 0, x _; "
    "Hate: i love you 1, i like you 0, i hate you 10, i don't care 9, x _; "
    "Indifference: i love you 0, i like you 1, i hate you 1, i don't care 10, x _; "
    "Size: elephant 8, mouse 2, ant 1, whale 9, x _; ",
]


multiprompts = [
    # Adjectives
    "pretty, slow, fast, tall, short, big, small, heavy, light, dark, bright, x; ",
    # Nouns
    "elephant, hat, cat, degree, raindrop, planet, x; ",
    # Verbs
    "run, walk, jump, sit, stand, sleep, eat, drink, x; ",
    # Pronouns
    "he, she, it, they, we, you, I, x; ",
    # Punctuation
    "., ,, ?, !, :, ;, ', x; ",
]


def main(model_name, target_prompt, samples, full=False):
    ti = TokenIdentity("", model_names[model_name], device="cpu")
    ti.patchscope.target.prompt = target_prompt

    source_layers = range(ti.patchscope.n_layers_source)
    target_layers = range(ti.patchscope.n_layers_target)

    # Remove everything except alphanumeric characters
    clean_prompt = "".join(e for e in target_prompt if e.isalnum())

    surprisals = []
    for prompt in samples:
        ti.filename = f"{'full' if full else ''}token_identity_{model_name}_{prompt.replace(' ', '')[:10]}_{target_prompt[-10:]}"
        ti.patchscope.source.prompt = prompt
        ti.run(
            source_layers=source_layers,
            target_layers=target_layers if full else None,
        ).compute_surprisal().visualize()
        surprisals.append(ti.surprisal)

    # Average the surprisals, calculate the standard deviation and plot with plotly
    import numpy as np

    if len(surprisals[0].shape) == 1:
        # Its a single set of layers
        mean_surprisal = np.mean(surprisals, axis=0)
        std_surprisal = np.std(surprisals, axis=0)

        fig = plot_surprisal(
            ti.source_layers,
            mean_surprisal,
            std_surprisal,
            f"{model_name} Surprisal of the first 1000 characters of {len(samples)} random samples from the OSCAR corpus with prompt {target_prompt}",
        )
        fig.write_html(
            f"mean_surprisal_{model_names[model_name]}_{len(samples)}_samples_target_{clean_prompt[-20:]}.html",
        )
        fig.show()

    elif len(surprisals[0].shape) == 2:
        # Its a set of layers for each token, meaning a heatmap. We dont botther with the std
        mean_surprisal = np.mean(surprisals, axis=0)

        fig = create_heatmap(
            ti.source_layers,
            ti.target_layers,
            mean_surprisal,
            f"{model_name} Surprisal of the first 1000 characters of {len(samples)} random samples from the OSCAR corpus with prompt {target_prompt}",
        )
        fig.write_html(
            f"mean_surprisal_heatmap_{model_names[model_name]}_{len(samples)}_samples_target_{clean_prompt[-20:]}.html",
        )
        fig.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate the surprisal of a set of samples using a model",
    )
    parser.add_argument("model_name", type=str, help="The name of the model to use")
    parser.add_argument("--n", type=int, default=5, help="The number of samples to average over")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Whether to run over all layers or pairs",
    )
    args = parser.parse_args()

    # for prompt in prompts:
    #     main(args.model_name, prompt, args.n, args.full)

    samples = []
    for example in shuffled_dataset.take(args.n):
        samples.append(example["text"])

    # Trim the samples to the first 300 characters
    samples = [sample[:1000] for sample in samples]

    # Make sure it ends on a space
    samples = [sample[: sample.rfind(" ")] for sample in samples]

    # Strip the spaces
    samples = [sample.strip() for sample in samples]

    # Run each in a seperate process
    import multiprocessing

    with multiprocessing.Pool(len(prompts)) as p:
        p.starmap(main, [(args.model_name, prompt, samples, args.full) for prompt in prompts])
