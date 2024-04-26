# Below code from these locations:
# https://github.com/PAIR-code/interpretability/blob/master/patchscopes/code/download_the_pile_text_data.py
# https://github.com/PAIR-code/interpretability/blob/master/patchscopes/code/next_token_prediction.ipynb

from __future__ import annotations

import argparse
import json

import datasets
import pandas as pd
from datasets import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        default="./the_pile_deduplicated",
        type=str,
        help="Specify the output path.",
    )
    parser.add_argument(
        "--num_samples",
        default=200000,
        type=int,
        help="Specify the number of the pile text samples.",
    )

    args = parser.parse_args()
    dataset = load_dataset("EleutherAI/the_pile_deduplicated", streaming=True, split="train")
    data_lst = list(dataset.take(args.num_samples))
    partial_pile_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=data_lst))
    partial_pile_dataset = partial_pile_dataset.filter(
        lambda x: len(x["text"].split(" ")) < 250 and len(x["text"]) < 2000,
    ).shuffle(seed=42)
    print(len(partial_pile_dataset))

    trn_n = 10000
    val_n = 2000
    pile_trn = partial_pile_dataset["text"][:trn_n]
    pile_val = partial_pile_dataset["text"][trn_n : trn_n + val_n]
    sentences = [(x, "train") for x in pile_trn] + [(x, "validation") for x in pile_val]

    with open("data/processed/sentences.json", "w") as f:
        json.dump(sentences, f)
