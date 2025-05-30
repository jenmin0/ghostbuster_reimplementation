from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from nltk.util import ngrams
from nltk.corpus import brown
import dill as pickle

from ghostbuster.utils.load import get_generate_dataset, Dataset
from ghostbuster.utils.symbolic import generate_symbolic_data
from ghostbuster.utils.featurize import normalize, t_featurize, select_features


wp_dataset = [
    Dataset("normal", "data/wp/human"),
    Dataset("normal", "data/wp/gpt"),
]

reuter_dataset = [
    Dataset("author", "data/reuter/human"),
    Dataset("author", "data/reuter/gpt"),
]

essay_dataset = [
    Dataset("normal", "data/essay/human"),
    Dataset("normal", "data/essay/gpt"),
]

eval_dataset = [
    Dataset("normal", "data/wp/claude"),
    Dataset("author", "data/reuter/claude"),
    Dataset("normal", "data/essay/claude"),
    Dataset("normal", "data/wp/gpt_prompt1"),
    Dataset("author", "data/reuter/gpt_prompt1"),
    Dataset("normal", "data/essay/gpt_prompt1"),
    Dataset("normal", "data/wp/gpt_prompt2"),
    Dataset("author", "data/reuter/gpt_prompt2"),
    Dataset("normal", "data/essay/gpt_prompt2"),
    Dataset("normal", "data/wp/gpt_writing"),
    Dataset("author", "data/reuter/gpt_writing"),
    Dataset("normal", "data/essay/gpt_writing"),
    Dataset("normal", "data/wp/gpt_semantic"),
    Dataset("author", "data/reuter/gpt_semantic"),
    Dataset("normal", "data/essay/gpt_semantic"),
]

if __name__ == "__main__":

    datasets = [
        *wp_dataset,
        *reuter_dataset,
        *essay_dataset,
    ]
    generate_dataset_fn = get_generate_dataset(*datasets)

    generate_symbolic_data(
            generate_dataset_fn,
            max_depth=4,
            output_file="symbolic_data_gpt_four",
            verbose=True,
        )

    t_data = generate_dataset_fn(t_featurize)
    pickle.dump(t_data, open("t_data", "wb"))

    labels = generate_dataset_fn(
        lambda file: 1 if any([m in file for m in ["gpt", "claude"]]) else 0
    )

    indices = np.arange(len(labels))
