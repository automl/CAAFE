"""
Runs the CAAFE algorithm on a dataset and saves the generated code and prompt to a file.
"""

import argparse
from functools import partial

from tabpfn.scripts import tabular_metrics
from tabpfn import TabPFNClassifier
import tabpfn
from tabpfn.scripts.tabular_baselines import clf_dict
import os
import openai
import torch

from caafe.data import get_data_split, load_all_data
from caafe.caafe import generate_features


def generate_and_save_feats(i, seed=0, iterative_method=None, iterations=10):
    if iterative_method is None:
        iterative_method = tabpfn

    ds = cc_test_datasets_multiclass[i]

    ds, df_train, df_test, df_train_old, df_test_old = get_data_split(ds, seed)
    code, prompt, messages = generate_features(
        ds,
        df_train,
        just_print_prompt=False,
        model=model,
        iterative=iterations,
        metric_used=metric_used,
        iterative_method=iterative_method,
        display_method="print",
    )

    data_dir = os.environ.get("DATA_DIR", "data/")
    f = open(
        f"{data_dir}/generated_code/{ds[0]}_{prompt_id}_{seed}_prompt.txt",
        "w",
    )
    f.write(prompt)
    f.close()

    f = open(f"{data_dir}/generated_code/{ds[0]}_{prompt_id}_{seed}_code.txt", "w")
    f.write(code)
    f.close()


if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--dataset_id",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--prompt_id",
        type=str,
        default="v3",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
    )
    args = parser.parse_args()
    prompt_id = args.prompt_id
    dataset_id = args.dataset_id
    iterations = args.iterations
    seed = args.seed

    model = "gpt-3.5-turbo" if prompt_id == "v3" else "gpt-4"

    openai.api_key = os.environ["OPENAI_API_KEY"]

    cc_test_datasets_multiclass = load_all_data()
    if dataset_id != -1:
        cc_test_datasets_multiclass = [cc_test_datasets_multiclass[dataset_id]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = TabPFNClassifier(device=device, N_ensemble_configurations=16)
    classifier.fit = partial(classifier.fit, overwrite_warning=True)
    tabpfn = partial(clf_dict["transformer"], classifier=classifier)
    metric_used = tabular_metrics.auc_metric

    for i in range(0, len(cc_test_datasets_multiclass)):
        generate_and_save_feats(i, seed=seed, iterations=iterations)
