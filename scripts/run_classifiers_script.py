"""
Runs downstream classifiers on the dataset and combines it with different feature sets.
"""

import argparse
import torch
from tabpfn.scripts import tabular_metrics
from cafe_feature_engineering import data, cafe, evaluate, settings
import os
import openai
from tabpfn.scripts.tabular_baselines import clf_dict
from tabpfn import TabPFNClassifier
from functools import partial


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
        default="v4",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = TabPFNClassifier(device=device, N_ensemble_configurations=16)
    classifier.fit = partial(classifier.fit, overwrite_warning=True)
    tabpfn = partial(clf_dict["transformer"], classifier=classifier)

    prompt_id = args.prompt_id
    dataset_id = args.dataset_id
    seed = args.seed
    methods = [tabpfn, "logistic", "random_forest", "xgb", "autosklearn2", "autogluon"]

    cc_test_datasets_multiclass = data.load_all_data()
    if dataset_id != -1:
        cc_test_datasets_multiclass = [cc_test_datasets_multiclass[dataset_id]]

    metric_used = tabular_metrics.auc_metric

    for i in range(0, len(cc_test_datasets_multiclass)):
        ds = cc_test_datasets_multiclass[i]
        evaluate.evaluate_dataset_with_and_without_cafe(
            ds, seed, methods, metric_used, prompt_id=prompt_id
        )
