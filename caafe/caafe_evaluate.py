import copy
import pandas as pd
import tabpfn
import numpy as np
from .data import get_X_y
from .preprocessing import make_datasets_numeric
from sklearn.base import BaseEstimator


def evaluate_dataset(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    prompt_id,
    name,
    method,
    metric_used,
    target_name,
    max_time=300,
    seed=0,
):
    df_train, df_test = make_datasets_numeric(df_train, df_test, target_name)

    if df_test is not None:
        test_x, test_y = get_X_y(df_test, target_name=target_name)

    x, y = get_X_y(df_train, target_name=target_name)
    feature_names = list(df_train.drop(target_name, axis=1).columns)

    np.random.seed(0)
    if method == "autogluon" or method == "autosklearn2":
        if method == "autogluon":
            from tabpfn.scripts.tabular_baselines import autogluon_metric

            clf = autogluon_metric
        elif method == "autosklearn2":
            from tabpfn.scripts.tabular_baselines import autosklearn2_metric

            clf = autosklearn2_metric
        metric, ys, res = clf(
            x, y, test_x, test_y, feature_names, metric_used, max_time=max_time
        )  #
    elif type(method) == str:
        if method == "gp":
            from tabpfn.scripts.tabular_baselines import gp_metric

            clf = gp_metric
        elif method == "knn":
            from tabpfn.scripts.tabular_baselines import knn_metric

            clf = knn_metric
        elif method == "xgb":
            from tabpfn.scripts.tabular_baselines import xgb_metric

            clf = xgb_metric
        elif method == "catboost":
            from tabpfn.scripts.tabular_baselines import catboost_metric

            clf = catboost_metric
        elif method == "random_forest":
            from tabpfn.scripts.tabular_baselines import random_forest_metric

            clf = random_forest_metric
        elif method == "logistic":
            from tabpfn.scripts.tabular_baselines import logistic_metric

            clf = logistic_metric
        metric, ys, res = clf(
            x,
            y,
            test_x,
            test_y,
            [],
            metric_used,
            max_time=max_time,
            no_tune={},
        )
    # If sklearn classifier
    elif isinstance(method, BaseEstimator):
        method.fit(X=x, y=y.long())
        ys = method.predict_proba(test_x)
    else:
        metric, ys, res = method(
            x,
            y,
            test_x,
            test_y,
            [],
            metric_used,
        )

    acc = tabpfn.scripts.tabular_metrics.accuracy_metric(test_y, ys)
    roc = tabpfn.scripts.tabular_metrics.auc_metric(test_y, ys)

    method_str = method if type(method) == str else "transformer"
    return {
        "acc": float(acc.numpy()),
        "roc": float(roc.numpy()),
        "prompt": prompt_id,
        "seed": seed,
        "name": name,
        "size": len(df_train),
        "method": method_str,
        "max_time": max_time,
        "feats": x.shape[-1],
    }


def get_leave_one_out_importance(
    df_train, df_test, ds, method, metric_used, max_time=30
):
    """Get the importance of each feature for a dataset by dropping it in the training and prediction."""
    res_base = evaluate_dataset(
        ds,
        df_train,
        df_test,
        prompt_id="",
        name=ds[0],
        method=method,
        metric_used=metric_used,
        max_time=max_time,
    )

    importances = {}
    for feat_idx, feat in enumerate(set(df_train.columns)):
        if feat == ds[4][-1]:
            continue
        df_train_ = df_train.copy().drop(feat, axis=1)
        df_test_ = df_test.copy().drop(feat, axis=1)
        ds_ = copy.deepcopy(ds)

        res = evaluate_dataset(
            ds_,
            df_train_,
            df_test_,
            prompt_id="",
            name=ds[0],
            method=method,
            metric_used=metric_used,
            max_time=max_time,
        )
        importances[feat] = (round(res_base["roc"] - res["roc"], 3),)
    return importances
