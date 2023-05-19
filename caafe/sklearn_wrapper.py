from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from .run_llm_code import run_llm_code
from .preprocessing import make_datasets_numeric, split_target_column
from .caafe import generate_features
from .metrics import auc_metric, accuracy_metric
import pandas as pd
import numpy as np


class CAAFEClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        base_classifier=None,
        optimization_metric="accuracy",
        iterations=10,
        llm_model="gpt-3.5-turbo",
        n_splits=10,
        n_repeats=2,
    ):
        self.base_classifier = base_classifier
        if self.base_classifier is None:
            from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
            import torch
            from functools import partial

            self.base_classifier = TabPFNClassifier(
                N_ensemble_configurations=16,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            self.base_classifier.fit = partial(
                self.base_classifier.fit, overwrite_warning=True
            )
        self.llm_model = llm_model
        self.iterations = iterations
        self.optimization_metric = optimization_metric
        self.n_splits = n_splits
        self.n_repeats = n_repeats

    def fit_pandas(self, df, dataset_description, target_column_name, **kwargs):
        X, y = df.drop(columns=[target_column_name]), df[target_column_name]
        return self.fit(
            X, y, dataset_description, X.columns, target_column_name, **kwargs
        )

    def fit(
        self, X, y, dataset_description, feature_names, target_name, disable_caafe=False
    ):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        self.dataset_description = dataset_description
        self.feature_names = list(feature_names)
        self.target_name = target_name

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        ds = [
            "dataset",
            X,
            y,
            [],
            self.feature_names + [target_name],
            {},
            dataset_description,
        ]
        # Add X and y as one dataframe
        df_train = pd.DataFrame(
            np.concatenate([X, y.reshape(-1, 1)], axis=1),
            columns=self.feature_names + [target_name],
        )
        if disable_caafe:
            self.code = ""
        else:
            self.code, prompt, messages = generate_features(
                ds,
                df_train,
                model=self.llm_model,
                iterative=self.iterations,
                metric_used=auc_metric,
                iterative_method=self.base_classifier,
                display_method="markdown",
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
            )

        df_train = run_llm_code(
            self.code,
            df_train,
        )

        df_train, _ = make_datasets_numeric(
            df_train, df_test=None, target_column=target_name
        )

        df_train, y = split_target_column(df_train, target_name)

        self.base_classifier.fit(df_train.values, y.values)

        # Return the classifier
        return self

    def predict_preprocess(self, X):

        # Check if fit has been called
        check_is_fitted(self)

        if type(X) != pd.DataFrame:
            X = pd.DataFrame(X, columns=self.X_.columns)

        X = X[self.feature_names]

        X = run_llm_code(
            self.code,
            X,
        )

        X = X.values

        # Input validation
        X = check_array(X)

        return X

    def predict_proba(self, X):
        X = self.predict_preprocess(X)
        return self.base_classifier.predict_proba(X)

    def predict(self, X):
        X = self.predict_preprocess(X)
        return self.base_classifier.predict(X)
