import pandas as pd
import copy
import numpy as np


def create_mappings(df_train):
    mappings = {}
    for col in df_train.columns:
        if (
            df_train[col].dtype.name == "category"
            or df_train[col].dtype.name == "object"
        ):
            mappings[col] = dict(
                enumerate(df_train[col].astype("category").cat.categories)
            )
    return mappings


def convert_categorical_to_integer_f(column, mapping=None):
    if mapping is not None:
        # if column is categorical
        if column.dtype.name == "category":
            column = column.cat.add_categories([-1])
        return column.map(mapping).fillna(-1).astype(int)
    return column


def split_target_column(df, target):
    return df[[c for c in df.columns if c != target]], df[target] if target else None


def make_dataset_numeric(df, mappings):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.apply(
        lambda col: convert_categorical_to_integer_f(
            col, mapping=mappings.get(col.name)
        ),
        axis=0,
    )
    df = df.astype(float)

    return df


def make_datasets_numeric(df_train, df_test, target_column):
    df_train = copy.deepcopy(df_train)
    if df_test is not None:
        df_test = copy.deepcopy(df_test)

    # Create the mappings using the train and test datasets
    mappings = create_mappings(df_train)

    # Apply the mappings to the train and test datasets
    non_target = [c for c in df_train.columns if c != target_column]
    df_train[non_target] = make_dataset_numeric(df_train[non_target], mappings)

    if df_test is not None:
        df_test[non_target] = make_dataset_numeric(df_test[non_target], mappings)

    return df_train, df_test
