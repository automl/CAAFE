import copy
import numpy as np
from .preprocessing import convert_categorical_to_integer_f


def run_llm_code(code, df, convert_categorical_to_integer=True, fill_na=True):
    try:
        loc = {}
        df = copy.deepcopy(df)

        if fill_na and False:
            df.loc[:, (df.dtypes == object)] = df.loc[:, (df.dtypes == object)].fillna(
                ""
            )
        if convert_categorical_to_integer and False:
            df = df.apply(convert_categorical_to_integer_f)

        access_scope = {"df": df, "pd": pd, "np": np}
        parsed = ast.parse(code)
        check_ast(parsed)
        exec(compile(parsed, filename="<ast>", mode="exec"), access_scope, loc)
        df = copy.deepcopy(df)

    except Exception as e:
        print("Code could not be executed", e)
        raise (e)

    return df


import ast
import pandas as pd


def check_ast(node):
    allowed_nodes = {
        ast.Module,
        ast.Expr,
        ast.Load,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Num,
        ast.Str,
        ast.Bytes,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.Name,
        ast.Call,
        ast.Attribute,
        ast.keyword,
        ast.Subscript,
        ast.Index,
        ast.Slice,
        ast.ExtSlice,
        ast.Assign,
        ast.AugAssign,
        ast.NameConstant,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
        ast.And,
        ast.Or,
        ast.BitOr,
        ast.BitAnd,
        ast.BitXor,
        ast.Invert,
        ast.Not,
        ast.Constant,
        ast.Store,
        ast.If,
        ast.IfExp,
        # These nodes represent loop structures. If you allow arbitrary loops, a user could potentially create an infinite loop that consumes system resources and slows down or crashes your system.
        ast.For,
        ast.While,
        ast.Break,
        ast.Continue,
        ast.Pass,
        ast.Assert,
        ast.Return,
        ast.FunctionDef,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.Await,
        # These nodes represent the yield keyword, which is used in generator functions. If you allow arbitrary generator functions, a user might be able to create a generator that produces an infinite sequence, potentially consuming system resources and slowing down or crashing your system.
        ast.Yield,
        ast.YieldFrom,
        ast.Lambda,
        ast.BoolOp,
        ast.FormattedValue,
        ast.JoinedStr,
        ast.Set,
        ast.Ellipsis,
        ast.expr,
        ast.stmt,
        ast.expr_context,
        ast.boolop,
        ast.operator,
        ast.unaryop,
        ast.cmpop,
        ast.comprehension,
        ast.arguments,
        ast.arg,
        ast.Import,
        ast.ImportFrom,
        ast.alias,
    }

    allowed_packages = {"numpy", "pandas", "sklearn"}

    allowed_funcs = {
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "enumerate": enumerate,
        "zip": zip,
        "range": range,
        "sorted": sorted,
        "reversed": reversed,
        # Add other functions you want to allow here.
    }

    allowed_attrs = {
        # NP
        "array",
        "arange",
        "values",
        "linspace",
        # PD
        "mean",
        "sum",
        "contains",
        "min",
        "max",
        "median",
        "std",
        "sqrt",
        "pow",
        "iloc",
        "cut",
        "qcut",
        "inf",
        "nan",
        "isna",
        "map",
        "reshape",
        "split",
        "var",
        "codes",
        "abs",
        "cumsum",
        "cumprod",
        "cummax",
        "cummin",
        "diff",
        "repeat",
        "index",
        "log",
        "log10",
        "log1p",
        "exp",
        "expm1",
        "pow",
        "pct_change",
        "corr",
        "cov",
        "round",
        "clip",
        "dot",
        "transpose",
        "T",
        "astype",
        "copy",
        "drop",
        "dropna",
        "fillna",
        "replace",
        "merge",
        "append",
        "join",
        "groupby",
        "resample",
        "rolling",
        "expanding",
        "ewm",
        "agg",
        "aggregate",
        "filter",
        "transform",
        "apply",
        "pivot",
        "melt",
        "sort_values",
        "sort_index",
        "reset_index",
        "set_index",
        "reindex",
        "shift",
        "extract",
        "rename",
        "tail",
        "head",
        "describe",
        "count",
        "value_counts",
        "unique",
        "nunique",
        "idxmin",
        "idxmax",
        "isin",
        "between",
        "duplicated",
        "rank",
        "to_numpy",
        "to_dict",
        "to_list",
        "to_frame",
        "squeeze",
        "add",
        "sub",
        "mul",
        "div",
        "mod",
        "columns",
        "loc",
        "lt",
        "le",
        "eq",
        "ne",
        "ge",
        "gt",
        "all",
        "any",
        "clip",
        "conj",
        "conjugate",
        "round",
        "trace",
        "cumprod",
        "cumsum",
        "prod",
        "dot",
        "flatten",
        "ravel",
        "T",
        "transpose",
        "swapaxes",
        "clip",
        "item",
        "tolist",
        "argmax",
        "argmin",
        "argsort",
        "max",
        "mean",
        "min",
        "nonzero",
        "ptp",
        "sort",
        "std",
        "var",
        "str",
        "dt",
        "cat",
        "sparse",
        "plot"
        # Add other DataFrame methods you want to allow here.
    }

    if type(node) not in allowed_nodes:
        raise ValueError(f"Disallowed code: {ast.unparse(node)} is {type(node)}")

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id not in allowed_funcs:
            raise ValueError(f"Disallowed function: {node.func.id}")

    if isinstance(node, ast.Attribute) and node.attr not in allowed_attrs:
        raise ValueError(f"Disallowed attribute: {node.attr}")

    if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
        for alias in node.names:
            if alias.name not in allowed_packages:
                raise ValueError(f"Disallowed package import: {alias.name}")

    for child in ast.iter_child_nodes(node):
        check_ast(child)
