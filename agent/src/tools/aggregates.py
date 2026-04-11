from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


MAX_GROUP_KEYS = 2
MAX_NUMERIC_COLS = 2
MAX_FREQ_COLS = 3
MAX_CAT_NUNIQUE_COLS = 1
MAX_PAIR_NUMERIC_COLS = 4
LOW_CARDINALITY_LIMIT = 50
UNIQUE_RATIO_LIMIT = 0.95


def _split_combined_features(
    combined_features: pd.DataFrame,
    train_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_features = combined_features.iloc[:train_rows].reset_index(drop=True)
    test_features = combined_features.iloc[train_rows:].reset_index(drop=True)
    return train_features, test_features


def _detect_group_keys(input_train: pd.DataFrame, target_col: str) -> list[str]:
    keys: list[str] = []
    n_rows = max(len(input_train), 1)
    for col in input_train.columns:
        if col == target_col:
            continue
        lower = col.lower()
        nunique = input_train[col].nunique(dropna=True)
        unique_ratio = nunique / n_rows
        if nunique <= 1 or unique_ratio >= UNIQUE_RATIO_LIMIT:
            continue
        if lower.endswith("_id") or lower == "id":
            keys.append(col)
    return keys[:MAX_GROUP_KEYS]


def _select_numeric_columns(
    df: pd.DataFrame,
    excluded: set[str],
) -> list[str]:
    scored: list[tuple[float, str]] = []
    for col in df.columns:
        if col in excluded:
            continue
        if not is_numeric_dtype(df[col]):
            continue
        if col.lower().endswith("_id"):
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().all():
            continue
        if series.nunique(dropna=True) <= 2:
            continue
        score = float(series.std(skipna=True) or 0.0)
        scored.append((score, col))
    scored.sort(reverse=True)
    return [col for _, col in scored[:MAX_NUMERIC_COLS]]


def _select_pair_numeric_columns(
    df: pd.DataFrame,
    excluded: set[str],
) -> list[str]:
    priority_keywords = (
        "reorder",
        "cart",
        "order",
        "days",
        "hour",
        "dow",
        "count",
        "share",
    )
    priority: list[str] = []
    scored: list[tuple[float, str]] = []
    for col in df.columns:
        if col in excluded:
            continue
        if not is_numeric_dtype(df[col]):
            continue
        if col.lower().endswith("_id"):
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().all():
            continue
        if series.nunique(dropna=True) <= 1:
            continue
        lower = col.lower()
        if any(keyword in lower for keyword in priority_keywords):
            priority.append(col)
            continue
        score = float(series.std(skipna=True) or 0.0)
        scored.append((score, col))

    unique_priority: list[str] = []
    for col in priority:
        if col not in unique_priority:
            unique_priority.append(col)
    if len(unique_priority) >= MAX_PAIR_NUMERIC_COLS:
        return unique_priority[:MAX_PAIR_NUMERIC_COLS]

    scored.sort(reverse=True)
    for _, col in scored:
        if col not in unique_priority:
            unique_priority.append(col)
        if len(unique_priority) >= MAX_PAIR_NUMERIC_COLS:
            break
    return unique_priority


def _select_frequency_columns(
    df: pd.DataFrame,
    excluded: set[str],
) -> list[str]:
    selected: list[tuple[int, str]] = []
    for col in df.columns:
        if col in excluded:
            continue
        nunique = df[col].nunique(dropna=True)
        if nunique <= 1 or nunique > LOW_CARDINALITY_LIMIT:
            continue
        selected.append((int(nunique), col))
    selected.sort()
    return [col for _, col in selected[:MAX_FREQ_COLS]]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def generate_aggregate_feature_pool(
    input_train: pd.DataFrame,
    input_test: pd.DataFrame,
    merged_train: pd.DataFrame,
    merged_test: pd.DataFrame,
    target_col: str = "target",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_base = merged_train.drop(columns=[target_col], errors="ignore").reset_index(drop=True)
    test_base = merged_test.drop(columns=[target_col], errors="ignore").reset_index(drop=True)
    combined = pd.concat([train_base, test_base], axis=0, ignore_index=True)
    features = pd.DataFrame(index=combined.index)

    group_keys = _detect_group_keys(input_train, target_col=target_col)
    excluded = set(group_keys) | {target_col}
    numeric_cols = _select_numeric_columns(train_base, excluded=excluded)
    pair_numeric_cols = _select_pair_numeric_columns(train_base, excluded=excluded)
    frequency_cols = _select_frequency_columns(train_base, excluded=excluded)

    for col in frequency_cols:
        counts = combined[col].astype("string").fillna("__nan__").value_counts(dropna=False)
        freq_map = (counts / len(combined)).to_dict()
        features[f"freq_{col}"] = combined[col].astype("string").fillna("__nan__").map(freq_map).astype(float)

    for key in group_keys:
        key_size = combined.groupby(key, dropna=False)[key].transform("size").astype(float)
        features[f"rows_per_{key}"] = key_size

        for other_key in group_keys:
            if other_key == key:
                continue
            features[f"nunique_{other_key}_per_{key}"] = (
                combined.groupby(key, dropna=False)[other_key].transform("nunique").astype(float)
            )

        for num_col in numeric_cols:
            mean_by_key = combined.groupby(key, dropna=False)[num_col].transform("mean")
            features[f"{num_col}_mean_by_{key}"] = mean_by_key.astype(float)
            features[f"{num_col}_ratio_to_mean_by_{key}"] = _safe_ratio(
                pd.to_numeric(combined[num_col], errors="coerce"),
                pd.to_numeric(mean_by_key, errors="coerce"),
            ).astype(float)

        for cat_col in frequency_cols[:MAX_CAT_NUNIQUE_COLS]:
            if cat_col == key:
                continue
            features[f"nunique_{cat_col}_per_{key}"] = (
                combined.groupby(key, dropna=False)[cat_col].transform("nunique").astype(float)
            )

    for key_left, key_right in combinations(group_keys, 2):
        pair_size = combined.groupby([key_left, key_right], dropna=False)[key_left].transform("size").astype(float)
        left_size = combined.groupby(key_left, dropna=False)[key_left].transform("size").astype(float)
        right_size = combined.groupby(key_right, dropna=False)[key_right].transform("size").astype(float)
        features[f"rows_per_{key_left}_{key_right}"] = pair_size
        features[f"share_{key_left}_{key_right}_in_{key_left}"] = _safe_ratio(pair_size, left_size).astype(float)
        features[f"share_{key_left}_{key_right}_in_{key_right}"] = _safe_ratio(pair_size, right_size).astype(float)

        for num_col in pair_numeric_cols:
            pair_mean = combined.groupby([key_left, key_right], dropna=False)[num_col].transform("mean")
            left_mean = combined.groupby(key_left, dropna=False)[num_col].transform("mean")
            right_mean = combined.groupby(key_right, dropna=False)[num_col].transform("mean")
            features[f"{num_col}_mean_by_{key_left}_{key_right}"] = pair_mean.astype(float)
            features[f"{num_col}_pair_to_{key_left}_mean_ratio"] = _safe_ratio(
                pd.to_numeric(pair_mean, errors="coerce"),
                pd.to_numeric(left_mean, errors="coerce"),
            ).astype(float)
            features[f"{num_col}_pair_to_{key_right}_mean_ratio"] = _safe_ratio(
                pd.to_numeric(pair_mean, errors="coerce"),
                pd.to_numeric(right_mean, errors="coerce"),
            ).astype(float)

    features = features.loc[:, ~features.columns.duplicated()]
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(axis=1, how="all")

    train_features, test_features = _split_combined_features(features, train_rows=len(train_base))
    return train_features, test_features, features.columns.tolist()
