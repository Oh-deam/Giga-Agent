from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from catboost import CatBoostClassifier
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


MAX_SELECTION_ROWS = 40_000


@dataclass
class SelectionResult:
    selected_features: list[str]
    candidate_features: list[str]
    prefiltered_features: list[str]
    feature_importance: dict[str, float]
    validation_score: float | None
    pool_validation_score: float | None


def _prepare_X(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[int]]:
    X = df[feature_cols].copy()
    cat_features: list[int] = []
    for idx, col in enumerate(X.columns):
        if not is_numeric_dtype(X[col]):
            X[col] = X[col].astype(str)
            cat_features.append(idx)
    return X, cat_features


def _fit_catboost(X: pd.DataFrame, y: pd.Series) -> tuple[CatBoostClassifier, list[int]]:
    X_local, cat_features = _prepare_X(X, list(X.columns))
    model = CatBoostClassifier(allow_writing_files=False)
    model.fit(X_local, y, cat_features=cat_features or None, verbose=False)
    return model, cat_features


def _drop_invalid_features(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    valid_features: list[str] = []
    seen_features: set[str] = set()
    for col in feature_cols:
        if col in seen_features:
            continue
        seen_features.add(col)
        series = df[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        if bool(series.isna().all()):
            continue
        if int(series.nunique(dropna=False)) <= 1:
            continue
        valid_features.append(col)
    return valid_features


def _sample_for_selection(train_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if len(train_df) <= MAX_SELECTION_ROWS:
        return train_df

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=MAX_SELECTION_ROWS, random_state=42)
    indices, _ = next(splitter.split(train_df, train_df[target_col]))
    return train_df.iloc[indices].reset_index(drop=True)


def _split_train_valid(train_df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    y = train_df[target_col].copy()
    train_idx, valid_idx = next(splitter.split(train_df, y))
    return train_df.iloc[train_idx].reset_index(drop=True), train_df.iloc[valid_idx].reset_index(drop=True)


def _evaluate_fitted_model(
    model: CatBoostClassifier,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> float:
    y_valid = valid_df[target_col].copy()
    X_valid, _ = _prepare_X(valid_df, feature_cols)
    preds = model.predict_proba(X_valid)[:, 1]
    return roc_auc_score(y_valid, preds)


def _fit_selection_model(
    train_part: pd.DataFrame,
    valid_part: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> tuple[CatBoostClassifier, float]:
    X_train, cat_features = _prepare_X(train_part, feature_cols)
    model = CatBoostClassifier(allow_writing_files=False)
    model.fit(X_train, train_part[target_col], cat_features=cat_features or None, verbose=False)
    score = _evaluate_fitted_model(model, valid_part, feature_cols, target_col)
    return model, score


def select_top_features(
    train_df: pd.DataFrame,
    target_col: str,
    candidate_features: list[str],
    max_output_features: int = 5,
) -> SelectionResult:
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in train dataframe")

    candidate_features = _drop_invalid_features(train_df, candidate_features)
    if not candidate_features:
        return SelectionResult([], [], [], {}, None)

    selection_df = _sample_for_selection(train_df[[target_col, *candidate_features]].copy(), target_col)
    train_part, valid_part = _split_train_valid(selection_df, target_col)
    prefilter_limit = min(max_output_features, len(candidate_features))

    base_model, pool_validation_score = _fit_selection_model(train_part, valid_part, target_col, candidate_features)
    importances = (
        pd.Series(base_model.get_feature_importance(), index=candidate_features)
        .sort_values(ascending=False)
    )

    prefiltered_features = importances.head(prefilter_limit).index.tolist()
    selected_features = prefiltered_features[:max_output_features]
    selected_validation_score = None
    if selected_features:
        if selected_features == candidate_features:
            selected_validation_score = pool_validation_score
        else:
            _, selected_validation_score = _fit_selection_model(
                train_part,
                valid_part,
                target_col,
                selected_features,
            )

    return SelectionResult(
        selected_features=selected_features,
        candidate_features=candidate_features,
        prefiltered_features=prefiltered_features,
        feature_importance=importances.to_dict(),
        validation_score=selected_validation_score,
        pool_validation_score=pool_validation_score,
    )
