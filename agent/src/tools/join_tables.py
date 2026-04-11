import loguru
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.schemas.tables import JoinCondition, JoinConditions
from src.utils.storage import Storage

ALLOWED_AGG_METHODS = {"mean", "sum", "min", "max", "count", "nunique", "median", "std", "first", "last"}


def _collapse_to_join_key(df: pd.DataFrame, join_key: str) -> pd.DataFrame:
    if join_key not in df.columns:
        return df
    if not df[join_key].duplicated().any():
        return df

    loguru.logger.warning(
        f"Duplicate rows remain after aggregation for join key '{join_key}', collapsing to one row per key"
    )
    agg_dict: dict[str, str] = {}
    for col in df.columns:
        if col == join_key:
            continue
        agg_dict[col] = "mean" if is_numeric_dtype(df[col]) else "first"

    if not agg_dict:
        return df.drop_duplicates(subset=[join_key])
    return df.groupby(join_key, as_index=False, dropna=False).agg(agg_dict)


def _agregate_before_join(df: pd.DataFrame, condition: JoinCondition) -> pd.DataFrame:
    aggs = condition.aggregations
    group_by_cols = [c for c in aggs.group_by if c in df.columns]

    if condition.on_col2 not in df.columns:
        loguru.logger.warning(
            f"Join key {condition.on_col2} is absent in {condition.table_name}; available columns: {list(df.columns)}"
        )
        return df

    if condition.on_col2 not in group_by_cols:
        loguru.logger.warning(
            f"Invalid group_by for {condition.table_name}: {group_by_cols}. Replacing with join key [{condition.on_col2}]"
        )
        group_by_cols = [condition.on_col2]

    agg_dict = {
        a.col_name: a.method.lower()
        for a in aggs.aggregations
        if (
            a.col_name in df.columns
            and a.col_name not in group_by_cols
            and a.method.lower() in ALLOWED_AGG_METHODS
        )
    }

    if not group_by_cols or not agg_dict:
        collapsed = df.drop_duplicates(subset=[condition.on_col2])
        return _collapse_to_join_key(collapsed, condition.on_col2)

    aggregated = df.groupby(group_by_cols, as_index=False, dropna=False).agg(agg_dict)
    return _collapse_to_join_key(aggregated, condition.on_col2)


def merge_tables(joinconditions: JoinConditions, storage: Storage):
    df_train = storage.get_table("train.csv")
    df_test = storage.get_table("test.csv")

    for condition in joinconditions.conditions:
        df_for_join = storage.get_table(condition.table_name)
        df_for_join = _agregate_before_join(df_for_join, condition)

        if condition.on_col1 not in df_train.columns:
            loguru.logger.warning(
                f"Skip join {condition.table_name}: left key {condition.on_col1} absent in train columns {list(df_train.columns)}"
            )
            continue
        if condition.on_col1 not in df_test.columns:
            loguru.logger.warning(
                f"Skip join {condition.table_name}: left key {condition.on_col1} absent in test columns {list(df_test.columns)}"
            )
            continue
        if condition.on_col2 not in df_for_join.columns:
            loguru.logger.warning(
                f"Skip join {condition.table_name}: right key {condition.on_col2} absent after aggregation; columns={list(df_for_join.columns)}"
            )
            continue

        df_train = pd.merge(
            df_train,
            df_for_join,
            how="left",
            left_on=condition.on_col1,
            right_on=condition.on_col2,
        )
        df_test = pd.merge(
            df_test,
            df_for_join,
            how="left",
            left_on=condition.on_col1,
            right_on=condition.on_col2,
        )

    return df_train, df_test
