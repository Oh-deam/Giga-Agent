import pandas as pd
from src.schemas.tables import JoinCondition, JoinConditions
from src.utils.storage import Storage


def _agregate_before_join(df: pd.DataFrame, condition: JoinCondition) -> pd.DataFrame:
    
    print(f"ИТЕРАЦИЯ ДЛЯ {condition.table_name}")

    aggs = condition.aggregations

    group_by_cols = [c for c in aggs.group_by if c in df.columns]
    print("=============================================================")
    print(f"AGGS = {aggs}")
    print(f"ГРУППИРОВКА ПО КАЛОНКАМ = {group_by_cols}")
    agg_dict = {
        a.col_name: a.method
        for a in aggs.aggregations
        if a.col_name in df.columns and a.col_name not in group_by_cols
    }

    print(f"СЛОВАРЬ АГГ {agg_dict}")
    print("============================================================")
          
    if not group_by_cols or not agg_dict:
        return df.drop_duplicates(subset=[condition.on_col2])

    return df.groupby(group_by_cols, as_index=False, dropna=False).agg(agg_dict)


def merge_tables(joinconditions: JoinConditions, storage: Storage):
    
    df_train = storage.get_table("train.csv")
    df_test = storage.get_table("test.csv")
    
    for condition in joinconditions.conditions:
        df_for_join = storage.get_table(f"{condition.table_name}")
        df_for_join = _agregate_before_join(df_for_join, condition)
        # df_for_join = df_for_join.drop_duplicates(subset=[condition.on_col2])

        print(f"table for join: {condition.table_name}, \n columns: {df_for_join.columns}, \n train_columns: {df_train.columns} \n on_col1 {condition.on_col1}, \n on_col2 {condition.on_col2}")
        print("============================================================")
        df_train = pd.merge(df_train, df_for_join, left_on=f"{condition.on_col1}", right_on=f"{condition.on_col2}")
        df_test = pd.merge(df_test, df_for_join, left_on=f"{condition.on_col1}", right_on=f"{condition.on_col2}")

    return df_train, df_test


