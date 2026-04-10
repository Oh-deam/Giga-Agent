import pandas as pd
from src.schemas.tables import JoinConditions
from src.utils.storage import Storage

def merge_tables(joinconditions: JoinConditions, storage: Storage):
    
    df_train = storage.get_table("train.csv")
    df_test = storage.get_table("test.csv")
    
    for condition in joinconditions.conditions:
        df_for_join = storage.get_table(f"{condition.table_name}")
        df_for_join = df_for_join.drop_duplicates(subset=[condition.on_col2])

        print(f"table for join: {condition.table_name}, \n columns: {df_for_join.columns}, \n train_columns: {df_train.columns} \n on_col1 {condition.on_col1}, \n on_col2 {condition.on_col2}")
        print("============================================================")
        df_train = pd.merge(df_train, df_for_join, left_on=f"{condition.on_col1}", right_on=f"{condition.on_col2}")
        df_test = pd.merge(df_test, df_for_join, left_on=f"{condition.on_col1}", right_on=f"{condition.on_col2}")

    return df_train, df_test





