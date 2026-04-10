import pandas as pd
from src.schemas.tables import JoinConditions
from src.utils.storage import Storage

def merge_tables(joinconditions: JoinConditions, storage: Storage):
    
    df_train = storage.get_table("train.csv")
    df_test = storage.get_table("test.csv")
    
    for condition in joinconditions:
        df_for_join = storage.get_table(f"{condition.table_name}")

        pd.merge(df_train, df_for_join, left_on=f"{condition.on_col1}", right_on=f"{condition.on_col2}")
        pd.merge(df_test, df_for_join, left_on=f"{condition.on_col1}", right_on=f"{condition.on_col2}")





