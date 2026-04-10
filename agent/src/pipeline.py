import pandas as pd
from langchain_gigachat import GigaChat

from .utils.storage import Storage
from .schemas.tables import JoinConditions
from .tools.join_tables import merge_tables


def pipeline(
        model: GigaChat,
        directory: str = "data",
        debug: bool = False,
):
    print("Start pipeline")
    storage = Storage(directory=directory, debug=debug)

    structured_llm = model.with_structured_output(JoinConditions)
    prompt = f"""Посмотри структуру таблиц и скажи, каким образом их объединять.
    Ответ должен быть последовательностью, с которой таблицы будут объединяться.
    Первая таблица train.csv
    {storage.tables_headers}
"""
    result = structured_llm.invoke(prompt)
    df_train, df_test = merge_tables(joinconditions=result, storage=storage)

    print(f"Df_train shape: {df_train.shape}")
    print(f"Df_test shape: {df_test.shape}")


