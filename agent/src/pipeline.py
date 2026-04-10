import pandas as pd
from langchain_gigachat import GigaChat

from .utils.storage import Storage
from .schemas.tables import JoinConditions


def pipeline(
        model: GigaChat,
        directory: str = "data",
        debug: bool = False,
):
    print("Start pipeline")
    storage = Storage(directory=directory, debug=debug)

    structured_llm = model.with_structured_output(JoinConditions)
    prompt = "Посмотри структуру таблиц и скажи, каким образом их объединять. Ответ должен быть последовательностью, с которой таблицы будут объединяться. Первая таблица train.csv"
    result = structured_llm.invoke(prompt)
    print(result)

