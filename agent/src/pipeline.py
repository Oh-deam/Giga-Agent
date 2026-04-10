import loguru

import pandas as pd
from langchain_gigachat import GigaChat

from .schemas.future import Proposal
from .utils.storage import Storage
from .schemas.tables import JoinConditions
from .tools.join_tables import merge_tables
from .tools.stat import create_stat


def pipeline(
        model: GigaChat,
        directory: str = "data",
        debug: bool = False,
):
    print("Start pipeline")
    storage = Storage(directory=directory, debug=debug)

    structured_llm = model.with_structured_output(JoinConditions)
    prompt = f"""Посмотри структуру таблиц и определи порядок их объединения с train.csv.

                КРИТИЧЕСКИ ВАЖНО:
                - Файл data_dictionary.csv (или подобные файлы в которых описана структура таблиц (названия колонок)) — это СЛОВАРЬ МЕТАДАННЫХ (описание полей), а НЕ таблица с данными. ПОЛНОСТЬЮ ИСКЛЮЧИ его из результата. Никогда не добавляй data_dictionary.csv в список джойнов.
                - Исключи также test.csv — он джойнится отдельно вне твоего плана.
                - Имена таблиц и столбцов бери ПОБУКВЕННО из списка ниже. Запрещено выдумывать, переименовывать, сокращать или додумывать столбцы, которых нет в списке.
                - Если таблицу невозможно корректно присоединить (нет общего столбца ни с train, ни с уже присоединёнными) — НЕ включай её в результат.

                Правила построения последовательности:
                1. Начинаем с train.csv. Её исходные столбцы: row_id, user_id, product_id, target.
                2. На каждом шаге on_col1 — это столбец, который УЖЕ ЕСТЬ в накопленной таблице (train + все ранее присоединённые таблицы из предыдущих шагов).
                3. on_col2 — столбец присоединяемой таблицы; он ОБЯЗАН присутствовать в списке столбцов этой таблицы ниже.
                4. После джойна в накопленную таблицу добавляются все столбцы присоединённой таблицы — учитывай это для следующих шагов.
                5. Нельзя джойнить по столбцу, которого ещё нет в накопленной таблице. Если таблица требует столбец, который появится только после другого джойна — поставь её позже.
                6. Каждую таблицу можно присоединить не более одного раза.
                7. Перед выдачей ответа мысленно проверь КАЖДЫЙ шаг: on_col1 действительно есть в накопленной таблице на этом шаге, on_col2 действительно есть в списке столбцов присоединяемой таблицы. Если проверка не проходит — удали или переставь шаг.

                Доступные таблицы и их столбцы:
                {storage.tables_headers}

                Верни упорядоченную последовательность джойнов строго по схеме JoinConditions. В ответе не должно быть ни data_dictionary.csv, ни test.csv, ни несуществующих столбцов.
"""
    result = structured_llm.invoke(prompt)
    loguru.logger.debug(f"Result from GigaChat: {result}")
    df_train, df_test = merge_tables(joinconditions=result, storage=storage)

    loguru.logger.debug(f"Df_train shape: {df_train.shape}")
    loguru.logger.debug(f"Df_test shape: {df_test.shape}")

    # describe = call function for get stat
    # ask llm for creating new features
    # parse llm's answer (schema for answer?)
    #
    stat = create_stat(df_train, "target")
    description = storage.description

    structured_future_llm = model.with_structured_output(Proposal)
    future_engineering_prompt = f"""
        Придумай новые фичи исходя из описания колонок
        {stat}\n
        {description}\n
    """
    result = structured_future_llm.invoke(future_engineering_prompt)
    loguru.logger.debug(f"Result from GigaChat: {result}")
