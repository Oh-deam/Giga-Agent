import os

import loguru

import pandas as pd
from langchain_gigachat import GigaChat

from .schemas.future import Proposal
from .tools.prompt import PromptFactory
from .utils.storage import Storage
from .schemas.tables import JoinConditions
from .tools.join_tables import merge_tables
from .tools.stat import create_stat
from .tools.future import create_new_futures
from .tools.competition import future_competition


def pipeline(
        model: GigaChat,
        directory: str = "data",
        debug: bool = False,
):
    print("Start pipeline")
    storage = Storage(directory=directory, debug=debug)
    train_columns = storage.get_table("train.csv").columns
    structured_llm = model.with_structured_output(JoinConditions)
    prompt = f"""Посмотри структуру таблиц и определи порядок их объединения с train.csv.

                ДОСТУПНЫЕ ТАБЛИЦЫ И ИХ СТОЛБЦЫ:
                {storage.tables_headers}

                ШАГ 0 — ПОДГОТОВКА (выполни МЫСЛЕННО до генерации, не выводи в ответ):
                1. Для КАЖДОЙ таблицы из списка выше построй у себя в голове множество её столбцов — назови его COLS[<имя_таблицы>]. Это единственный источник имён, которые разрешено подставлять, когда эта таблица фигурирует в FutureProposal.
                2. Заведи множество ACCUMULATED — это колонки, доступные в накопленной таблице по мере построения плана. В начале ACCUMULATED = COLS[train.csv] = {train_columns}.
                3. После каждого запланированного джойна мысленно обнови ACCUMULATED ← ACCUMULATED ∪ COLS[присоединённая_таблица] (минус дубликаты ключей).
                4. Любое имя колонки, которое ты собираешься вписать в ответ, должно браться ПОБУКВЕННО из соответствующего множества. Если имени там нет — его НЕ СУЩЕСТВУЕТ, и ты НЕ имеешь права его использовать. Лучше пропустить таблицу, чем выдумать столбец.

                КРИТИЧЕСКИ ВАЖНО (исключения из плана):
                - Файл data_dictionary.csv (и любые "словарные" файлы с описанием колонок) — это МЕТАДАННЫЕ, а не таблица данных. ПОЛНОСТЬЮ ИСКЛЮЧИ его из результата.
                - Файл test.csv присоединяется отдельно вне твоего плана — НЕ добавляй его в JoinConditions.
                - Если у таблицы НЕТ ни одного общего столбца с текущим ACCUMULATED — эту таблицу в план не включай вообще.

                ПРАВИЛА ПОСТРОЕНИЯ ПОСЛЕДОВАТЕЛЬНОСТИ JoinCondition:
                1. Начинаем с train.csv, её столбцы уже в ACCUMULATED (см. ШАГ 0).
                2. on_col1 ДОЛЖЕН принадлежать ACCUMULATED на момент этого шага. Никогда не бери on_col1 из будущей таблицы.
                3. on_col2 ДОЛЖЕН принадлежать COLS[condition.table_name] — то есть столбцам именно той таблицы, которую сейчас присоединяешь, а не соседней.
                4. Каждую таблицу можно присоединить максимум один раз.
                5. Если таблица требует столбец, который появится только после другого джойна — поставь эту таблицу позже, не раньше.
                6. После того как шаг добавлен в план, обнови ACCUMULATED и переходи к следующему шагу.

                ПРАВИЛА ДЛЯ ПОЛЯ aggregations В КАЖДОМ JoinCondition (это САМАЯ ЧАСТАЯ ТОЧКА ГАЛЛЮЦИНАЦИЙ — будь предельно внимателен):
                A. aggregations.group_by — это СПИСОК СТОЛБЦОВ ТОЛЬКО ИЗ COLS[condition.table_name]. Не из train, не из уже накопленной таблицы, не из соседних справочников. ТОЛЬКО из той таблицы, которую ты сейчас агрегируешь.
                B. on_col2 ОБЯЗАН быть внутри aggregations.group_by (иначе после агрегации он исчезнет и merge не пройдёт).
                C. aggregations.aggregations — это список пар (col_name, method), где КАЖДОЕ col_name берётся ТОЛЬКО из COLS[condition.table_name] и при этом НЕ совпадает ни с одной колонкой из group_by (pandas не даёт агрегировать колонку, по которой группируешь).
                D. Метод агрегации выбирай осмысленно и строго в НИЖНЕМ РЕГИСТРЕ из набора: mean, sum, min, max, count, nunique, median, std, first, last. Если колонка — ключ/идентификатор и агрегация не имеет смысла, используй метод first.
                E. Если в таблице по ключу присоединения (on_col2) уже и так одна строка на ключ (например, users.csv с уникальным user_id) — всё равно заполни aggregations: group_by = [on_col2], а в aggregations положи все остальные столбцы с методом first. Это валидный и безопасный вариант "без реальной агрегации".
                F. Если в таблице много строк на один ключ (как order_items.csv с множеством записей на один order_id) — группируй по ключу присоединения и выбирай адекватные методы (sum/mean/count/nunique и т.д.) для числовых, first — для категориальных идентификаторов.
                G. Пример НЕПРАВИЛЬНОГО поведения, которого НЕ ДОЛЖНО быть: положить в group_by колонку, которой нет в COLS[condition.table_name] (типичная ошибка — взять 'order_id' в группировку таблицы users.csv, где этой колонки нет).

                ФИНАЛЬНАЯ САМОПРОВЕРКА (выполни до выдачи ответа):
                Пройдись по своему плану по порядку. Для каждого JoinCondition проверь по пунктам:
                  (1) condition.table_name не равен data_dictionary.csv и не равен test.csv;
                  (2) on_col1 ∈ ACCUMULATED на момент этого шага;
                  (3) on_col2 ∈ COLS[condition.table_name];
                  (4) aggregations.group_by ⊆ COLS[condition.table_name] И on_col2 ∈ aggregations.group_by;
                  (5) каждое aggregations.aggregations[i].col_name ∈ COLS[condition.table_name] И отсутствует в group_by;
                  (6) каждый method записан строчными буквами и входит в допустимый набор методов.
                Если ХОТЯ БЫ ОДИН пункт не выполняется — УДАЛИ или ПЕРЕСТАВЬ этот JoinCondition перед выдачей ответа. Сокращай план, но НЕ выдумывай колонки.

                Верни упорядоченную последовательность джойнов строго по схеме JoinConditions. В ответе не должно быть ни data_dictionary.csv, ни test.csv, ни несуществующих столбцов, ни методов агрегации в верхнем регистре.
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

    best_train_df, best_test_df = future_competition(model, df_train, df_test, storage)
    os.makedirs("./output", exist_ok=True)
    best_train_df.to_csv(f"./output/train.csv", index=False)
    best_test_df.to_csv(f"./output/test.csv", index=False)




