import loguru

import pandas as pd
from langchain_gigachat import GigaChat

from .schemas.future import Proposal
from .utils.storage import Storage
from .schemas.tables import JoinConditions
from .tools.join_tables import merge_tables
from .tools.stat import create_stat
from .tools.future import create_new_futures


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
                4. Если перед джойном необходимо провести агрегацию, напиши по каким столбцам и методы агрегации для оставшихся столбцов. Если агрегация для какого-либо столбца не нужна, напиши в методе First 
                5. После джойна в накопленную таблицу добавляются все столбцы присоединённой таблицы — учитывай это для следующих шагов.
                6. Нельзя джойнить по столбцу, которого ещё нет в накопленной таблице. Если таблица требует столбец, который появится только после другого джойна — поставь её позже.
                7. Каждую таблицу можно присоединить не более одного раза.
                8. Перед выдачей ответа мысленно проверь КАЖДЫЙ шаг: on_col1 действительно есть в накопленной таблице на этом шаге, on_col2 действительно есть в списке столбцов присоединяемой таблицы. Если проверка не проходит — удали или переставь шаг.
                9. Есть таблица order_items.csv, в ней много одинаковый order_id, значит нужно сгруппировать по нему и как-то агрегировать оставшиеся столбцы 
                Доступные таблицы и их столбцы:
                {storage.tables_headers}

                Верни упорядоченную последовательность джойнов строго по схеме JoinConditions (НАЗВАНИЯ МЕТОДОВ АГРЕГИРОВАНИЯ В НИЖНЕМ РЕГИСТРЕ). В ответе не должно быть ни data_dictionary.csv, ни test.csv, ни несуществующих столбцов.
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
Ты — опытный ML-инженер. Задача — бинарная классификация (target ∈ {{0, 1}}).
Твоя цель: сгенерировать НАБОР новых фичей в виде списка FutureProposal, которые помогут модели машинного обучения лучше разделять классы.

========== СТАТИСТИКА ДАТАСЕТА ==========
{stat}

========== ОПИСАНИЕ КОЛОНОК И ДОМЕННАЯ ОБЛАСТЬ ==========
{description}
==========================================

КАК УСТРОЕНА СХЕМА:
Каждый FutureProposal — это ОДНО бинарное (или унарное) действие над колонками:
  col1  — имя существующей колонки (или временной, созданной в предыдущем FutureProposal из этого же ответа)
  col2  — имя другой колонки ИЛИ числовая константа (для Logarithm значение col2 игнорируется, но поле обязано быть заполнено — передай 0)
  action — одно из: Addition, Subtraction, Multiplication, Division, Degree, Logarithm
  new_col_name — короткое уникальное snake_case имя результата
  rationale — 1–3 предложения на русском С ССЫЛКОЙ на конкретные числа из статистики
  save_col — True, если колонка нужна в финальном датасете; False, если это промежуточный шаг

ЦЕПОЧКИ (СЛОЖНЫЕ ФИЧИ):
Сложная фича собирается ПОСЛЕДОВАТЕЛЬНОСТЬЮ FutureProposal. Промежуточные шаги помечай save_col=False, финальный — save_col=True. Порядок в списке имеет значение: в col1 последующего шага можно ссылаться ТОЛЬКО на new_col_name, уже появившийся в предыдущих шагах (или на исходную колонку датасета).
(ПРОБУЙ ИНОГДА ГНЕНЕРИРОВАТЬ ФИЧИ ИЗ НОВЫХ ФИЧЕЙ)

Пример цепочки "средняя цена товара в корзине":
  1) FutureProposal(col1='items_price_sum', col2='items_count', new_col_name='avg_item_price', action=Division, rationale='...', save_col=True)
Пример из 3 шагов:
  1) col1='a', col2='b', action=Addition, new_col_name='a_plus_b', save_col=False
  2) col1='a_plus_b', col2='c', action=Multiplication, new_col_name='ab_mul_c', save_col=False
  3) col1='ab_mul_c', col2=2, action=Degree, new_col_name='ab_mul_c_sq', save_col=True

ЖЁСТКИЕ ПРАВИЛА:
1. Используй ТОЛЬКО колонки, РЕАЛЬНО присутствующие в разделе "ОБЩАЯ ИНФОРМАЦИЯ" статистики. Имена — побуквенно, без переименований. Выдумывать колонки запрещено.
2. НИКОГДА не ставь 'target' в col1 или col2 — это утечка таргета.
3. Для Division выбирай знаменатели, которые по describe() не содержат нулей (или где это редкость) — иначе результат бессмыслен.
4. Logarithm применяй только к положительным числовым колонкам со скошенным распределением (mean заметно отличается от median, высокий max, много выбросов по 1.5·IQR).
5. Degree со степенью 2/3 используй для нелинейных эффектов; со степенью 0.5 — как альтернатива логарифму.
6. new_col_name должны быть уникальными в пределах всего ответа.
7. rationale ОБЯЗАН ссылаться на конкретные наблюдения из статистики: "корреляция X и Y = 0.74 → их произведение усилит совместный сигнал", "средние X по классам отличаются в 2.3 раза → нелинейное преобразование поможет", "пропусков нет, std/mean = 1.8 → распределение скошенное, подходит Logarithm" и т.п.
8. Верни ОТ 8 ДО 15 FutureProposal в поле proposal. Среди них должно быть как минимум: одна Division/Ratio, одна Multiplication, одна Logarithm или Degree, и хотя бы одна цепочка из 2+ шагов.
9. fields обязательно должен содержать как минимум два элемента! Например ["col_1", "col_3"] или ["col_2", 2] 
ПРИОРИТИЗАЦИЯ (в порядке важности):
- Пары из раздела "КОРРЕЛЯЦИЯ PEARSON — ТОП-10 ПАР" — кандидаты на Multiplication / Division.
- Колонки, у которых в "ВЗАИМОСВЯЗЬ ЧИСЛОВЫХ ФИЧЕЙ С ТАРГЕТОМ" средние по классам 0 и 1 заметно различаются — их нелинейные преобразования и взаимодействия должны усилить сигнал.
- Колонки с высокой долей выбросов (раздел "ВЫБРОСЫ") — кандидаты на Logarithm или Degree с дробной степенью.
- Содержательные отношения из доменного описания (например, "сумма по заказам" / "число заказов" = средний чек).

Верни результат СТРОГО в формате схемы Proposal: заполни поле 'proposal' списком FutureProposal. Никакого текста вне схемы.
"""
    loguru.logger.debug(f"Len prompt: {len(future_engineering_prompt)}")
    loguru.logger.debug(f"Future prompt: {future_engineering_prompt}")
    result = structured_future_llm.invoke(future_engineering_prompt)
    loguru.logger.debug(f"Result from GigaChat: {result}")

    for prop in result.proposal:
        loguru.logger.debug(f"Proposal: {prop}")

    loguru.logger.debug(f"New futures: {len(result.proposal)}")
    new_df = create_new_futures(df_train, result)
    loguru.logger.debug(f"After future engineering: {new_df.shape[1]}")
    print(new_df.head())