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

                ДОСТУПНЫЕ ТАБЛИЦЫ И ИХ СТОЛБЦЫ:
                {storage.tables_headers}=

                ШАГ 0 — ПОДГОТОВКА (выполни МЫСЛЕННО до генерации, не выводи в ответ):
                1. Для КАЖДОЙ таблицы из списка выше построй у себя в голове множество её столбцов — назови его COLS[<имя_таблицы>]. Это единственный источник имён, которые разрешено подставлять, когда эта таблица фигурирует в FutureProposal.
                2. Заведи множество ACCUMULATED — это колонки, доступные в накопленной таблице по мере построения плана. В начале ACCUMULATED = COLS[train.csv] = {{row_id, user_id, product_id, target}}.
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
    stat = create_stat(df_train, "target")
    description = storage.description

    structured_future_llm = model.with_structured_output(Proposal)
    future_engineering_prompt = f"""
Ты — опытный ML-инженер. Задача — бинарная классификация (target ∈ {{0, 1}}).
Твоя цель: сгенерировать НАБОР новых фичей в виде списка FutureProposal, которые помогут модели машинного обучения лучше разделять классы.
СТАТИСТИКА ДАТАСЕТА:
{stat}
ОПИСАНИЕ КОЛОНОКЖ:
{description}

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

Верни результат СТРОГО в формате схемы Proposal (МЕТОДЫ АГРЕГИРОВАНИЯ СТРОГО В НИЖНЕМ РЕГИСТРЕ): заполни поле 'proposal' списком FutureProposal. Никакого текста вне схемы.
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