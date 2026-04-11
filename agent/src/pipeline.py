import os

import loguru

import pandas as pd
from pandas.api.types import is_numeric_dtype
from langchain_gigachat import GigaChat

from .schemas.future import Proposal
from .utils.storage import Storage
from .schemas.tables import JoinConditions
from .tools.join_tables import merge_tables
from .tools.stat import create_stat
from .tools.future import create_new_futures
from .tools.selection import select_top_features
from .tools.aggregates import generate_aggregate_feature_pool
from .tools.event_history import generate_event_history_features
from .config.config import config


MAX_RAW_CAT_CARDINALITY = 200
EXCLUDED_RAW_COLUMNS = {"target", "eval_set"}


def _is_identifier_column(series: pd.Series, col_name: str) -> bool:
    lower = col_name.lower()
    if lower == "id" or lower.endswith("_id"):
        return True
    if len(series) == 0:
        return False
    unique_ratio = series.nunique(dropna=False) / max(len(series), 1)
    return unique_ratio >= 0.95


def _is_output_id_column(series: pd.Series, col_name: str) -> bool:
    lower = col_name.lower()
    if lower in {"id", "row_id"}:
        return True
    unique_ratio = series.nunique(dropna=False) / max(len(series), 1)
    if unique_ratio >= 0.95:
        return True
    return lower.endswith("_id") and unique_ratio >= 0.95


def _select_output_id_columns(input_train: pd.DataFrame, input_test: pd.DataFrame) -> list[str]:
    common_cols = [col for col in input_train.columns if col in input_test.columns]
    id_cols = [col for col in common_cols if _is_output_id_column(input_train[col], col)]
    if id_cols:
        return id_cols
    return common_cols[:1]


def _build_raw_feature_pool(
        merged_train: pd.DataFrame,
        merged_test: pd.DataFrame,
        target_col: str = "target",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_base = merged_train.drop(columns=[target_col], errors="ignore").reset_index(drop=True)
    test_base = merged_test.drop(columns=[target_col], errors="ignore").reset_index(drop=True)

    selected_cols: list[str] = []
    for col in train_base.columns:
        if col in EXCLUDED_RAW_COLUMNS:
            continue
        train_series = train_base[col]
        if _is_identifier_column(train_series, col):
            continue
        if train_series.nunique(dropna=False) <= 1:
            continue
        if is_numeric_dtype(train_series):
            selected_cols.append(col)
            continue
        nunique = train_series.astype("string").nunique(dropna=True)
        if 1 < nunique <= MAX_RAW_CAT_CARDINALITY:
            selected_cols.append(col)

    return (
        train_base[selected_cols].copy(),
        test_base[selected_cols].copy(),
        selected_cols,
    )


def _build_feature_feedback_block(
        selection_result,
        selected_feature_names: list[str],
        existing_feature_names: list[str],
) -> str:
    pool_score = (
        f"{selection_result.pool_validation_score:.6f}"
        if selection_result.pool_validation_score is not None
        else "n/a"
    )
    top5_score = (
        f"{selection_result.validation_score:.6f}"
        if selection_result.validation_score is not None
        else "n/a"
    )
    importance_items = sorted(
        selection_result.feature_importance.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    top_lines = [f"- {name}: {value:.6f}" for name, value in importance_items[:8]]
    weak_lines = [f"- {name}: {value:.6f}" for name, value in importance_items[-5:]]
    return f"""
РЕЗУЛЬТАТ ПРЕДЫДУЩЕЙ ИТЕРАЦИИ НА TRAIN-ONLY VALIDATION:
- score CatBoost ROC-AUC на всем candidate pool: {pool_score}
- score CatBoost ROC-AUC на финальных top-5: {top5_score}
- выбранные фичи сейчас: {selected_feature_names}
- top importances:
{chr(10).join(top_lines) if top_lines else '- нет'}
- weakest importances:
{chr(10).join(weak_lines) if weak_lines else '- нет'}

ЗАДАЧА НОВОЙ ИТЕРАЦИИ:
- предложи ДО 8 НОВЫХ ИЛИ УЛУЧШЕННЫХ фич;
- можно использовать как исходные колонки, так и уже созданные фичи из списка выше;
- НЕ дублируй существующие new_col_name;
- уже занятые new_col_name: {existing_feature_names}
- если текущий набор уже достаточно хорош и разумных улучшений нет, верни пустой список proposal.
"""


def pipeline(
        model: GigaChat,
        directory: str = "data",
        debug: bool = False,
):
    print("Start pipeline")
    storage = Storage(directory=directory, debug=debug)
    input_train = storage.get_table("train.csv").copy()
    input_test = storage.get_table("test.csv").copy()
    train_columns = storage.get_columns("train.csv")

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

    raw_train_features, raw_test_features, raw_feature_names = _build_raw_feature_pool(
        merged_train=df_train,
        merged_test=df_test,
        target_col="target",
    )
    loguru.logger.info(f"Raw merged candidate features: {len(raw_feature_names)}")

    deterministic_train_features, deterministic_test_features, deterministic_feature_names = generate_aggregate_feature_pool(
        input_train=input_train,
        input_test=input_test,
        merged_train=df_train,
        merged_test=df_test,
        target_col="target",
    )
    loguru.logger.info(f"Deterministic aggregate features: {len(deterministic_feature_names)}")

    history_train_features, history_test_features, history_feature_names = generate_event_history_features(
        storage=storage,
        input_train=input_train,
        input_test=input_test,
        merged_train=df_train,
        merged_test=df_test,
    )
    loguru.logger.info(f"Event history features: {len(history_feature_names)}")

    # describe = call function for get stat
    # ask llm for creating new features
    # parse llm's answer (schema for answer?)
    #
    stat = create_stat(df_train, "target")
    description = storage.description

    structured_future_llm = model.with_structured_output(Proposal)
    base_future_engineering_prompt = """
Ты — опытный ML-инженер. Задача — бинарная классификация (target ∈ {{0, 1}}).
Твоя цель: сгенерировать ПУЛ кандидатов новых фичей в виде списка FutureProposal, которые помогут модели машинного обучения лучше разделять классы.
После этого локальный CatBoost сам отберет лучшие 5 фич по train, поэтому ты можешь предложить больше кандидатов, если они осмысленны.
СТАТИСТИКА ДАТАСЕТА:
{stat}
ОПИСАНИЕ КОЛОНОКЖ:
{description}
{feedback_block}

КАК УСТРОЕНА СХЕМА:
Каждый FutureProposal описывается полями:
  fields      — список входов. Первый элемент ВСЕГДА существующая колонка или промежуточная колонка из предыдущего шага.
  actions     — список действий той же длины, что и число применяемых шагов.
  new_col_name — короткое уникальное snake_case имя результата.
  reason      — 1–3 предложения на русском с привязкой к статистике.
  save_col    — True, если колонка должна попасть в финальный candidate pool; False, если это промежуточный шаг для цепочки.

ДОСТУПНЫЕ actions:
  Addition, Subtraction, Multiplication, Division, Degree, Logarithm,
  Threshold, BinaryCategory, StringConcat, IsMissing

СЕМАНТИКА actions:
  Addition/Subtraction/Multiplication/Division/Degree:
    fields = [col_or_tmp, other_col_or_number]
  Logarithm:
    fields = [col_or_tmp]
  Threshold:
    fields = [numeric_col_or_tmp, threshold_number]
    результат = 1, если значение >= threshold, иначе 0
  BinaryCategory:
    fields = [categorical_col_or_tmp, category_value]
    результат = 1, если значение равно category_value, иначе 0
  StringConcat:
    fields = [cat_col_or_tmp, other_cat_col_or_value]
    результат = конкатенация строк через "__"
  IsMissing:
    fields = [col_or_tmp]
    результат = 1, если значение пропущено, иначе 0

ЦЕПОЧКИ (СЛОЖНЫЕ ФИЧИ):
Сложная фича собирается ПОСЛЕДОВАТЕЛЬНОСТЬЮ FutureProposal. Промежуточные шаги помечай save_col=False, финальный — save_col=True. Порядок в списке имеет значение: в первом элементе fields последующего шага можно ссылаться ТОЛЬКО на new_col_name, уже появившийся в предыдущих шагах (или на исходную колонку датасета).
(ПРОБУЙ ИНОГДА ГНЕНЕРИРОВАТЬ ФИЧИ ИЗ НОВЫХ ФИЧЕЙ)

Пример цепочки "средняя цена товара в корзине":
  1) FutureProposal(fields=['items_price_sum', 'items_count'], actions=['Division'], new_col_name='avg_item_price', reason='...', save_col=True)
Пример из 3 шагов:
  1) fields=['a', 'b'], actions=['Addition'], new_col_name='a_plus_b', save_col=False
  2) fields=['a_plus_b', 'c'], actions=['Multiplication'], new_col_name='ab_mul_c', save_col=False
  3) fields=['ab_mul_c', 2], actions=['Degree'], new_col_name='ab_mul_c_sq', save_col=True
Пример бинаризации:
  1) fields=['pdays', 999], actions=['Threshold'], new_col_name='pdays_ge_999', save_col=True
Пример бинарной категориальной:
  1) fields=['default', 'yes'], actions=['BinaryCategory'], new_col_name='default_is_yes', save_col=True
Пример кросса категорий:
  1) fields=['job', 'education'], actions=['StringConcat'], new_col_name='job_x_education', save_col=True

ЖЁСТКИЕ ПРАВИЛА:
1. Используй ТОЛЬКО колонки, РЕАЛЬНО присутствующие в разделе "ОБЩАЯ ИНФОРМАЦИЯ" статистики. Имена — побуквенно, без переименований. Выдумывать колонки запрещено.
2. НИКОГДА не используй 'target' в fields — это утечка таргета.
3. Для Division выбирай знаменатели, которые по describe() не содержат нулей (или где это редкость) — иначе результат бессмыслен.
4. Logarithm применяй только к положительным числовым колонкам со скошенным распределением (mean заметно отличается от median, высокий max, много выбросов по 1.5·IQR).
5. Degree со степенью 2/3 используй для нелинейных эффектов; со степенью 0.5 — как альтернатива логарифму.
6. new_col_name должны быть уникальными в пределах всего ответа.
7. reason ОБЯЗАН ссылаться на конкретные наблюдения из статистики: "корреляция X и Y = 0.74 → их произведение усилит совместный сигнал", "средние X по классам отличаются в 2.3 раза → нелинейное преобразование поможет", "пропусков нет, std/mean = 1.8 → распределение скошенное, подходит Logarithm" и т.п.
8. Верни ОТ 8 ДО 12 FutureProposal в поле proposal. Среди них должно быть как минимум:
   - одна Division/Ratio,
   - одна Multiplication,
   - одна Logarithm или Degree,
   - одна Threshold или IsMissing,
   - одна BinaryCategory или StringConcat,
   - хотя бы одна цепочка из 2+ шагов.
9. Для unary actions (Logarithm, IsMissing) fields содержит ровно 1 элемент. Для остальных действий fields содержит ровно 2 элемента.
ПРИОРИТИЗАЦИЯ (в порядке важности):
- Пары из раздела "КОРРЕЛЯЦИЯ PEARSON — ТОП-10 ПАР" — кандидаты на Multiplication / Division.
- Колонки, у которых в "ВЗАИМОСВЯЗЬ ЧИСЛОВЫХ ФИЧЕЙ С ТАРГЕТОМ" средние по классам 0 и 1 заметно различаются — их нелинейные преобразования и взаимодействия должны усилить сигнал.
- Колонки с высокой долей выбросов (раздел "ВЫБРОСЫ") — кандидаты на Logarithm или Degree с дробной степенью.
- Числовые колонки со специальными значениями (например 0, 1, 999) и явными разрывами распределения — кандидаты на Threshold.
- Бинарные категориальные признаки и осмысленные пары категорий низкой кардинальности — кандидаты на BinaryCategory и StringConcat.
- Содержательные отношения из доменного описания (например, "сумма по заказам" / "число заказов" = средний чек).

Верни результат СТРОГО в формате схемы Proposal: заполни поле 'proposal' списком FutureProposal. Никакого текста вне схемы.
"""

    base_train_features = pd.concat(
        [raw_train_features, deterministic_train_features, history_train_features],
        axis=1,
    )
    base_test_features = pd.concat(
        [raw_test_features, deterministic_test_features, history_test_features],
        axis=1,
    )
    base_train_features = base_train_features.loc[:, ~base_train_features.columns.duplicated()]
    base_test_features = base_test_features.loc[:, ~base_test_features.columns.duplicated()]

    best_train_features = pd.DataFrame(index=input_train.index)
    best_test_features = pd.DataFrame(index=input_test.index)
    best_score = float("-inf")
    feedback_block = ""

    if not base_train_features.empty:
        loguru.logger.info("Start baseline feature package selection on train")
        baseline_selection = select_top_features(
            train_df=pd.concat([input_train.reset_index(drop=True), base_train_features.reset_index(drop=True)], axis=1),
            target_col="target",
            candidate_features=base_train_features.columns.tolist(),
            max_output_features=5,
        )
        loguru.logger.info(f"Baseline candidate features: {baseline_selection.candidate_features}")
        loguru.logger.info(f"Baseline selected features: {baseline_selection.selected_features}")
        loguru.logger.info(f"Baseline top-5 validation score: {baseline_selection.validation_score}")
        if baseline_selection.selected_features and baseline_selection.validation_score is not None:
            best_score = baseline_selection.validation_score
            best_train_features = base_train_features[baseline_selection.selected_features].copy()
            best_test_features = base_test_features[baseline_selection.selected_features].copy()
            feedback_block = _build_feature_feedback_block(
                baseline_selection,
                baseline_selection.selected_features,
                existing_feature_names=[],
            )

    for round_idx in range(1, config.FEATURE_SEARCH_ROUNDS + 1):
        future_engineering_prompt = base_future_engineering_prompt.format(
            stat=stat,
            description=description,
            feedback_block=feedback_block,
        )
        loguru.logger.info(f"Start feature generation round {round_idx}/{config.FEATURE_SEARCH_ROUNDS}")
        result = structured_future_llm.invoke(future_engineering_prompt)
        loguru.logger.debug(f"Result from GigaChat round {round_idx}: {result}")

        if not result.proposal:
            loguru.logger.info(f"GigaChat returned empty proposal list on round {round_idx}, stopping feature search")
            break

        for prop in result.proposal:
            loguru.logger.debug(f"Proposal round {round_idx}: {prop}")

        llm_train_features = create_new_futures(df_train, result)
        llm_test_features = create_new_futures(df_test, result)
        train_features = pd.concat([base_train_features, llm_train_features], axis=1)
        test_features = pd.concat([base_test_features, llm_test_features], axis=1)
        train_features = train_features.loc[:, ~train_features.columns.duplicated()]
        test_features = test_features.loc[:, ~test_features.columns.duplicated()]
        loguru.logger.debug(f"After future engineering round {round_idx}: {train_features.shape[1]}")

        if train_features.empty:
            continue

        loguru.logger.info(f"Start CatBoost feature selection on train for round {round_idx}")
        selection_result = select_top_features(
            train_df=pd.concat([input_train.reset_index(drop=True), train_features.reset_index(drop=True)], axis=1),
            target_col="target",
            candidate_features=train_features.columns.tolist(),
            max_output_features=5,
        )
        selected_feature_names = selection_result.selected_features
        loguru.logger.info(f"Candidate features round {round_idx}: {selection_result.candidate_features}")
        loguru.logger.info(f"Prefiltered features round {round_idx}: {selection_result.prefiltered_features}")
        loguru.logger.info(f"Selected features round {round_idx}: {selected_feature_names}")
        loguru.logger.info(f"Pool validation score round {round_idx}: {selection_result.pool_validation_score}")
        loguru.logger.info(f"Top-5 validation score round {round_idx}: {selection_result.validation_score}")

        if selected_feature_names and selection_result.validation_score is not None:
            if selection_result.validation_score > best_score:
                best_score = selection_result.validation_score
                best_train_features = train_features[selected_feature_names].copy()
                best_test_features = test_features[selected_feature_names].copy()

        if round_idx == config.FEATURE_SEARCH_ROUNDS:
            break

        feedback_block = _build_feature_feedback_block(
            selection_result,
            selected_feature_names,
            existing_feature_names=[prop.new_col_name for prop in result.proposal],
        )

    if best_score == float("-inf") and not base_train_features.empty:
        loguru.logger.info("No valid LLM feature round was selected, fallback to baseline feature package only")
        fallback_selection = select_top_features(
            train_df=pd.concat([input_train.reset_index(drop=True), base_train_features.reset_index(drop=True)], axis=1),
            target_col="target",
            candidate_features=base_train_features.columns.tolist(),
            max_output_features=5,
        )
        fallback_features = fallback_selection.selected_features
        best_train_features = base_train_features[fallback_features].copy() if fallback_features else pd.DataFrame(index=input_train.index)
        best_test_features = base_test_features[fallback_features].copy() if fallback_features else pd.DataFrame(index=input_test.index)

    output_id_cols = _select_output_id_columns(input_train, input_test)
    train_df = pd.concat(
        [
            input_train[output_id_cols].reset_index(drop=True),
            input_train[["target"]].reset_index(drop=True),
            best_train_features.reset_index(drop=True),
        ],
        axis=1,
    )
    test_df = pd.concat(
        [
            input_test[output_id_cols].reset_index(drop=True),
            best_test_features.reset_index(drop=True),
        ],
        axis=1,
    )

    os.makedirs("./output", exist_ok=True)
    train_df.to_csv("./output/train.csv", index=False)
    test_df.to_csv("./output/test.csv", index=False)
