# Архитектура AI-агента для генерации признаков
## Хакатон "Код Риска" — Детальная спецификация

---

## 1. Общая философия решения

### 1.1 Принципы проектирования

**Детерминированный контроллер + LLM-мозг.** Оркестратор — это обычный Python-код с жёстким потоком управления (if/else, циклы, таймеры). LLM вызывается только в точках, где нужен интеллект: понимание данных, генерация идей, написание кода. Это даёт предсказуемость и контроль над временем.

**Fail-safe на каждом уровне.** Каждый компонент, зависящий от LLM, имеет детерминированный fallback. Если GigaChat не ответил, ответил мусором или код с ошибкой — система не падает, а переключается на безопасный вариант. Финальный fallback — набор простейших статистических фичей, которые гарантированно работают.

**Бюджет времени как первый приоритет.** TimerManager контролирует каждый шаг. Если осталось мало времени — агент немедленно сохраняет лучшее, что есть. Лучше 3 посредственных фичи, чем таймаут и 0 баллов.

**Одна функция — один контракт.** Код генерации фичей оформлен как единственная Python-функция `generate_features(train_df, test_df) -> (train_out, test_out)`. Это гарантирует идентичную логику для train и test, исключает data leakage.

### 1.2 Почему не чистый LLM-агент (ReAct loop)

Чистый ReAct (LLM сам решает, что делать на каждом шаге) опасен в условиях хакатона:
- Непредсказуемое время: LLM может "зависнуть" в цикле рассуждений
- Непредсказуемый результат: может решить делать что-то бесполезное
- Трудно дебажить: каждый запуск уникален

Вместо этого — **управляемый pipeline** с LLM-вызовами в конкретных точках. Мы точно знаем, сколько раз будет вызван LLM, сколько раундов генерации будет, и можем это контролировать.

---

## 2. Структура проекта (файловое дерево)

```
FeaturesAgentTemplate/
│
├── run.py                           # Единственная точка входа
├── .env                             # GIGACHAT_CREDENTIALS, GIGACHAT_MODEL
├── .env.example                     # Шаблон .env
├── pyproject.toml                   # Зависимости проекта
│
├── configs/
│   └── settings.py                  # Все константы и конфигурация
│
├── src/
│   ├── __init__.py
│   │
│   ├── orchestrator.py              # Главный контроллер пайплайна
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py                # Обёртка над GigaChat API
│   │   └── prompts.py               # Все промпты (шаблоны)
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── explorer.py              # Разведка данных (без LLM)
│   │   ├── profiler.py              # Профилирование таблиц
│   │   ├── join_planner.py          # Планирование join'ов (LLM)
│   │   └── preparer.py              # Выполнение join'ов и подготовка
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── idea_generator.py        # Генерация идей фичей (LLM)
│   │   ├── code_generator.py        # Генерация Python-кода (LLM)
│   │   ├── code_executor.py         # Выполнение кода в sandbox
│   │   ├── evaluator.py             # CatBoost CV оценка
│   │   ├── selector.py              # Выбор лучшего набора
│   │   └── fallback.py              # Аварийные фичи (без LLM)
│   │
│   ├── output/
│   │   ├── __init__.py
│   │   ├── finalizer.py             # Сохранение результатов
│   │   └── validator.py             # Валидация выходных файлов
│   │
│   └── utils/
│       ├── __init__.py
│       ├── timer.py                 # Контроль бюджета времени
│       ├── logger.py                # Структурированное логирование
│       ├── scoring.py               # Скрипт скоринга (из шаблона)
│       └── check_submission.py      # Проверка сабмита (из шаблона)
│
├── data/                            # Входные данные (монтируется)
│   ├── readme.txt
│   ├── train.csv
│   ├── test.csv
│   └── *.csv                        # Доп. таблицы
│
└── output/                          # Выходные данные
    ├── train.csv
    └── test.csv
```

---

## 3. Детальное описание каждого модуля

---

### 3.1 `run.py` — Точка входа

**Ответственность:** загрузить переменные окружения, создать оркестратор, запустить его, обработать критические ошибки верхнего уровня.

```python
# Псевдокод
def main():
    load_dotenv()

    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    model = os.getenv("GIGACHAT_MODEL", "GigaChat-2-Max")

    orchestrator = Orchestrator(
        credentials=credentials,
        model=model,
        data_dir="data",
        output_dir="output",
        time_budget=580  # 580 вместо 600 — запас 20 сек
    )

    try:
        orchestrator.run()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        # Аварийное сохранение fallback-фичей
        orchestrator.emergency_save()

if __name__ == "__main__":
    main()
```

**Важные детали:**
- `time_budget=580` — оставляем 20 секунд запаса на загрузку окружения, сохранение файлов, непредвиденные задержки
- `emergency_save()` — даже при полном крахе агента сохраняем хоть что-то через FallbackFeatureGenerator

---

### 3.2 `configs/settings.py` — Конфигурация

**Ответственность:** все магические числа, лимиты, пороги в одном месте.

```python
# Псевдокод
class Settings:
    # Время
    TOTAL_TIME_BUDGET = 580          # секунд
    MIN_TIME_FOR_ROUND = 120         # если меньше — не начинать новый раунд
    MIN_TIME_FOR_SAVE = 30           # минимум на финализацию
    CODE_EXECUTION_TIMEOUT = 60      # таймаут выполнения сгенерированного кода
    LLM_CALL_TIMEOUT = 30            # таймаут одного вызова LLM

    # Раунды
    MAX_ROUNDS = 3                   # максимум раундов генерации
    MAX_RETRIES_PER_ROUND = 3        # макс. попыток исправить код в раунде
    MAX_FEATURES = 5                 # ограничение по условию задачи

    # Данные
    DATA_DIR = "data"
    OUTPUT_DIR = "output"
    README_FILENAME = "readme.txt"
    MAX_SAMPLE_ROWS = 5              # строк в sample для промпта
    MAX_PROFILE_CATEGORIES = 10      # топ категорий в профиле
    MAX_README_CHARS = 3000          # обрезаем readme для промпта

    # Оценка
    CV_FOLDS = 5
    CV_RANDOM_SEED = 42
    CATBOOST_ITERATIONS = 500
    CATBOOST_VERBOSE = 0

    # LLM
    GIGACHAT_MODEL = "GigaChat-2-Max"
    GIGACHAT_TEMPERATURE = 0.3       # ниже → более детерминированные ответы
    GIGACHAT_MAX_TOKENS = 4096
```

---

### 3.3 `src/utils/timer.py` — TimerManager

**Ответственность:** единый источник правды о времени. Все компоненты спрашивают у него "сколько осталось" и "можно ли начинать X".

```python
# Псевдокод
class TimerManager:
    def __init__(self, budget: float):
        self.budget = budget
        self.start_time = time.time()
        self.checkpoints: list[tuple[str, float]] = []

    def elapsed(self) -> float:
        """Сколько секунд прошло с начала."""
        return time.time() - self.start_time

    def remaining(self) -> float:
        """Сколько секунд осталось."""
        return max(0, self.budget - self.elapsed())

    def can_start_round(self) -> bool:
        """Хватит ли времени на ещё один полный раунд."""
        return self.remaining() > Settings.MIN_TIME_FOR_ROUND

    def can_save(self) -> bool:
        """Хватит ли времени хотя бы на сохранение."""
        return self.remaining() > Settings.MIN_TIME_FOR_SAVE

    def must_stop_now(self) -> bool:
        """Критически мало времени — нужно срочно сохранять."""
        return self.remaining() <= Settings.MIN_TIME_FOR_SAVE

    def checkpoint(self, label: str):
        """Зафиксировать контрольную точку."""
        self.checkpoints.append((label, self.elapsed()))

    def report(self) -> str:
        """Отчёт по всем контрольным точкам."""
        lines = []
        prev = 0
        for label, t in self.checkpoints:
            lines.append(f"  {label}: {t:.1f}s (delta: {t - prev:.1f}s)")
            prev = t
        return "\n".join(lines)
```

**Использование в оркестраторе:**
```python
if timer.must_stop_now():
    logger.warning("Time critical — saving best result immediately")
    finalizer.save(state.best_round)
    return

if not timer.can_start_round():
    logger.info("Not enough time for another round — finalizing")
    break
```

---

### 3.4 `src/utils/logger.py` — Logger

**Ответственность:** структурированное логирование всех действий, вызовов LLM, результатов, ошибок.

```python
# Псевдокод
class AgentLogger:
    def __init__(self):
        self.log_entries: list[dict] = []
        # Настраиваем стандартный logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s"
        )
        self.logger = logging.getLogger("agent")

    def step(self, phase: str, message: str):
        """Логировать шаг пайплайна."""

    def llm_call(self, purpose: str, prompt_preview: str,
                 response_preview: str, duration: float):
        """Логировать вызов LLM."""

    def error(self, component: str, error: str, recoverable: bool):
        """Логировать ошибку."""

    def metric(self, name: str, value: float):
        """Логировать метрику (ROC-AUC, время и т.д.)."""

    def summary(self) -> str:
        """Итоговый отчёт."""
```

---

### 3.5 `src/llm/client.py` — LLM Client

**Ответственность:** единая точка общения с GigaChat. Обрабатывает таймауты, ретраи, ошибки API.

```python
# Псевдокод
class GigaChatClient:
    def __init__(self, credentials: str, model: str):
        self.llm = GigaChat(
            credentials=credentials,
            model=model,
            verify_ssl_certs=False,
            timeout=Settings.LLM_CALL_TIMEOUT
        )
        self.call_count = 0
        self.total_tokens = 0

    def ask(self, system_prompt: str, user_prompt: str,
            temperature: float = 0.3,
            max_retries: int = 2) -> str:
        """
        Отправить запрос к GigaChat.

        Возвращает текстовый ответ.
        При ошибке — до max_retries повторов с экспоненциальным backoff.
        При полном провале — выбрасывает LLMError.
        """
        for attempt in range(max_retries + 1):
            try:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                response = self.llm.invoke(messages)
                self.call_count += 1
                return response.content
            except Exception as e:
                if attempt < max_retries:
                    wait = 2 ** attempt  # 1, 2, 4 секунды
                    time.sleep(wait)
                else:
                    raise LLMError(f"GigaChat failed after {max_retries + 1} attempts: {e}")

    def ask_json(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Запрос с ожиданием JSON-ответа.
        Парсит JSON из ответа, обрабатывает markdown-блоки ```json```.
        """
        raw = self.ask(system_prompt, user_prompt)
        return self._parse_json(raw)

    def ask_code(self, system_prompt: str, user_prompt: str) -> str:
        """
        Запрос с ожиданием Python-кода.
        Извлекает код из markdown-блоков ```python```.
        """
        raw = self.ask(system_prompt, user_prompt)
        return self._parse_code(raw)

    def _parse_json(self, text: str) -> dict:
        """Извлечь JSON из текста (с учётом ```json``` блоков)."""
        # 1. Попробовать найти ```json ... ```
        # 2. Попробовать json.loads напрямую
        # 3. Попробовать найти { ... } с балансировкой скобок
        ...

    def _parse_code(self, text: str) -> str:
        """Извлечь Python-код из текста (с учётом ```python``` блоков)."""
        # 1. Найти ```python ... ```
        # 2. Если не нашли — взять весь текст
        # 3. Убрать лишние импорты (os, subprocess, sys)
        ...

    def stats(self) -> dict:
        return {
            "total_calls": self.call_count,
            "total_tokens": self.total_tokens
        }
```

---

### 3.6 `src/llm/prompts.py` — Хранилище промптов

**Ответственность:** все текстовые шаблоны промптов в одном файле. Промпты — это "конфигурация" интеллекта агента.

```python
# Псевдокод — каждый промпт в виде шаблонной строки

# ─────────────── СИСТЕМНЫЕ ПРОМПТЫ ───────────────

SYSTEM_GENERAL = """Ты — эксперт по feature engineering для табличных данных.
Твоя задача — создавать признаки для бинарной классификации, которые
максимизируют ROC-AUC на модели CatBoost.

Строгие правила:
1. НИКОГДА не используй целевую переменную (target) для вычисления признаков
2. Код должен работать ОДИНАКОВО для train и test
3. Все fit-операции только на train, transform на обоих
4. Обрабатывай NaN — заполняй или отфильтровывай
5. Все фичи должны быть числовыми (int или float)
6. Максимум 5 признаков"""


# ─────────────── JOIN PLANNING ───────────────

JOIN_PLANNER_PROMPT = """Вот описание данных:

{readme_text}

Профили таблиц:
{tables_profile}

Определи:
1. По каким ключам нужно соединять таблицы с train/test
2. Тип join (left/inner) для каждого соединения
3. Нужна ли предварительная агрегация перед join
   (если в дополнительной таблице несколько строк на один ключ)

Верни ответ строго в формате JSON:
{{
  "joins": [
    {{
      "right_table": "transactions.csv",
      "join_key": "client_id",
      "join_type": "left",
      "needs_aggregation": true,
      "aggregations": {{
        "amount": ["mean", "sum", "count", "std", "max", "min"],
        "date": ["max", "min"]
      }}
    }}
  ],
  "reasoning": "Почему именно такие join'ы"
}}"""


# ─────────────── FEATURE IDEA GENERATION ───────────────

FEATURE_IDEAS_ROUND1 = """Данные:
{data_summary}

Колонки после join:
{column_info}

Статистики:
{statistics}

Придумай ровно 5 признаков для бинарной классификации.
Сфокусируйся на:
- Агрегации (count, mean, sum, std по группам)
- Отношения между числовыми колонками (ratio A/B)
- Бинарные флаги (превышение порога, наличие/отсутствие)
- Частотное кодирование категорий
- Базовые статистики

Верни JSON-массив:
[
  {{
    "name": "feat_tx_count",
    "description": "Количество транзакций клиента",
    "formula": "transactions.groupby('client_id').size()",
    "reasoning": "Активные клиенты имеют другой профиль риска"
  }},
  ...
]"""


FEATURE_IDEAS_ROUND2 = """Данные:
{data_summary}

Колонки:
{column_info}

В предыдущем раунде были созданы фичи:
{previous_features}
Они дали ROC-AUC = {previous_score:.4f}

Придумай 5 ДРУГИХ признаков, более сложных:
- Полиномиальные взаимодействия (A * B, A² - B²)
- Оконные/временные агрегации (за последние N дней)
- Отклонения от среднего по группам (z-score внутри категории)
- RFM-подобные метрики (recency, frequency, monetary)
- Кросс-табличные статистики

НЕ ПОВТОРЯЙ фичи из предыдущего раунда.
Верни JSON-массив как в прошлый раз."""


FEATURE_IDEAS_ROUND3 = """Данные:
{data_summary}

Лучший ROC-AUC пока: {best_score:.4f}
Лучшие фичи: {best_features}

Попробуй нестандартные подходы:
- Ранги и процентили внутри групп
- Энтропия категориальных переменных
- Расстояния (разность) от центроидов кластеров
- Target encoding с регуляризацией (только на train!)
- Комбинации лучших фичей из прошлых раундов

Верни 5 фичей в JSON."""


# ─────────────── CODE GENERATION ───────────────

CODE_GENERATION_PROMPT = """Напиши Python-функцию, которая генерирует признаки.

Доступные библиотеки: pandas, numpy, scipy.stats
Входные DataFrames уже содержат все колонки после join.

ID колонка: {id_col}
Target колонка (только в train): {target_col}

Колонки с типами:
{column_types}

Примеры данных (первые 3 строки train):
{sample_data}

Признаки для генерации:
{feature_ideas}

Напиши ОДНУ функцию:

```python
import pandas as pd
import numpy as np

def generate_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    '''
    Args:
        train_df: обучающая выборка (с target)
        test_df: тестовая выборка (без target)
    Returns:
        (train_features, test_features):
        train_features — DataFrame с колонками [{id_col}, {target_col}, feat1, ..., feat5]
        test_features — DataFrame с колонками [{id_col}, feat1, ..., feat5]
    '''
    # Твой код здесь
    ...
    return train_features, test_features
```

ПРАВИЛА:
- Функция должна быть самодостаточной (все import внутри или перед ней)
- НЕ используй {target_col} для вычисления признаков
- Обработай NaN через fillna() — CatBoost умеет работать с NaN,
  но лучше заполнить -999 или медианой
- Все фичи должны быть числовыми
- Если fit нужен (scaler, encoder) — fit на train, transform на обоих
- НЕ используй import os, subprocess, sys, shutil"""


CODE_FIX_PROMPT = """Код вызвал ошибку. Исправь его.

Исходный код:
```python
{original_code}
```

Ошибка:
{error_message}

Попытка: {attempt} из {max_attempts}

Верни ПОЛНЫЙ исправленный код функции generate_features.
Не пиши объяснений — только код в блоке ```python```."""
```

---

### 3.7 `src/data/explorer.py` — DataExplorer

**Ответственность:** сканирование папки `data/`, чтение readme, первичная инвентаризация всех файлов. Полностью детерминированный — никаких вызовов LLM.

```python
# Псевдокод
class DataExplorer:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def scan(self) -> ExplorationResult:
        """Главный метод — полная разведка."""

        # 1. Найти все файлы
        files = self._list_files()
        # → ["readme.txt", "train.csv", "test.csv", "transactions.csv"]

        # 2. Прочитать readme
        readme_text = self._read_readme()

        # 3. Определить разделитель каждого CSV
        separators = {}
        for f in csv_files:
            separators[f] = self._detect_separator(f)
            # пробуем ',', ';', '\t', '|'

        # 4. Загрузить все таблицы
        tables = {}
        for f in csv_files:
            tables[f] = pd.read_csv(
                path, sep=separators[f],
                nrows=None,  # читаем целиком
                low_memory=False
            )

        # 5. Определить target и ID
        target_col, id_col = self._detect_target_and_id(
            tables, readme_text
        )

        # 6. Построить профили
        profiles = {}
        for name, df in tables.items():
            profiles[name] = Profiler.profile(df)

        return ExplorationResult(
            readme_text=readme_text,
            tables=tables,
            profiles=profiles,
            target_col=target_col,
            id_col=id_col,
            separators=separators
        )

    def _detect_separator(self, filepath: str) -> str:
        """Определить разделитель CSV."""
        with open(filepath, 'r') as f:
            sample = f.read(4096)
        # csv.Sniffer или подсчёт частоты символов

    def _detect_target_and_id(self, tables, readme) -> tuple:
        """
        Определить целевую переменную и ID-колонку.

        Стратегия:
        1. Ищем в readme ключевые слова: "target", "целевая",
           "метка", "label", "id", "идентификатор"
        2. В train.csv ищем колонку с ровно 2 уникальными значениями
           (0/1) — кандидат на target
        3. В train.csv ищем колонку где nunique == nrows — кандидат на ID
        4. Колонка есть в train, но нет в test — кандидат на target
        """
        train_df = tables.get("train.csv")
        test_df = tables.get("test.csv")

        # Колонки, которые есть в train, но нет в test → target
        train_only_cols = set(train_df.columns) - set(test_df.columns)

        # Среди train_only_cols найти бинарную
        target_candidates = []
        for col in train_only_cols:
            if train_df[col].nunique() == 2:
                target_candidates.append(col)

        # ID — колонка с максимальным nunique
        id_candidates = []
        for col in train_df.columns:
            if train_df[col].nunique() == len(train_df):
                id_candidates.append(col)

        # Парсинг readme для уточнения
        # ...

        return target_col, id_col
```

---

### 3.8 `src/data/profiler.py` — TableProfiler

**Ответственность:** построение детального статистического профиля одной таблицы. Используется для формирования промпта к LLM — чем качественнее профиль, тем умнее фичи придумает LLM.

```python
# Псевдокод
@dataclass
class ColumnProfile:
    name: str
    dtype: str             # "int64", "float64", "object", "datetime64"
    semantic_type: str     # "numeric", "categorical", "datetime", "id", "text"
    nunique: int
    null_count: int
    null_pct: float
    sample_values: list    # 5 случайных значений

    # Только для numeric:
    min: float | None
    max: float | None
    mean: float | None
    median: float | None
    std: float | None
    q25: float | None
    q75: float | None
    skew: float | None

    # Только для categorical:
    top_values: list[tuple[str, int]] | None   # (value, count), top-10
    is_binary: bool

    # Только для datetime:
    date_min: str | None
    date_max: str | None
    date_range_days: int | None

@dataclass
class TableProfile:
    filename: str
    shape: tuple[int, int]     # (rows, cols)
    columns: list[ColumnProfile]
    memory_mb: float
    head_str: str              # df.head(3).to_string()
    possible_keys: list[str]   # колонки с высоким nunique (>80% rows)
    possible_dates: list[str]  # колонки, похожие на даты

class Profiler:
    @staticmethod
    def profile(df: pd.DataFrame, filename: str) -> TableProfile:
        columns = []
        for col in df.columns:
            cp = ColumnProfile(...)

            # Определение semantic_type:
            if df[col].dtype in ['int64', 'float64']:
                if df[col].nunique() == 2:
                    cp.semantic_type = "binary"
                elif df[col].nunique() < 20:
                    cp.semantic_type = "categorical"  # закодированная
                elif df[col].nunique() == len(df):
                    cp.semantic_type = "id"
                else:
                    cp.semantic_type = "numeric"
            elif df[col].dtype == 'object':
                if Profiler._looks_like_date(df[col]):
                    cp.semantic_type = "datetime"
                elif df[col].nunique() > 0.5 * len(df):
                    cp.semantic_type = "text"
                else:
                    cp.semantic_type = "categorical"
            ...

            columns.append(cp)

        return TableProfile(
            filename=filename,
            shape=df.shape,
            columns=columns,
            ...
        )

    @staticmethod
    def to_prompt_string(profile: TableProfile) -> str:
        """Конвертировать профиль в строку для промпта LLM."""
        lines = [f"Таблица: {profile.filename} ({profile.shape[0]} строк, {profile.shape[1]} колонок)"]
        for col in profile.columns:
            line = f"  - {col.name} ({col.semantic_type}, dtype={col.dtype})"
            if col.semantic_type == "numeric":
                line += f" mean={col.mean:.2f}, std={col.std:.2f}, null={col.null_pct:.1f}%"
            elif col.semantic_type == "categorical":
                top3 = [v for v, c in col.top_values[:3]]
                line += f" nunique={col.nunique}, top={top3}, null={col.null_pct:.1f}%"
            elif col.semantic_type == "datetime":
                line += f" range=[{col.date_min} .. {col.date_max}]"
            lines.append(line)
        return "\n".join(lines)
```

---

### 3.9 `src/data/join_planner.py` — JoinPlanner

**Ответственность:** понять, как связать таблицы. Это первый вызов LLM — критически важный шаг.

```python
# Псевдокод
@dataclass
class JoinStep:
    right_table: str           # имя файла
    join_key_left: str         # ключ в основной таблице
    join_key_right: str        # ключ в правой таблице
    join_type: str             # "left" или "inner"
    needs_aggregation: bool
    aggregations: dict | None  # {"col": ["mean","sum","count"]}

@dataclass
class JoinPlan:
    steps: list[JoinStep]
    reasoning: str

class JoinPlanner:
    def __init__(self, llm_client: GigaChatClient):
        self.llm = llm_client

    def plan(self, exploration: ExplorationResult) -> JoinPlan:
        """Спросить LLM, как соединить таблицы."""

        # Если таблица одна (train+test без доп. таблиц) — join не нужен
        extra_tables = [t for t in exploration.tables
                        if t not in ("train.csv", "test.csv")]
        if not extra_tables:
            return JoinPlan(steps=[], reasoning="Нет доп. таблиц")

        # Формируем промпт
        profiles_str = "\n\n".join(
            Profiler.to_prompt_string(exploration.profiles[t])
            for t in exploration.tables
        )
        prompt = PROMPTS.JOIN_PLANNER_PROMPT.format(
            readme_text=exploration.readme_text[:Settings.MAX_README_CHARS],
            tables_profile=profiles_str
        )

        # Вызов LLM
        try:
            response = self.llm.ask_json(
                system_prompt=PROMPTS.SYSTEM_GENERAL,
                user_prompt=prompt
            )
            plan = self._parse_plan(response)
            self._validate_plan(plan, exploration)
            return plan
        except (LLMError, ValidationError) as e:
            logger.error("JoinPlanner", str(e), recoverable=True)
            return self._fallback_plan(exploration)

    def _fallback_plan(self, exploration: ExplorationResult) -> JoinPlan:
        """
        Аварийный план: найти общие колонки между train и каждой
        доп. таблицей, сделать left join с агрегацией mean/count.
        """
        steps = []
        train_cols = set(exploration.tables["train.csv"].columns)

        for name, df in exploration.tables.items():
            if name in ("train.csv", "test.csv"):
                continue
            common_cols = train_cols & set(df.columns)
            if common_cols:
                # Выбрать колонку с наибольшим overlap по значениям
                best_key = self._find_best_key(
                    exploration.tables["train.csv"], df, common_cols
                )
                numeric_cols = df.select_dtypes(include='number').columns
                aggs = {col: ["mean", "count"] for col in numeric_cols
                        if col != best_key}
                steps.append(JoinStep(
                    right_table=name,
                    join_key_left=best_key,
                    join_key_right=best_key,
                    join_type="left",
                    needs_aggregation=True,
                    aggregations=aggs
                ))
        return JoinPlan(steps=steps, reasoning="Fallback: common columns")

    def _validate_plan(self, plan: JoinPlan, exploration):
        """Проверить, что план адекватный."""
        for step in plan.steps:
            # Правая таблица существует
            assert step.right_table in exploration.tables
            # Ключи существуют
            assert step.join_key_right in exploration.tables[step.right_table].columns
            assert step.join_key_left in exploration.tables["train.csv"].columns
```

---

### 3.10 `src/data/preparer.py` — DataPreparer

**Ответственность:** выполнить JoinPlan и получить готовые train_df и test_df с одинаковой структурой.

```python
# Псевдокод
class DataPreparer:
    def prepare(self, exploration: ExplorationResult,
                plan: JoinPlan) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Выполнить join'ы, вернуть (train_joined, test_joined)."""

        train = exploration.tables["train.csv"].copy()
        test = exploration.tables["test.csv"].copy()

        for step in plan.steps:
            right_df = exploration.tables[step.right_table].copy()

            # Предварительная агрегация
            if step.needs_aggregation and step.aggregations:
                agg_dict = {}
                for col, funcs in step.aggregations.items():
                    if col in right_df.columns:
                        for func in funcs:
                            agg_dict[f"{col}_{func}"] = (col, func)

                right_agg = right_df.groupby(step.join_key_right).agg(
                    **{name: pd.NamedAgg(column=col, aggfunc=func)
                       for name, (col, func) in agg_dict.items()}
                ).reset_index()
            else:
                right_agg = right_df

            # Join
            train = train.merge(
                right_agg,
                left_on=step.join_key_left,
                right_on=step.join_key_right,
                how=step.join_type,
                suffixes=('', f'_{step.right_table.split(".")[0]}')
            )
            test = test.merge(
                right_agg,
                left_on=step.join_key_left,
                right_on=step.join_key_right,
                how=step.join_type,
                suffixes=('', f'_{step.right_table.split(".")[0]}')
            )

        # Привести типы
        train = self._cast_types(train)
        test = self._cast_types(test)

        # Убедиться что колонки одинаковые (кроме target)
        self._align_columns(train, test, exploration.target_col)

        return train, test

    def _cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Привести колонки к правильным типам."""
        for col in df.columns:
            # Попытка парсинга дат
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        return df

    def _align_columns(self, train, test, target_col):
        """Убрать из test колонки, которых нет в train, и наоборот."""
        common = set(train.columns) & set(test.columns)
        train_extra = set(train.columns) - common - {target_col}
        test_extra = set(test.columns) - common
        # Удалить лишние
        train.drop(columns=list(train_extra), inplace=True, errors='ignore')
        test.drop(columns=list(test_extra), inplace=True, errors='ignore')

    def get_column_roles(self, df: pd.DataFrame,
                         target_col: str, id_col: str) -> ColumnRoles:
        """Классифицировать колонки по типам."""
        numeric = []
        categorical = []
        datetime_cols = []

        for col in df.columns:
            if col in (target_col, id_col):
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric.append(col)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
            else:
                categorical.append(col)

        return ColumnRoles(
            numeric=numeric,
            categorical=categorical,
            datetime=datetime_cols,
            target=target_col,
            id=id_col
        )
```

---

### 3.11 `src/features/idea_generator.py` — FeatureIdeaGenerator

**Ответственность:** генерация идей признаков через LLM. Получает профиль данных, возвращает структурированный список идей.

```python
# Псевдокод
@dataclass
class FeatureIdea:
    name: str
    description: str
    formula: str
    reasoning: str

class FeatureIdeaGenerator:
    def __init__(self, llm: GigaChatClient):
        self.llm = llm

    def generate(self, data_summary: str, column_info: str,
                 statistics: str, round_num: int,
                 previous_features: str = "",
                 previous_score: float = 0.0) -> list[FeatureIdea]:
        """Сгенерировать 5 идей фичей для данного раунда."""

        # Выбираем шаблон промпта в зависимости от раунда
        if round_num == 1:
            prompt_template = PROMPTS.FEATURE_IDEAS_ROUND1
        elif round_num == 2:
            prompt_template = PROMPTS.FEATURE_IDEAS_ROUND2
        else:
            prompt_template = PROMPTS.FEATURE_IDEAS_ROUND3

        prompt = prompt_template.format(
            data_summary=data_summary,
            column_info=column_info,
            statistics=statistics,
            previous_features=previous_features,
            previous_score=previous_score,
            best_score=previous_score,
            best_features=previous_features
        )

        try:
            response = self.llm.ask_json(
                system_prompt=PROMPTS.SYSTEM_GENERAL,
                user_prompt=prompt
            )
            ideas = [FeatureIdea(**item) for item in response]

            # Ровно 5 идей
            ideas = ideas[:5]
            while len(ideas) < 5:
                ideas.append(self._dummy_idea(len(ideas)))

            return ideas
        except (LLMError, Exception) as e:
            logger.error("FeatureIdeaGenerator", str(e), recoverable=True)
            return self._fallback_ideas()

    def _fallback_ideas(self) -> list[FeatureIdea]:
        """Аварийные идеи, если LLM не ответил."""
        return [
            FeatureIdea("null_count", "Количество пропусков в строке",
                        "row.isna().sum()", "Пропуски могут коррелировать с target"),
            FeatureIdea("numeric_mean", "Среднее числовых колонок",
                        "row[numeric_cols].mean()", "Общий уровень значений"),
            FeatureIdea("numeric_std", "Разброс числовых колонок",
                        "row[numeric_cols].std()", "Вариативность признаков"),
            FeatureIdea("max_min_ratio", "Отношение макс к мин",
                        "row[numeric_cols].max() / (row[numeric_cols].min() + 1)",
                        "Масштаб разброса"),
            FeatureIdea("top_cat_freq", "Частота самой популярной категории",
                        "value_counts encoding",
                        "Частотность категорий")
        ]
```

---

### 3.12 `src/features/code_generator.py` — FeatureCodeGenerator

**Ответственность:** превращение идей в исполняемый Python-код через LLM.

```python
# Псевдокод
class FeatureCodeGenerator:
    def __init__(self, llm: GigaChatClient):
        self.llm = llm

    def generate(self, ideas: list[FeatureIdea],
                 column_roles: ColumnRoles,
                 sample_data: str,
                 id_col: str,
                 target_col: str) -> str:
        """Сгенерировать код функции generate_features."""

        # Форматируем информацию о колонках
        column_types = "\n".join(
            f"  {col}: {role}" for role, cols in [
                ("numeric", column_roles.numeric),
                ("categorical", column_roles.categorical),
                ("datetime", column_roles.datetime)
            ] for col in cols
        )

        # Форматируем идеи
        ideas_str = "\n".join(
            f"{i+1}. {idea.name}: {idea.description}\n"
            f"   Формула: {idea.formula}\n"
            f"   Почему: {idea.reasoning}"
            for i, idea in enumerate(ideas)
        )

        prompt = PROMPTS.CODE_GENERATION_PROMPT.format(
            id_col=id_col,
            target_col=target_col,
            column_types=column_types,
            sample_data=sample_data,
            feature_ideas=ideas_str
        )

        code = self.llm.ask_code(
            system_prompt=PROMPTS.SYSTEM_GENERAL,
            user_prompt=prompt
        )

        # Валидация синтаксиса
        self._validate_syntax(code)

        # Проверка безопасности
        self._check_safety(code)

        return code

    def fix(self, original_code: str, error: str,
            attempt: int) -> str:
        """Попросить LLM исправить код."""
        prompt = PROMPTS.CODE_FIX_PROMPT.format(
            original_code=original_code,
            error_message=error[:1000],  # обрезаем длинный traceback
            attempt=attempt,
            max_attempts=Settings.MAX_RETRIES_PER_ROUND
        )
        return self.llm.ask_code(
            system_prompt=PROMPTS.SYSTEM_GENERAL,
            user_prompt=prompt
        )

    def _validate_syntax(self, code: str):
        """Проверить что код парсится."""
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            raise CodeValidationError(f"Syntax error: {e}")

    def _check_safety(self, code: str):
        """Проверить что код не содержит опасных конструкций."""
        forbidden = [
            'import os', 'import sys', 'import subprocess',
            'import shutil', 'open(', 'exec(', 'eval(',
            '__import__', 'os.system', 'os.popen'
        ]
        for pattern in forbidden:
            if pattern in code:
                raise CodeSafetyError(f"Forbidden pattern: {pattern}")
```

---

### 3.13 `src/features/code_executor.py` — CodeExecutor

**Ответственность:** безопасное выполнение сгенерированного кода, валидация результатов.

```python
# Псевдокод
@dataclass
class ExecutionResult:
    success: bool
    train_features: pd.DataFrame | None
    test_features: pd.DataFrame | None
    error: str | None
    execution_time: float

class CodeExecutor:
    def execute(self, code: str,
                train_df: pd.DataFrame,
                test_df: pd.DataFrame) -> ExecutionResult:
        """Выполнить код в изолированном namespace."""

        start = time.time()

        # Подготовить namespace
        namespace = {
            'pd': pd,
            'np': np,
            'train_df': train_df.copy(),
            'test_df': test_df.copy(),
        }

        try:
            # Компиляция
            compiled = compile(code, "<generated>", "exec")

            # Выполнение с таймаутом
            # (используем signal.alarm на Linux или threading)
            exec(compiled, namespace)

            # Вызвать функцию
            generate_fn = namespace.get('generate_features')
            if generate_fn is None:
                return ExecutionResult(
                    success=False, error="Function generate_features not found",
                    train_features=None, test_features=None,
                    execution_time=time.time() - start
                )

            train_feat, test_feat = generate_fn(
                train_df.copy(), test_df.copy()
            )

            # Валидация
            errors = self._validate(train_feat, test_feat,
                                     train_df, test_df)
            if errors:
                return ExecutionResult(
                    success=False, error="; ".join(errors),
                    train_features=train_feat, test_features=test_feat,
                    execution_time=time.time() - start
                )

            return ExecutionResult(
                success=True,
                train_features=train_feat,
                test_features=test_feat,
                error=None,
                execution_time=time.time() - start
            )

        except Exception as e:
            tb = traceback.format_exc()
            return ExecutionResult(
                success=False, error=f"{str(e)}\n\n{tb[-500:]}",
                train_features=None, test_features=None,
                execution_time=time.time() - start
            )

    def _validate(self, train_feat, test_feat,
                  orig_train, orig_test) -> list[str]:
        """Проверить выходные DataFrame'ы."""
        errors = []

        # 1. Это DataFrame?
        if not isinstance(train_feat, pd.DataFrame):
            errors.append("train_features is not DataFrame")
            return errors
        if not isinstance(test_feat, pd.DataFrame):
            errors.append("test_features is not DataFrame")
            return errors

        # 2. Количество строк совпадает
        if len(train_feat) != len(orig_train):
            errors.append(f"train row count mismatch: "
                         f"{len(train_feat)} vs {len(orig_train)}")
        if len(test_feat) != len(orig_test):
            errors.append(f"test row count mismatch: "
                         f"{len(test_feat)} vs {len(orig_test)}")

        # 3. Колонки фичей совпадают (кроме target)
        train_feat_cols = set(train_feat.columns) - {id_col, target_col}
        test_feat_cols = set(test_feat.columns) - {id_col}
        if train_feat_cols != test_feat_cols:
            errors.append(f"Feature columns mismatch: "
                         f"train={train_feat_cols}, test={test_feat_cols}")

        # 4. Не больше 5 фичей
        n_features = len(train_feat_cols)
        if n_features > 5:
            errors.append(f"Too many features: {n_features} > 5")

        # 5. Все фичи числовые
        for col in train_feat_cols:
            if not pd.api.types.is_numeric_dtype(train_feat[col]):
                errors.append(f"Feature {col} is not numeric: "
                             f"{train_feat[col].dtype}")

        # 6. Нет бесконечностей
        for col in train_feat_cols:
            if np.isinf(train_feat[col]).any():
                errors.append(f"Feature {col} has Inf values in train")
            if np.isinf(test_feat[col]).any():
                errors.append(f"Feature {col} has Inf values in test")

        return errors
```

---

### 3.14 `src/features/evaluator.py` — Evaluator

**Ответственность:** оценка качества набора фичей через CatBoost с кросс-валидацией. Точно повторяет логику скоринга организаторов.

```python
# Псевдокод
@dataclass
class EvalResult:
    mean_auc: float
    std_auc: float
    fold_scores: list[float]
    feature_importances: dict[str, float]
    feature_names: list[str]
    eval_time: float

class Evaluator:
    def evaluate(self, train_features: pd.DataFrame,
                 target_col: str, id_col: str) -> EvalResult:
        """Оценить набор фичей через CatBoost CV."""

        start = time.time()

        # Подготовка
        feature_cols = [c for c in train_features.columns
                        if c not in (target_col, id_col)]
        X = train_features[feature_cols].values
        y = train_features[target_col].values

        # CV
        cv = StratifiedKFold(
            n_splits=Settings.CV_FOLDS,
            shuffle=True,
            random_state=Settings.CV_RANDOM_SEED
        )

        fold_scores = []
        importances = np.zeros(len(feature_cols))

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = CatBoostClassifier(
                iterations=Settings.CATBOOST_ITERATIONS,
                verbose=Settings.CATBOOST_VERBOSE,
                random_seed=Settings.CV_RANDOM_SEED,
                # дефолтные гиперпараметры — как у организаторов
            )
            model.fit(X_train, y_train, eval_set=(X_val, y_val),
                      early_stopping_rounds=50)

            y_pred = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)
            fold_scores.append(auc)

            importances += model.feature_importances_

        importances /= Settings.CV_FOLDS

        return EvalResult(
            mean_auc=np.mean(fold_scores),
            std_auc=np.std(fold_scores),
            fold_scores=fold_scores,
            feature_importances=dict(zip(feature_cols, importances)),
            feature_names=feature_cols,
            eval_time=time.time() - start
        )
```

---

### 3.15 `src/features/selector.py` — FeatureSelector

**Ответственность:** сравнение результатов раундов, выбор лучшего набора.

```python
# Псевдокод
@dataclass
class RoundResult:
    round_num: int
    ideas: list[FeatureIdea]
    code: str
    train_features: pd.DataFrame
    test_features: pd.DataFrame
    eval_result: EvalResult
    total_time: float

class FeatureSelector:
    def select_best(self, rounds: list[RoundResult]) -> RoundResult:
        """Выбрать лучший раунд по ROC-AUC."""
        if not rounds:
            raise NoResultsError("No successful rounds")

        # Сортировка по mean_auc, при равенстве — по std_auc (меньше = стабильнее)
        rounds_sorted = sorted(
            rounds,
            key=lambda r: (r.eval_result.mean_auc, -r.eval_result.std_auc),
            reverse=True
        )

        best = rounds_sorted[0]
        logger.info(
            f"Best round: #{best.round_num} "
            f"AUC={best.eval_result.mean_auc:.4f} "
            f"(±{best.eval_result.std_auc:.4f})"
        )
        return best

    def get_feedback_for_next_round(self, result: RoundResult) -> str:
        """Сформировать обратную связь для следующего раунда."""
        imp = result.eval_result.feature_importances
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)

        lines = [f"ROC-AUC: {result.eval_result.mean_auc:.4f}"]
        lines.append("Feature importances:")
        for name, importance in sorted_imp:
            lines.append(f"  {name}: {importance:.1f}")
        lines.append("\nИдеи фичей:")
        for idea in result.ideas:
            lines.append(f"  {idea.name}: {idea.description}")

        return "\n".join(lines)
```

---

### 3.16 `src/features/fallback.py` — FallbackFeatureGenerator

**Ответственность:** аварийный генератор фичей, который работает БЕЗ LLM. Гарантирует, что output всегда будет заполнен.

```python
# Псевдокод
class FallbackFeatureGenerator:
    def generate(self, train_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 id_col: str,
                 target_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Сгенерировать 5 простых фичей без LLM.
        Работает ВСЕГДА, на любых данных.
        """

        numeric_cols = train_df.select_dtypes(include='number').columns
        numeric_cols = [c for c in numeric_cols
                        if c not in (id_col, target_col)]

        def compute(df):
            features = pd.DataFrame()
            features[id_col] = df[id_col]

            if numeric_cols:
                # 1. Количество пропусков
                features['feat_null_count'] = df[numeric_cols].isna().sum(axis=1)

                # 2. Среднее числовых
                features['feat_numeric_mean'] = df[numeric_cols].mean(axis=1)

                # 3. Стд числовых
                features['feat_numeric_std'] = df[numeric_cols].std(axis=1)

                # 4. Макс числовых
                features['feat_numeric_max'] = df[numeric_cols].max(axis=1)

                # 5. Мин числовых
                features['feat_numeric_min'] = df[numeric_cols].min(axis=1)
            else:
                # Если нет числовых — частотное кодирование категорий
                cat_cols = df.select_dtypes(include='object').columns
                cat_cols = [c for c in cat_cols if c != id_col][:5]
                for i, col in enumerate(cat_cols):
                    freq = df[col].value_counts(normalize=True)
                    features[f'feat_freq_{i}'] = df[col].map(freq).fillna(0)

            features.fillna(-999, inplace=True)
            return features

        train_feat = compute(train_df)
        train_feat[target_col] = train_df[target_col].values

        test_feat = compute(test_df)

        return train_feat, test_feat
```

---

### 3.17 `src/output/finalizer.py` — Finalizer

**Ответственность:** финальное сохранение результатов и валидация.

```python
# Псевдокод
class Finalizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def save(self, result: RoundResult,
             id_col: str, target_col: str):
        """Сохранить результаты в output/."""

        os.makedirs(self.output_dir, exist_ok=True)

        train_out = result.train_features.copy()
        test_out = result.test_features.copy()

        # Убедиться что есть ID и target в train
        assert id_col in train_out.columns
        assert target_col in train_out.columns
        assert id_col in test_out.columns
        assert target_col not in test_out.columns

        # Убедиться что фичи одинаковые
        feat_cols_train = sorted(
            [c for c in train_out.columns if c not in (id_col, target_col)]
        )
        feat_cols_test = sorted(
            [c for c in test_out.columns if c != id_col]
        )
        assert feat_cols_train == feat_cols_test, \
            f"Mismatch: {feat_cols_train} vs {feat_cols_test}"

        # Порядок колонок: id, target (train), feat1..feat5
        train_cols = [id_col, target_col] + feat_cols_train
        test_cols = [id_col] + feat_cols_test

        train_out = train_out[train_cols]
        test_out = test_out[test_cols]

        # Заполнить NaN
        for col in feat_cols_train:
            train_out[col] = train_out[col].fillna(-999)
            test_out[col] = test_out[col].fillna(-999)

        # Заменить inf
        train_out.replace([np.inf, -np.inf], -999, inplace=True)
        test_out.replace([np.inf, -np.inf], -999, inplace=True)

        # Сохранить
        train_path = os.path.join(self.output_dir, "train.csv")
        test_path = os.path.join(self.output_dir, "test.csv")

        train_out.to_csv(train_path, index=False)
        test_out.to_csv(test_path, index=False)

        logger.info(f"Saved train: {train_path} ({train_out.shape})")
        logger.info(f"Saved test: {test_path} ({test_out.shape})")

        # Финальная валидация
        self._validate_saved_files(train_path, test_path)

    def _validate_saved_files(self, train_path, test_path):
        """Перечитать и проверить файлы."""
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)

        assert len(train) > 0, "Empty train output"
        assert len(test) > 0, "Empty test output"
        assert not train.isna().any().any(), "NaN in train output"
        assert not test.isna().any().any(), "NaN in test output"
```

---

### 3.18 `src/orchestrator.py` — Orchestrator (главный контроллер)

**Ответственность:** координация всех компонентов. Это "мозг" системы, но детерминированный.

```python
# Псевдокод
class Orchestrator:
    def __init__(self, credentials, model, data_dir, output_dir, time_budget):
        self.timer = TimerManager(time_budget)
        self.llm = GigaChatClient(credentials, model)
        self.explorer = DataExplorer(data_dir)
        self.join_planner = JoinPlanner(self.llm)
        self.preparer = DataPreparer()
        self.idea_gen = FeatureIdeaGenerator(self.llm)
        self.code_gen = FeatureCodeGenerator(self.llm)
        self.executor = CodeExecutor()
        self.evaluator = Evaluator()
        self.selector = FeatureSelector()
        self.fallback = FallbackFeatureGenerator()
        self.finalizer = Finalizer(output_dir)

        self.state = OrchestratorState()

    def run(self):
        """Главный метод — полный пайплайн."""

        # ══════════════════════════════════════
        #  PHASE 1: РАЗВЕДКА И ПОДГОТОВКА ДАННЫХ
        # ══════════════════════════════════════

        logger.step("PHASE 1", "Data exploration")
        exploration = self.explorer.scan()
        self.timer.checkpoint("exploration_done")

        logger.step("PHASE 1", "Planning joins")
        join_plan = self.join_planner.plan(exploration)
        self.timer.checkpoint("join_plan_done")

        logger.step("PHASE 1", "Executing joins")
        train_df, test_df = self.preparer.prepare(exploration, join_plan)
        column_roles = self.preparer.get_column_roles(
            train_df, exploration.target_col, exploration.id_col
        )
        self.timer.checkpoint("data_prepared")

        # Сформировать summary для промптов
        data_summary = self._build_summary(exploration, train_df, column_roles)
        column_info = self._build_column_info(column_roles, train_df)
        statistics = self._build_statistics(train_df, column_roles)
        sample_data = train_df.head(Settings.MAX_SAMPLE_ROWS).to_string()

        # ══════════════════════════════════════
        # PHASE 1.5: НЕМЕДЛЕННЫЙ FALLBACK
        # ══════════════════════════════════════
        # Генерируем fallback-фичи СРАЗУ, до любых LLM-раундов.
        # Если всё пойдёт не так — у нас всегда есть результат.
        logger.step("PHASE 1.5", "Generating fallback features")
        fallback_train, fallback_test = self.fallback.generate(
            train_df, test_df, exploration.id_col, exploration.target_col
        )
        fallback_eval = self.evaluator.evaluate(
            fallback_train, exploration.target_col, exploration.id_col
        )
        self.state.fallback_result = RoundResult(
            round_num=0,
            ideas=[],
            code="fallback",
            train_features=fallback_train,
            test_features=fallback_test,
            eval_result=fallback_eval,
            total_time=self.timer.elapsed()
        )
        self.state.best_round = self.state.fallback_result
        logger.metric("fallback_auc", fallback_eval.mean_auc)
        self.timer.checkpoint("fallback_done")

        # ══════════════════════════════════════
        #  PHASE 2: ИТЕРАТИВНАЯ ГЕНЕРАЦИЯ ФИЧЕЙ
        # ══════════════════════════════════════

        round_num = 0
        while round_num < Settings.MAX_ROUNDS and self.timer.can_start_round():
            round_num += 1
            logger.step("PHASE 2", f"Round {round_num}")

            round_result = self._run_round(
                round_num=round_num,
                data_summary=data_summary,
                column_info=column_info,
                statistics=statistics,
                sample_data=sample_data,
                train_df=train_df,
                test_df=test_df,
                column_roles=column_roles,
                exploration=exploration
            )

            if round_result is not None:
                self.state.rounds.append(round_result)
                logger.metric(f"round_{round_num}_auc",
                             round_result.eval_result.mean_auc)

                # Обновить лучший результат
                if (round_result.eval_result.mean_auc >
                        self.state.best_round.eval_result.mean_auc):
                    self.state.best_round = round_result
                    logger.step("PHASE 2",
                               f"New best: AUC={round_result.eval_result.mean_auc:.4f}")

            self.timer.checkpoint(f"round_{round_num}_done")

            # Проверка времени
            if self.timer.must_stop_now():
                logger.warning("Time critical — breaking loop")
                break

        # ══════════════════════════════════════
        #  PHASE 3: ФИНАЛИЗАЦИЯ
        # ══════════════════════════════════════

        logger.step("PHASE 3", "Finalizing")
        self.finalizer.save(
            self.state.best_round,
            exploration.id_col,
            exploration.target_col
        )
        self.timer.checkpoint("finalized")

        # Отчёт
        logger.step("DONE", self.timer.report())
        logger.step("DONE", f"Best AUC: {self.state.best_round.eval_result.mean_auc:.4f}")
        logger.step("DONE", f"LLM calls: {self.llm.stats()}")

    def _run_round(self, round_num, data_summary, column_info,
                   statistics, sample_data, train_df, test_df,
                   column_roles, exploration) -> RoundResult | None:
        """Один раунд: идеи → код → выполнение → оценка."""

        round_start = time.time()

        # Обратная связь от предыдущего раунда
        previous_features = ""
        previous_score = 0.0
        if self.state.rounds:
            last = self.state.rounds[-1]
            previous_features = self.selector.get_feedback_for_next_round(last)
            previous_score = last.eval_result.mean_auc

        # 1. Генерация идей
        ideas = self.idea_gen.generate(
            data_summary=data_summary,
            column_info=column_info,
            statistics=statistics,
            round_num=round_num,
            previous_features=previous_features,
            previous_score=previous_score
        )

        if self.timer.must_stop_now():
            return None

        # 2. Генерация кода + ретраи
        code = None
        exec_result = None

        for attempt in range(1, Settings.MAX_RETRIES_PER_ROUND + 1):
            if self.timer.must_stop_now():
                return None

            try:
                if code is None or not exec_result or not exec_result.success:
                    if attempt == 1:
                        code = self.code_gen.generate(
                            ideas=ideas,
                            column_roles=column_roles,
                            sample_data=sample_data,
                            id_col=exploration.id_col,
                            target_col=exploration.target_col
                        )
                    else:
                        code = self.code_gen.fix(
                            original_code=code,
                            error=exec_result.error,
                            attempt=attempt
                        )

                exec_result = self.executor.execute(code, train_df, test_df)

                if exec_result.success:
                    break

            except (LLMError, CodeValidationError, CodeSafetyError) as e:
                logger.error("Round", f"Attempt {attempt}: {e}", recoverable=True)
                if attempt == Settings.MAX_RETRIES_PER_ROUND:
                    return None

        if exec_result is None or not exec_result.success:
            return None

        # 3. Оценка
        eval_result = self.evaluator.evaluate(
            exec_result.train_features,
            exploration.target_col,
            exploration.id_col
        )

        return RoundResult(
            round_num=round_num,
            ideas=ideas,
            code=code,
            train_features=exec_result.train_features,
            test_features=exec_result.test_features,
            eval_result=eval_result,
            total_time=time.time() - round_start
        )

    def emergency_save(self):
        """Аварийное сохранение при полном крахе."""
        try:
            if self.state.best_round:
                self.finalizer.save(
                    self.state.best_round,
                    self.explorer.last_id_col or "id",
                    self.explorer.last_target_col or "target"
                )
            else:
                # Совсем крайний случай — тупо скопировать ID
                exploration = self.explorer.scan()
                train_df = exploration.tables["train.csv"]
                test_df = exploration.tables["test.csv"]
                fb_train, fb_test = self.fallback.generate(
                    train_df, test_df,
                    exploration.id_col,
                    exploration.target_col
                )
                self.finalizer.save(
                    RoundResult(0, [], "", fb_train, fb_test,
                               EvalResult(0,0,[],{},[],0), 0),
                    exploration.id_col,
                    exploration.target_col
                )
        except Exception as e:
            logger.critical(f"Emergency save failed: {e}")
```

---

## 4. Диаграмма потока данных (полная)

```
┌─────────────────────────────────────────────────────────────────┐
│                          run.py                                 │
│                    load .env → Orchestrator.run()                │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 1: РАЗВЕДКА                           │
│                                                                 │
│  data/readme.txt ──→ DataExplorer.read_readme()                 │
│  data/*.csv      ──→ DataExplorer.profile_table()               │
│                      │                                          │
│                      ▼                                          │
│               ExplorationResult                                 │
│               ├── readme_text                                   │
│               ├── tables: {name: DataFrame}                     │
│               ├── profiles: {name: TableProfile}                │
│               ├── target_col                                    │
│               └── id_col                                        │
│                      │                                          │
│                      ▼                                          │
│               JoinPlanner ──LLM──→ JoinPlan                     │
│                      │          (fallback если LLM сбой)        │
│                      ▼                                          │
│               DataPreparer ──→ (train_df, test_df)              │
│                               ColumnRoles                       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 1.5: FALLBACK                           │
│                                                                 │
│  FallbackFeatureGenerator.generate(train_df, test_df)           │
│  Evaluator.evaluate(fallback_features)                          │
│  → state.best_round = fallback (гарантия результата)            │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PHASE 2: ГЕНЕРАЦИЯ (до 3 раундов)              │
│                                                                 │
│  ┌────────────── ROUND LOOP ─────────────────────────┐          │
│  │                                                    │          │
│  │  1. FeatureIdeaGenerator ──LLM──→ 5 ideas         │          │
│  │     (промпт зависит от номера раунда)              │          │
│  │                │                                   │          │
│  │                ▼                                   │          │
│  │  2. FeatureCodeGenerator ──LLM──→ Python code      │          │
│  │                │                                   │          │
│  │                ▼                                   │          │
│  │  3. CodeExecutor.execute(code, train, test)        │          │
│  │                │                                   │          │
│  │         ┌──────┴──────┐                            │          │
│  │         │ SUCCESS?    │                            │          │
│  │         │             │                            │          │
│  │    YES  ▼        NO   ▼                            │          │
│  │   Evaluator    CodeGenerator.fix() ──LLM           │          │
│  │     │              │                               │          │
│  │     │         CodeExecutor.execute() (retry)       │          │
│  │     │              │                               │          │
│  │     │         (до 3 попыток)                       │          │
│  │     │                                              │          │
│  │     ▼                                              │          │
│  │  RoundResult                                       │          │
│  │  ├── mean_auc                                      │          │
│  │  ├── feature_importances                           │          │
│  │  ├── train_features                                │          │
│  │  └── test_features                                 │          │
│  │     │                                              │          │
│  │     ▼                                              │          │
│  │  Selector.compare(round, best)                     │          │
│  │  → обновить state.best_round если лучше            │          │
│  │  → сформировать feedback для след. раунда          │          │
│  │                                                    │          │
│  │  Timer.can_start_round()? ─── NO ──→ BREAK         │          │
│  │         │                                          │          │
│  │    YES  ▼                                          │          │
│  │  (следующий раунд)                                 │          │
│  └────────────────────────────────────────────────────┘          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 3: ФИНАЛИЗАЦИЯ                          │
│                                                                 │
│  Finalizer.save(state.best_round)                               │
│  ├── output/train.csv  (id, target, feat1..feat5)               │
│  ├── output/test.csv   (id, feat1..feat5)                       │
│  └── Validator.validate()                                       │
│                                                                 │
│  Logger.summary()                                               │
│  Timer.report()                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Бюджет LLM-вызовов и времени

### 5.1 Вызовы LLM по сценариям

| Сценарий | Компонент | Вызовов | ~Время |
|---|---|---|---|
| Планирование join | JoinPlanner | 1 | 5-8 сек |
| Идеи раунд 1 | IdeaGenerator | 1 | 5-8 сек |
| Код раунд 1 | CodeGenerator | 1 | 8-15 сек |
| Фикс кода (0-3) | CodeGenerator.fix | 0-3 | 0-45 сек |
| Идеи раунд 2 | IdeaGenerator | 1 | 5-8 сек |
| Код раунд 2 | CodeGenerator | 1 | 8-15 сек |
| Фикс кода (0-3) | CodeGenerator.fix | 0-3 | 0-45 сек |
| Идеи раунд 3 | IdeaGenerator | 0-1 | 0-8 сек |
| Код раунд 3 | CodeGenerator | 0-1 | 0-15 сек |
| **ИТОГО** | | **5-15** | **~50-170 сек** |

### 5.2 Бюджет времени по фазам

| Фаза | Мин. время | Макс. время | Описание |
|---|---|---|---|
| Phase 1 | 20 сек | 60 сек | Разведка, join, подготовка |
| Phase 1.5 | 10 сек | 30 сек | Fallback + его оценка |
| Phase 2 Round 1 | 60 сек | 150 сек | Идеи + код + ретраи + eval |
| Phase 2 Round 2 | 60 сек | 150 сек | То же |
| Phase 2 Round 3 | 0 сек | 150 сек | Опционально |
| Phase 3 | 5 сек | 15 сек | Сохранение |
| **Запас** | **15 сек** | **20 сек** | Safety margin |
| **ИТОГО** | **170 сек** | **575 сек** | **≤ 580** |

---

## 6. Data-модели (все dataclass'ы)

```python
@dataclass
class ColumnProfile:
    name: str
    dtype: str
    semantic_type: str        # numeric, categorical, datetime, id, binary, text
    nunique: int
    null_count: int
    null_pct: float
    sample_values: list
    min: float | None
    max: float | None
    mean: float | None
    median: float | None
    std: float | None
    q25: float | None
    q75: float | None
    skew: float | None
    top_values: list[tuple] | None
    is_binary: bool
    date_min: str | None
    date_max: str | None
    date_range_days: int | None

@dataclass
class TableProfile:
    filename: str
    shape: tuple[int, int]
    columns: list[ColumnProfile]
    memory_mb: float
    head_str: str
    possible_keys: list[str]
    possible_dates: list[str]

@dataclass
class ExplorationResult:
    readme_text: str
    tables: dict[str, pd.DataFrame]
    profiles: dict[str, TableProfile]
    target_col: str
    id_col: str
    separators: dict[str, str]

@dataclass
class JoinStep:
    right_table: str
    join_key_left: str
    join_key_right: str
    join_type: str
    needs_aggregation: bool
    aggregations: dict | None

@dataclass
class JoinPlan:
    steps: list[JoinStep]
    reasoning: str

@dataclass
class ColumnRoles:
    numeric: list[str]
    categorical: list[str]
    datetime: list[str]
    target: str
    id: str

@dataclass
class FeatureIdea:
    name: str
    description: str
    formula: str
    reasoning: str

@dataclass
class ExecutionResult:
    success: bool
    train_features: pd.DataFrame | None
    test_features: pd.DataFrame | None
    error: str | None
    execution_time: float

@dataclass
class EvalResult:
    mean_auc: float
    std_auc: float
    fold_scores: list[float]
    feature_importances: dict[str, float]
    feature_names: list[str]
    eval_time: float

@dataclass
class RoundResult:
    round_num: int
    ideas: list[FeatureIdea]
    code: str
    train_features: pd.DataFrame
    test_features: pd.DataFrame
    eval_result: EvalResult
    total_time: float

@dataclass
class OrchestratorState:
    fallback_result: RoundResult | None = None
    rounds: list[RoundResult] = field(default_factory=list)
    best_round: RoundResult | None = None
```

---

## 7. Обработка ошибок — полная карта

| Компонент | Ошибка | Реакция |
|---|---|---|
| DataExplorer | Файл не найден | Пропустить, логировать |
| DataExplorer | CSV не парсится | Попробовать другой разделитель |
| DataExplorer | Не нашёл target/id | Эвристика по readme + колонкам |
| JoinPlanner | LLM не ответил | Fallback: join по общим колонкам |
| JoinPlanner | LLM вернул невалидный JSON | Retry 1 раз, потом fallback |
| JoinPlanner | Указал несуществующие колонки | Fallback |
| DataPreparer | Join привёл к пустому DataFrame | Вернуть train/test без join |
| DataPreparer | Memory error (слишком большие данные) | Sample данных перед join |
| IdeaGenerator | LLM не ответил | Fallback-идеи (простые статистики) |
| IdeaGenerator | Вернул < 5 идей | Дополнить fallback-идеями |
| CodeGenerator | LLM не ответил | Fallback-код |
| CodeGenerator | Невалидный синтаксис | Retry fix, потом fallback |
| CodeGenerator | Опасный код (import os) | Отклонить, retry |
| CodeExecutor | Runtime error | Передать traceback в fix, retry |
| CodeExecutor | Timeout (> 60 сек) | Kill, retry с упрощением |
| CodeExecutor | Вывод не прошёл валидацию | Передать ошибки в fix, retry |
| Evaluator | CatBoost ошибка | Упростить (меньше iterations) |
| Evaluator | Все фичи константные | Вернуть AUC=0.5 |
| Finalizer | Не удалось сохранить | Retry, потом emergency_save |
| TimerManager | Время на исходе | Немедленно сохранить best |
| Orchestrator | Любое необработанное | emergency_save (fallback) |

---

## 8. Зависимости (`pyproject.toml`)

```toml
[project]
name = "features-agent"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    "langchain-gigachat>=0.4.0",
    "gigachat>=0.1.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "catboost>=1.2",
    "scikit-learn>=1.3.0",
    "scipy>=1.11.0",
    "python-dotenv>=1.0.0",
]
```
