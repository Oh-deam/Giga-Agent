from pydantic import BaseModel, Field

class Aggregation(BaseModel):
    """Агрегация для столбца"""
    col_name: str = Field(description="Название столбца")
    method: str = Field(description="Метод агрегации")

class Aggregations(BaseModel):
    """Список агрегаций для столбцов"""
    group_by: list[str] = Field(description="Группировка по столбцам")
    aggregations: list[Aggregation] = Field(description="Список агрегаций")


class JoinCondition(BaseModel):
    """Одно условие объединения текущей таблицы со следующей таблицей."""

    table_name: str = Field(description="Имя очередной таблицы для объединения")# Имя очередной таблицы для объединения
    on_col1: str = Field(description="Столбец для объединения в текущей таблице")# по какому столбцу в первой таблице
    on_col2: str = Field(description="Столбец для объединения в очередной таблице")# по какому столбцу во второй таблице
    aggregations: Aggregations = Field(description="Агрегации для столбцов") # Список агрегаций для столбцов

class JoinConditions(BaseModel):
    """Набор условий объединения таблиц в правильном порядке."""
    conditions: list[JoinCondition] = Field(description="Список таблиц с условиями объединения") # Список условий объединения

""" JoinConditions(
    conditions=[
        JoinCondition(table_name="table1", on_col1="id", on_col2="id"),
        JoinCondition(table_name="table2", on_col1="id", on_col2="id")
    ]
)"""

"""
for condition in JoinCondition.conditions:
    condition.table_name
    condition.on_col1
    condition.on_col2
"""