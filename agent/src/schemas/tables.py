from pydantic import BaseModel

class JoinCondition(BaseModel):
    table_name: str # Имя очередной таблицы для объединения
    on_col1: str # по какому столбцу в первой таблице
    on_col2: str # по какому столбцу во второй таблице


class JoinConditions(BaseModel):
    conditions: list[JoinCondition] # Список условий объединения

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