import enum

from pydantic import BaseModel, Field


class ACTIONS(enum.Enum):
    Multiplication = "Multiplication"
    Subtraction = "Subtraction"
    Addition = "Addition"
    Division = "Division"
    Degree = "Degree"
    Logarithm = "Logarithm"


class FutureProposal(BaseModel):
    """Предложенные фьючерсы"""

    fields: list[str | int | float] = Field(description="Список полей, которые нужно создать, могут быть названия строк или числа")
    new_col_name: str = Field(description="Имя новой колонки")
    actions: list[ACTIONS] = Field(description="Действия, которое нужно совершить последовательно для fields. Если fields не пустой, то actions тоже обязан быть заполнен")
    reason: str = Field(description="Причина, по которой нужно создать именно эту колонку")


class Proposal(BaseModel):
    """Предложения по улучшению датасета"""

    proposal: list[FutureProposal] = Field(description="Список предложений по созданию новых колонок в датасете")

