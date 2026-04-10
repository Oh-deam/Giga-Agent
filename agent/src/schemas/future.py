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

    col1: str = Field(description="Название существующей (или ранее созданной) колонки")
    col2: str | float | int = Field(description="Имя колонки или число для действия с первой колонкой")
    new_col_name: str = Field(description="Имя новой колонки")
    action: ACTIONS = Field(description="Действие, которое нужно совершить")
    save_col: bool = Field(description="Понадобится ли сохранить колонку в финальной версии датасета", default=False)


class Proposal(BaseModel):
    """Предложения по улучшению датасета"""

    proposal: list[FutureProposal] = Field(description="Список предложений по созданию новых колонок в датасете")

