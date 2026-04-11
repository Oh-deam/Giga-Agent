import enum

from pydantic import AliasChoices, BaseModel, Field


class ACTIONS(enum.Enum):
    Multiplication = "Multiplication"
    Subtraction = "Subtraction"
    Addition = "Addition"
    Division = "Division"
    Degree = "Degree"
    Logarithm = "Logarithm"
    Threshold = "Threshold"
    BinaryCategory = "BinaryCategory"
    StringConcat = "StringConcat"
    IsMissing = "IsMissing"


class FutureProposal(BaseModel):
    """Предложенные фьючерсы"""

    fields: list[str | int | float] = Field(description="Список полей, которые нужно создать, могут быть названия строк или числа")
    new_col_name: str = Field(description="Имя новой колонки")
    actions: list[ACTIONS] = Field(description="Действия, которое нужно совершить последовательно для fields. Если fields не пустой, то actions тоже обязан быть заполнен")
    reason: str = Field(
        default="",
        validation_alias=AliasChoices("reason", "rationale"),
        description="Причина, по которой нужно создать именно эту колонку",
    )
    save_col: bool = Field(default=True, description="Нужно ли сохранять колонку в итоговом output")


class Proposal(BaseModel):
    """Предложения по улучшению датасета"""

    proposal: list[FutureProposal] = Field(description="Список предложений по созданию новых колонок в датасете")
