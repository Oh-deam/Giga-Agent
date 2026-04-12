import enum

import pandas as pd
from pydantic import BaseModel, Field

from src.schemas.future import Proposal

class Attempt(BaseModel):
    """Состояние попытки"""
    features: Proposal = Field(description="Предложения по созданию новых фичей")
    roc_auc: float = Field(description="Метрика ROC AUC")
    improvement: dict[str, float] = Field(
        default_factory=dict,
        description="Важность фичей для модели CatBoost"
    )


class Decision(enum.Enum):
    RETRY="RETRY"
    IMPROVE="IMPROVE"
    FINISH="FINISH"


class FeatureState(BaseModel):
    """Состояние попытки"""
    attempts: list[Attempt] = Field(default_factory=list, description="Список попыток")
    attempt: int = Field(default=0, description="Текущая попытка")
    max_attempt: int = Field(default=10, description=" Максимум попыток")
    decision: Decision = Field(default=None, description="Последнее принятое решение")


class GigaChatDecision(BaseModel):
    """Решение на основе оценки"""
    decision: Decision = Field(description="Решение")
    reason: str = Field(description="Почему принято решение")