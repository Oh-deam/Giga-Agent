from typing import Literal

from src.schemas.future import Proposal
from src.schemas.state import FeatureState, GigaChatDecision, Decision, Attempt
from src.tools.competition import _test_dataframe
from src.tools.future import create_new_futures
from src.tools.prompt import PromptFactory

from src.tools.stat import create_stat


class Nodes:
    def __init__(self, llm, df_train, df_test, description: str):
        self.llm = llm
        self.df_train = df_train
        self.df_test = df_test
        self.description = description
        self.stat = create_stat(df_train, "target")


    def generate_features(self, state: FeatureState) -> FeatureState:
        """
        Первая генерация новых фичей.
        """
        prompt = PromptFactory.create_prompt_for_future_engineering(
            stat=self.stat,
            description=self.description,
        )

        structured_llm: Proposal = self.llm.with_structured_output(
            Proposal
        )

        result = structured_llm.invoke(prompt)

        df_tr = create_new_futures(self.df_train, result)
        df_tt = create_new_futures(self.df_test, result)

        roc_auc, top5_features = _test_dataframe(df_tr, df_tt)

        df_tr = df_tr[top5_features]
        df_tr["target"] = self.df_train["target"]
        df_tt = df_tt[top5_features]
        df_tt["target"] = self.df_test["target"]

        roc_auc, top5_features = _test_dataframe(df_tr, df_tt)


        # Здесь у вас должна быть ваша логика:
        # 1. посчитать новые фичи
        # 2. обучить модель
        # 3. получить roc_auc
        # 4. получить feature importance
        roc_auc = 0.71
        improvement = top5_features.model_dump_json(indent=2, ensure_ascii=False)

        new_attempt = Attempt(
            features=result,
            roc_auc=roc_auc,
            improvement=improvement,
        )

        return state.model_copy(update={
            "attempts": [*state.attempts, new_attempt],
            "attempt": state.attempt + 1,
        })

    def improve_features(self, state: FeatureState) -> FeatureState:
        """
        Улучшение последнего Proposal на основе результатов предыдущих попыток.
        """
        last_attempt = state.attempts[-1]

        prompt = f"""
            Ты улучшаешь набор предложенных фичей.
            
            Текущая попытка: {state.attempt} из {state.max_attempt}
            
            Последний Proposal:
            {last_attempt.features.model_dump_json(indent=2, ensure_ascii=False)}
            
            ROC AUC:
            {last_attempt.roc_auc}
            
            Важность фичей:
            {last_attempt.improvement}
            
            Задача:
            - улучшить или переформулировать фичи
            - предложить более сильные варианты
            - учитывать, какие признаки дали вклад
            - вернуть обновленный Proposal
            """

        improved_proposal: Proposal = self.llm.invoke_structured(
            prompt=prompt,
            response_model=Proposal,
        )

        # Здесь снова ваша бизнес-логика пересчёта
        roc_auc = 0.73
        improvement = {"feature_c": 0.37, "feature_d": 0.21}

        new_attempt = Attempt(
            features=improved_proposal,
            roc_auc=roc_auc,
            improvement=improvement,
        )

        return state.model_copy(update={
            "attempts": [*state.attempts, new_attempt],
            "attempt": state.attempt + 1,
        })


    def evaluate_features(self, state: FeatureState) -> FeatureState:
        """
        Оценка: retry / improve / finish.
        """
        last_attempt = state.attempts[-1]

        if state.attempt >= state.max_attempt:
            return state.model_copy(update={"decision": Decision.FINISH})

        prompt = f"""
                Ты оцениваешь результаты генерации фичей.
                
                Всего уже сделано попыток: {state.attempt}
                Максимум попыток: {state.max_attempt}
                
                Последняя попытка:
                - ROC AUC: {last_attempt.roc_auc}
                - Важность фичей: {last_attempt.improvement}
                - Proposal:
                {last_attempt.features.model_dump_json(indent=2, ensure_ascii=False)}
                
                История ROC AUC по всем попыткам:
                {[a.roc_auc for a in state.attempts]}
                
                Выбери одно решение:
                - RETRY: попробовать заново сгенерировать новый набор
                - IMPROVE: улучшить текущий Proposal
                - FINISH: завершить и сохранить лучший результат
                
                Верни строго GigaChatDecision.
                """

        structured_llm: GigaChatDecision = self.llm.with_structured_output(
            GigaChatDecision
        )
        result = structured_llm.invoke(prompt)
        return state.model_copy(update={"decision": result.decision})

    def save_features(self, state: FeatureState) -> FeatureState:
        """
        Сохранение лучшей попытки.
        Находим попытку с максимальным ROC AUC, применяем её фичи к train и test,
        сохраняем итоговые датафреймы в data/.
        """
        best_attempt = max(state.attempts, key=lambda x: x.roc_auc)

        df_train_final = create_new_futures(self.df_train, best_attempt.features)
        df_test_final = create_new_futures(self.df_test, best_attempt.features)

        df_train_final.to_csv("data/train_featured.csv", index=False)
        df_test_final.to_csv("data/test_featured.csv", index=False)

        return state
