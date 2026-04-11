import loguru
import numpy as np
import pandas as pd
from langchain_gigachat import GigaChat
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from src.schemas.future import Proposal
from src.tools.prompt import PromptFactory
from src.tools.stat import create_stat
from src.tools.future import create_new_futures
from src.utils.storage import Storage


def _test_dataframe(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        final_fit: bool = False,
) -> tuple[float, pd.Series]:
    TOP_K = 5
    boost = CatBoostClassifier(verbose=False)
    X_train = df_train.drop(columns=["target"])
    X_test = df_test.drop(columns=["target"])
    y_train = df_train["target"]
    y_test = df_test["target"]

    boost.fit(X_train, y_train)

    pred = boost.predict_proba(X_test)[:, 1]

    importances = pd.Series(boost.get_feature_importance(), index=X_train.columns).sort_values(ascending=False)
    top5_features = importances.head(TOP_K).index.tolist()
    if not final_fit:
        loguru.logger.debug(f"TOP {TOP_K}: {top5_features}")

    return roc_auc_score(y_test, pred), top5_features


def future_competition(
        model: GigaChat,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        storage: Storage,
        epochs: int = 2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stat = create_stat(df_train, "target")
    description = storage.description
    structured_future_llm = model.with_structured_output(Proposal)
    already_used_futures = Proposal(proposal=[])

    proposals = []
    roc_aucs = []
    for epoch in range(epochs):
    ### First Try
        prompt =  PromptFactory.create_prompt_for_future_engineering(
            stat=stat,
            description=description,
            early_futures=already_used_futures.proposal if len(already_used_futures.proposal) > 0 else None,
        )
        result = structured_future_llm.invoke(prompt)
        loguru.logger.debug(f"Epoch {epoch}: Features {result}")

        # add proposals to common list and add to list
        already_used_futures.proposal.extend(result.proposal)
        print(epoch, already_used_futures)

        df_tr = create_new_futures(df_train, result)
        df_tt = create_new_futures(df_test, result)

        loguru.logger.debug(f"Epoch {epoch}: Train validate model")
        roc_auc, top5_features = _test_dataframe(df_tr, df_tt)
        top5_features.append("target")
        loguru.logger.debug(f"Epoch {epoch}: roc_auc for {len(df_train.columns)}: {roc_auc}")

        df_tr = df_tr[top5_features]
        df_tt = df_tt[top5_features]

        loguru.logger.debug(f"Epoch {epoch}: Train model on 5 features")
        roc_auc, top5_features = _test_dataframe(df_tr, df_tt)
        loguru.logger.debug(f"Epoch {epoch}: roc_auc for {len(df_train.columns)}: {roc_auc}")

        add_proposals = Proposal(proposal=[])
        for proposal in result.proposal:
            if proposal.new_col_name in df_tr.columns:
                add_proposals.proposal.append(proposal)
        proposals.append(add_proposals)
        roc_aucs.append(roc_auc)


    max_roc_idx = np.argmax(roc_aucs)
    loguru.logger.info(f"Best roc auc: {roc_aucs[max_roc_idx]}")
    loguru.logger.debug(f"Best proposals: {proposals[max_roc_idx]}")
    final_proposal = proposals[max_roc_idx]
    df_train_final = create_new_futures(df_train, final_proposal)
    df_test_final = create_new_futures(df_test, final_proposal)
    return df_train_final, df_test_final