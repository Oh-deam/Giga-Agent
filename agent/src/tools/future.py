import loguru
import numpy as np
import pandas as pd

from src.schemas.future import Proposal, ACTIONS

def _validate_value(col: str):
    try:
        return float(col)
    except ValueError:
        return col

def create_new_futures(df, proposals: Proposal):
    new_df = pd.DataFrame()
    loguru.logger.debug(f"Columns in base dataframe: {df.columns}")
    print(f"Base df:\n {df.head()}")
    for proposal in proposals.proposal:

        start_col = proposal.fields[0]
        if start_col in df.columns:
            tmp_series = df[start_col].copy()
        elif start_col in new_df.columns:
            tmp_series = new_df[start_col]
        else:
            loguru.logger.warning(f"Col {start_col} was not found; Cols base df {df.columns}; Cols new df {new_df.columns}")
            continue

        # Check for logarithm
        if len(proposal.actions) == 1 and proposal.actions[0] == ACTIONS.Logarithm:
            new_df[proposal.new_col_name] = np.log(tmp_series)
            continue

        if len(proposal.fields) - 1 != len(proposal.actions):
            if proposal.actions != ACTIONS.Logarithm:
                loguru.logger.warning(f" Для {proposal.fields} Количество полей и действий {proposal.actions} не совпадает")
                continue

        for col, action in zip(proposal.fields[1:], proposal.actions):
            # Col - имя столбца или число
            col = _validate_value(col)
            # Если это имя колонки - проверяем в каком датафрейме она лежит
            if isinstance(col, str):
                if col in df.columns:
                    col = df[col]
                elif col in new_df.columns:
                    col = new_df[col]
                else:
                    loguru.logger.warning(f"Col {col} was not found; Cols base df {df.columns}; Cols new df {new_df.columns}")

            match action:
                case ACTIONS.Addition:
                    tmp_series = tmp_series + col
                case ACTIONS.Subtraction:
                    tmp_series = tmp_series - col
                case ACTIONS.Multiplication:
                    tmp_series = tmp_series * col
                case ACTIONS.Division:
                    tmp_series = tmp_series / col
                case ACTIONS.Degree:
                    tmp_series = tmp_series ** col
                case ACTIONS.Logarithm:
                    tmp_series = np.log(tmp_series)

        new_df[proposal.new_col_name] = tmp_series
    new_df["target"] = df["target"]
    return new_df