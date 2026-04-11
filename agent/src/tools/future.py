import loguru
import numpy as np
import pandas as pd

from src.schemas.future import Proposal, ACTIONS


def _validate_value(col: str):
    try:
        return float(col)
    except ValueError:
        return col


def _resolve_operand(value, df: pd.DataFrame, new_df: pd.DataFrame):
    value = _validate_value(value)
    if isinstance(value, str):
        if value in df.columns:
            return df[value], True
        if value in new_df.columns:
            return new_df[value], True
        return value, False
    return value, True


def _coerce_numeric(value):
    if isinstance(value, pd.Series):
        return pd.to_numeric(value, errors="coerce")
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]


def create_new_futures(df, proposals: Proposal):
    new_df = pd.DataFrame(index=df.index)
    saved_columns: list[str] = []
    seen_output_names: set[str] = set()
    loguru.logger.debug(f"Columns in base dataframe: {df.columns}")
    for proposal in proposals.proposal:
        if proposal.new_col_name in seen_output_names:
            loguru.logger.warning(f"Skip proposal {proposal.new_col_name}: duplicate new_col_name")
            continue
        seen_output_names.add(proposal.new_col_name)

        if not proposal.fields:
            loguru.logger.warning(f"Skip proposal {proposal.new_col_name}: empty fields")
            continue
        if not proposal.actions:
            loguru.logger.warning(f"Skip proposal {proposal.new_col_name}: empty actions")
            continue

        start_col = proposal.fields[0]
        if not isinstance(start_col, str):
            loguru.logger.warning(f"Skip proposal {proposal.new_col_name}: first field must be column name, got {start_col}")
            continue

        start_value, is_valid = _resolve_operand(start_col, df, new_df)
        if not is_valid:
            loguru.logger.warning(f"Col {start_col} was not found; Cols base df {df.columns}; Cols new df {new_df.columns}")
            continue
        tmp_series = start_value.copy()

        unary_actions = {ACTIONS.Logarithm, ACTIONS.IsMissing}
        required_operands = sum(action not in unary_actions for action in proposal.actions)
        if proposal.actions == [ACTIONS.Degree] and len(proposal.fields) == 1:
            required_operands = 0
        if len(proposal.fields) - 1 != required_operands:
            loguru.logger.warning(
                f"Skip proposal {proposal.new_col_name}: expected {required_operands} extra fields, got {len(proposal.fields) - 1}"
            )
            continue

        field_index = 1
        for action in proposal.actions:
            col = None
            if action not in unary_actions:
                if action == ACTIONS.Degree and field_index >= len(proposal.fields):
                    col = None
                elif field_index >= len(proposal.fields):
                    loguru.logger.warning(
                        f"Skip proposal {proposal.new_col_name}: action {action} requires extra field at position {field_index}"
                    )
                    tmp_series = None
                    break
                else:
                    raw_field = proposal.fields[field_index]
                    col, is_valid = _resolve_operand(raw_field, df, new_df)
                    field_index += 1
                    if not is_valid and action in {ACTIONS.BinaryCategory, ACTIONS.StringConcat}:
                        col, is_valid = raw_field, True
                    if not is_valid:
                        loguru.logger.warning(f"Col {col} was not found; Cols base df {df.columns}; Cols new df {new_df.columns}")
                        tmp_series = None
                        break

            match action:
                case ACTIONS.Addition:
                    tmp_series = _coerce_numeric(tmp_series) + _coerce_numeric(col)
                case ACTIONS.Subtraction:
                    tmp_series = _coerce_numeric(tmp_series) - _coerce_numeric(col)
                case ACTIONS.Multiplication:
                    tmp_series = _coerce_numeric(tmp_series) * _coerce_numeric(col)
                case ACTIONS.Division:
                    denominator = _coerce_numeric(col)
                    tmp_series = _coerce_numeric(tmp_series) / denominator
                case ACTIONS.Degree:
                    exponent = 2 if col is None else _coerce_numeric(col)
                    tmp_series = _coerce_numeric(tmp_series) ** exponent
                case ACTIONS.Logarithm:
                    tmp_series = _coerce_numeric(tmp_series)
                    tmp_series = pd.Series(np.where(tmp_series > 0, np.log(tmp_series), np.nan), index=df.index)
                case ACTIONS.Threshold:
                    tmp_series = (_coerce_numeric(tmp_series) >= _coerce_numeric(col)).astype("Int8")
                case ACTIONS.BinaryCategory:
                    if isinstance(col, pd.Series):
                        tmp_series = (
                            tmp_series.astype("string").fillna("__nan__")
                            == col.astype("string").fillna("__nan__")
                        ).astype("Int8")
                    else:
                        tmp_series = (
                            tmp_series.astype("string").fillna("__nan__") == str(col)
                        ).astype("Int8")
                case ACTIONS.StringConcat:
                    if isinstance(col, pd.Series):
                        tmp_series = tmp_series.astype("string").fillna("__nan__") + "__" + col.astype("string").fillna("__nan__")
                    else:
                        tmp_series = tmp_series.astype("string").fillna("__nan__") + "__" + str(col)
                case ACTIONS.IsMissing:
                    tmp_series = tmp_series.isna().astype("Int8")

        if tmp_series is None:
            continue

        new_df[proposal.new_col_name] = tmp_series.replace([np.inf, -np.inf], np.nan)
        if proposal.save_col and proposal.new_col_name not in saved_columns:
            saved_columns.append(proposal.new_col_name)

    if not saved_columns:
        return pd.DataFrame(index=df.index)
    return new_df[saved_columns]
