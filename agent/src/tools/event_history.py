from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.utils.storage import Storage


MAX_DIM_COLUMNS = 2
LOW_CARD_LIMIT = 200
UNIQUE_RATIO_LIMIT = 0.95


@dataclass
class HistorySchema:
    primary_key: str
    secondary_key: str
    event_key: str
    event_table_name: str
    interaction_table_name: str
    history_flag_col: str | None = None
    history_flag_value: str | None = None
    sequence_col: str | None = None
    repeat_col: str | None = None
    position_col: str | None = None
    dimension_table_name: str | None = None
    dimension_cols: list[str] | None = None


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, pd.NA)
    return numerator / denominator


def _list_table_names(storage: Storage) -> list[str]:
    return [header.split(" (", 1)[0] for header in storage.tables_headers]


def _is_id_like(col_name: str) -> bool:
    lower = col_name.lower()
    return lower == "id" or lower.endswith("_id")


def _detect_main_keys(base_df: pd.DataFrame, target_col: str = "target") -> list[str]:
    keys: list[str] = []
    n_rows = max(len(base_df), 1)
    for col in base_df.columns:
        if col == target_col or not _is_id_like(col):
            continue
        nunique = base_df[col].nunique(dropna=True)
        unique_ratio = nunique / n_rows
        if nunique <= 1 or unique_ratio >= UNIQUE_RATIO_LIMIT:
            continue
        keys.append(col)
    return keys


def _find_history_subset(event_df: pd.DataFrame) -> tuple[str | None, str | None]:
    priority_values = ("prior", "history", "historical", "past", "previous")
    for col in event_df.columns:
        lower = col.lower()
        if not any(token in lower for token in ("set", "split", "type", "stage", "period")):
            continue
        series = event_df[col].astype("string")
        nunique = series.nunique(dropna=True)
        if nunique <= 1 or nunique > 10:
            continue
        values = {str(v).lower() for v in series.dropna().unique()}
        for target in priority_values:
            if target in values:
                return col, target
    return None, None


def _pick_sequence_col(df: pd.DataFrame) -> str | None:
    priority = ("number", "sequence", "seq", "rank", "step")
    candidates: list[str] = []
    for col in df.columns:
        if not is_numeric_dtype(df[col]) or _is_id_like(col):
            continue
        lower = col.lower()
        if any(token in lower for token in priority):
            candidates.append(col)
    return candidates[0] if candidates else None


def _pick_repeat_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        lower = col.lower()
        if _is_id_like(col) or not is_numeric_dtype(df[col]):
            continue
        unique_values = set(pd.to_numeric(df[col], errors="coerce").dropna().unique().tolist())
        if not unique_values.issubset({0, 1}) or len(unique_values) <= 1:
            continue
        if any(token in lower for token in ("repeat", "reorder", "again", "flag", "binary")):
            return col
    for col in df.columns:
        if _is_id_like(col) or not is_numeric_dtype(df[col]):
            continue
        unique_values = set(pd.to_numeric(df[col], errors="coerce").dropna().unique().tolist())
        if unique_values.issubset({0, 1}) and len(unique_values) > 1:
            return col
    return None


def _pick_position_col(df: pd.DataFrame, excluded: set[str]) -> str | None:
    priority = ("cart", "position", "rank", "slot", "index")
    candidates: list[str] = []
    for col in df.columns:
        lower = col.lower()
        if col in excluded or _is_id_like(col) or not is_numeric_dtype(df[col]):
            continue
        if any(token in lower for token in priority):
            candidates.append(col)
    return candidates[0] if candidates else None


def _detect_dimension_table(
    storage: Storage,
    secondary_key: str,
    skip_tables: set[str],
) -> tuple[str | None, pd.DataFrame | None, list[str]]:
    for table_name in _list_table_names(storage):
        if table_name in skip_tables:
            continue
        try:
            df = storage.get_table(table_name)
        except Exception:
            continue
        if secondary_key not in df.columns:
            continue
        if df[secondary_key].nunique(dropna=True) < max(10, len(df) * 0.8):
            continue

        dim_cols: list[str] = []
        for col in df.columns:
            if col == secondary_key:
                continue
            nunique = df[col].nunique(dropna=True)
            if nunique <= 1 or nunique > LOW_CARD_LIMIT:
                continue
            dim_cols.append(col)
        if dim_cols:
            selected = dim_cols[:MAX_DIM_COLUMNS]
            return table_name, df[[secondary_key, *selected]].drop_duplicates(secondary_key), selected
    return None, None, []


def _infer_history_schema(
    storage: Storage,
    base_train: pd.DataFrame,
) -> HistorySchema | None:
    main_keys = _detect_main_keys(base_train)
    if len(main_keys) < 2:
        return None

    table_names = [
        name for name in _list_table_names(storage)
        if name not in {"train.csv", "test.csv", "data_dictionary.csv"}
    ]

    best_schema: HistorySchema | None = None
    best_score = float("-inf")

    for primary_key in main_keys:
        for secondary_key in main_keys:
            if secondary_key == primary_key:
                continue

            for event_table_name in table_names:
                try:
                    event_df = storage.get_table(event_table_name)
                except Exception:
                    continue
                if primary_key not in event_df.columns:
                    continue

                event_key_candidates = [
                    col for col in event_df.columns
                    if _is_id_like(col) and col != primary_key and col != secondary_key
                ]
                if not event_key_candidates:
                    continue

                for event_key in event_key_candidates:
                    event_unique_ratio = event_df[event_key].nunique(dropna=True) / max(len(event_df), 1)
                    if event_unique_ratio < 0.8:
                        continue

                    for interaction_table_name in table_names:
                        if interaction_table_name == event_table_name:
                            continue
                        try:
                            interaction_df = storage.get_table(interaction_table_name)
                        except Exception:
                            continue
                        if event_key not in interaction_df.columns or secondary_key not in interaction_df.columns:
                            continue
                        if len(interaction_df) <= interaction_df[event_key].nunique(dropna=True):
                            continue

                        history_flag_col, history_flag_value = _find_history_subset(event_df)
                        sequence_col = _pick_sequence_col(event_df)
                        repeat_col = _pick_repeat_col(interaction_df)
                        position_col = _pick_position_col(interaction_df, excluded={event_key, secondary_key})

                        score = 0.0
                        score += 3.0
                        score += 1.0 if history_flag_col else 0.0
                        score += 1.0 if sequence_col else 0.0
                        score += 1.0 if repeat_col else 0.0
                        score += 1.0 if position_col else 0.0
                        score += min(len(interaction_df) / max(len(event_df), 1), 5.0)

                        if score > best_score:
                            best_score = score
                            best_schema = HistorySchema(
                                primary_key=primary_key,
                                secondary_key=secondary_key,
                                event_key=event_key,
                                event_table_name=event_table_name,
                                interaction_table_name=interaction_table_name,
                                history_flag_col=history_flag_col,
                                history_flag_value=history_flag_value,
                                sequence_col=sequence_col,
                                repeat_col=repeat_col,
                                position_col=position_col,
                            )

    if best_schema is None:
        return None

    dim_table_name, _, dim_cols = _detect_dimension_table(
        storage=storage,
        secondary_key=best_schema.secondary_key,
        skip_tables={best_schema.event_table_name, best_schema.interaction_table_name},
    )
    best_schema.dimension_table_name = dim_table_name
    best_schema.dimension_cols = dim_cols
    return best_schema


def _prepare_history_dataframe(storage: Storage, schema: HistorySchema) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    event_df = storage.get_table(schema.event_table_name).copy()
    interaction_df = storage.get_table(schema.interaction_table_name).copy()

    event_cols = [schema.event_key, schema.primary_key]
    for col in [schema.history_flag_col, schema.sequence_col]:
        if col and col in event_df.columns and col not in event_cols:
            event_cols.append(col)
    event_small = event_df[event_cols].copy()
    if schema.history_flag_col and schema.history_flag_value:
        history_mask = event_small[schema.history_flag_col].astype("string").str.lower().eq(schema.history_flag_value)
        event_small = event_small[history_mask].copy()

    interaction_cols = [schema.event_key, schema.secondary_key]
    for col in [schema.repeat_col, schema.position_col]:
        if col and col in interaction_df.columns and col not in interaction_cols:
            interaction_cols.append(col)
    interaction_small = interaction_df[interaction_cols].copy()

    history = interaction_small.merge(event_small, on=schema.event_key, how="inner")
    if history.empty:
        return history, None

    dim_df = None
    if schema.dimension_table_name and schema.dimension_cols:
        dim_table = storage.get_table(schema.dimension_table_name)
        dim_df = dim_table[[schema.secondary_key, *schema.dimension_cols]].drop_duplicates(schema.secondary_key)
        history = history.merge(dim_df, on=schema.secondary_key, how="left")
    return history, dim_df


def generate_event_history_features(
    storage: Storage,
    input_train: pd.DataFrame,
    input_test: pd.DataFrame,
    merged_train: pd.DataFrame,
    merged_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    schema = _infer_history_schema(storage, input_train)
    if schema is None:
        return pd.DataFrame(index=merged_train.index), pd.DataFrame(index=merged_test.index), []

    history, dim_df = _prepare_history_dataframe(storage, schema)
    if history.empty:
        return pd.DataFrame(index=merged_train.index), pd.DataFrame(index=merged_test.index), []

    entity_total_events = history.groupby(schema.primary_key, dropna=False)[schema.event_key].nunique().rename("entity_event_count")

    pair_agg: dict[str, str | list[str]] = {schema.event_key: "nunique"}
    if schema.repeat_col and schema.repeat_col in history.columns:
        pair_agg[schema.repeat_col] = ["sum", "mean"]
    if schema.position_col and schema.position_col in history.columns:
        pair_agg[schema.position_col] = "mean"
    if schema.sequence_col and schema.sequence_col in history.columns:
        pair_agg[schema.sequence_col] = ["max", "min"]

    pair_stats = history.groupby([schema.primary_key, schema.secondary_key], dropna=False).agg(pair_agg)
    pair_columns: list[str] = []
    for col in pair_stats.columns:
        if col == (schema.event_key, "nunique") or col == schema.event_key:
            pair_columns.append("pair_event_count")
        elif schema.repeat_col and col == (schema.repeat_col, "sum"):
            pair_columns.append("pair_repeat_sum")
        elif schema.repeat_col and col == (schema.repeat_col, "mean"):
            pair_columns.append("pair_repeat_rate")
        elif schema.position_col and (col == (schema.position_col, "mean") or col == schema.position_col):
            pair_columns.append("pair_position_mean")
        elif schema.sequence_col and col == (schema.sequence_col, "max"):
            pair_columns.append("pair_last_sequence")
        elif schema.sequence_col and col == (schema.sequence_col, "min"):
            pair_columns.append("pair_first_sequence")
        else:
            pair_columns.append("_".join(str(part) for part in col if part) if isinstance(col, tuple) else str(col))
    pair_stats.columns = pair_columns
    pair_stats = pair_stats.reset_index().merge(entity_total_events.reset_index(), on=schema.primary_key, how="left")
    pair_stats["pair_event_rate"] = _safe_ratio(pair_stats["pair_event_count"], pair_stats["entity_event_count"])
    if "pair_last_sequence" in pair_stats.columns:
        primary_last_seq = history.groupby(schema.primary_key, dropna=False)[schema.sequence_col].max().rename("entity_last_sequence")
        pair_stats = pair_stats.merge(primary_last_seq.reset_index(), on=schema.primary_key, how="left")
        pair_stats["pair_events_since_last"] = pair_stats["entity_last_sequence"] - pair_stats["pair_last_sequence"]
        pair_stats = pair_stats.drop(columns=["entity_last_sequence"], errors="ignore")
    if "pair_first_sequence" in pair_stats.columns and "pair_last_sequence" in pair_stats.columns:
        pair_stats["pair_event_span"] = pair_stats["pair_last_sequence"] - pair_stats["pair_first_sequence"]
    pair_stats = pair_stats.drop(columns=["entity_event_count"], errors="ignore")

    secondary_agg = {
        schema.event_key: "nunique",
        schema.primary_key: "nunique",
    }
    if schema.repeat_col and schema.repeat_col in history.columns:
        secondary_agg[schema.repeat_col] = "mean"
    if schema.position_col and schema.position_col in history.columns:
        secondary_agg[schema.position_col] = "mean"
    secondary_stats = history.groupby(schema.secondary_key, dropna=False).agg(secondary_agg).reset_index()
    secondary_stats = secondary_stats.rename(
        columns={
            schema.event_key: "secondary_event_count",
            schema.primary_key: "secondary_primary_count",
            schema.repeat_col: "secondary_repeat_rate" if schema.repeat_col else schema.repeat_col,
            schema.position_col: "secondary_position_mean" if schema.position_col else schema.position_col,
        }
    )

    dim_feature_frames: list[tuple[pd.DataFrame, list[str]]] = []
    if dim_df is not None and schema.dimension_cols:
        for dim_col in schema.dimension_cols:
            dim_stats = history.groupby([schema.primary_key, dim_col], dropna=False).agg(
                dim_event_count=(schema.event_key, "nunique")
            ).reset_index()
            dim_stats = dim_stats.merge(entity_total_events.reset_index(), on=schema.primary_key, how="left")
            count_col = f"{dim_col}_event_count"
            dim_stats = dim_stats.rename(columns={"dim_event_count": count_col})
            dim_stats[f"{dim_col}_event_rate"] = _safe_ratio(dim_stats[count_col], dim_stats["entity_event_count"])
            dim_stats = dim_stats.drop(columns=["entity_event_count"], errors="ignore")
            dim_feature_frames.append((dim_stats, [schema.primary_key, dim_col]))

    def apply_features(base_df: pd.DataFrame) -> pd.DataFrame:
        required_keys = [schema.primary_key, schema.secondary_key]
        if not all(key in base_df.columns for key in required_keys):
            return pd.DataFrame(index=base_df.index)

        result = base_df[required_keys].copy()
        if dim_df is not None:
            result = result.merge(dim_df, on=schema.secondary_key, how="left")
        result = result.merge(pair_stats, on=[schema.primary_key, schema.secondary_key], how="left")
        result = result.merge(secondary_stats, on=schema.secondary_key, how="left")
        for dim_frame, join_keys in dim_feature_frames:
            if all(key in result.columns for key in join_keys):
                result = result.merge(dim_frame, on=join_keys, how="left")
        return result.drop(columns=[schema.primary_key, schema.secondary_key, *(schema.dimension_cols or [])], errors="ignore")

    train_features = apply_features(merged_train).reset_index(drop=True)
    test_features = apply_features(merged_test).reset_index(drop=True)
    train_features = train_features.loc[:, ~train_features.columns.duplicated()]
    test_features = test_features.loc[:, ~test_features.columns.duplicated()]
    return train_features, test_features, train_features.columns.tolist()
