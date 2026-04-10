import pandas as pd
import numpy as np


def create_stat(df: pd.DataFrame, target_col: str = None) -> str:
    lines = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    lines.append("=" * 70)
    lines.append("ОБЩАЯ ИНФОРМАЦИЯ")
    lines.append("=" * 70)
    lines.append(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
    lines.append(f"Числовые колонки ({len(numeric_cols)}): {numeric_cols}")
    lines.append(f"Категориальные колонки ({len(categorical_cols)}): {categorical_cols}")
    lines.append(f"Datetime колонки ({len(datetime_cols)}): {datetime_cols}")

    lines.append("\nПропуски по колонкам (%):")
    missing = (df.isna().mean() * 100).round(2)
    missing_nonzero = missing[missing > 0].sort_values(ascending=False)
    if missing_nonzero.empty:
        lines.append("пропусков нет")
    else:
        for col, pct in missing_nonzero.items():
            lines.append(f"{col}: {pct}%")

    lines.append("\n" + "=" * 70)
    lines.append("DESCRIBE — ЧИСЛОВЫЕ")
    lines.append("=" * 70)
    if numeric_cols:
        lines.append(df[numeric_cols].describe().round(4).to_string())
    else:
        lines.append("нет числовых колонок")

    lines.append("\n" + "=" * 70)
    lines.append("DESCRIBE — КАТЕГОРИАЛЬНЫЕ")
    lines.append("=" * 70)
    if categorical_cols:
        lines.append(df[categorical_cols].describe().to_string())
    else:
        lines.append("нет категориальных колонок")
    
    lines.append("\n" + "=" * 70)
    lines.append("КОРРЕЛЯЦИЯ PEARSON — ТОП-10 ПАР ПО |corr|")
    lines.append("=" * 70)
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(method="pearson")
        pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                val = corr.iloc[i, j]
                if pd.notna(val):
                    pairs.append((numeric_cols[i], numeric_cols[j], float(val)))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        top_pairs = pairs[:10]
        if top_pairs:
            for c1, c2, val in top_pairs:
                lines.append(f"  {c1} <-> {c2}: {val:.4f}")
        else:
            lines.append("нет валидных пар")
    else:
        lines.append("недостаточно числовых колонок для корреляции")

    lines.append("\n" + "=" * 70)
    lines.append("УНИКАЛЬНЫЕ ЗНАЧЕНИЯ КАТЕГОРИАЛЬНЫХ (топ-5)")
    lines.append("=" * 70)
    if categorical_cols:
        for col in categorical_cols:
            n_unique = df[col].nunique(dropna=True)
            lines.append(f"\n{col}: {n_unique} уникальных значений")
            top5 = df[col].value_counts(dropna=True, normalize=True).head(5)
            for val, share in top5.items():
                lines.append(f"  {val!r}: {share * 100:.2f}%")
    else:
        lines.append("нет категориальных колонок")

    target_present = target_col is not None and target_col in df.columns

    #распределение и дисбаланс
    if target_present:
        lines.append("\n" + "=" * 70)
        lines.append(f"РАСПРЕДЕЛЕНИЕ ТАРГЕТА: {target_col} (классификация)")
        lines.append("=" * 70)
        target_series = df[target_col]
        counts = target_series.value_counts(dropna=False)
        shares = target_series.value_counts(dropna=False, normalize=True)
        n_classes = target_series.nunique(dropna=True)
        lines.append(f"Количество классов: {n_classes}")
        lines.append("Баланс классов:")
        for val in counts.index:
            lines.append(f"  {val!r}: {counts[val]} ({shares[val] * 100:.2f}%)")
        # индикатор дисбаланса
        if len(shares) >= 2:
            ratio = shares.max() / shares.min()
            lines.append(f"Соотношение max/min класса: {ratio:.2f}")

    # выбросы верх и низ
    lines.append("\n" + "=" * 70)
    lines.append("ВЫБРОСЫ (за пределами 1.5 × IQR)")
    lines.append("=" * 70)
    if numeric_cols:
        n_rows = len(df)
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
            pct_out = (n_out / n_rows * 100) if n_rows else 0.0
            lines.append(f"  {col}: {n_out} ({pct_out:.2f}%)  [границы: {lower:.4f} .. {upper:.4f}]")
    else:
        lines.append("нет числовых колонок")

    # числа и таргет 
    if target_present:
        lines.append("\n" + "=" * 70)
        lines.append(f"ВЗАИМОСВЯЗЬ ЧИСЛОВЫХ ФИЧЕЙ С ТАРГЕТОМ: {target_col}")
        lines.append("=" * 70)
        numeric_no_target = [c for c in numeric_cols if c != target_col]
        if not numeric_no_target:
            lines.append("нет числовых фичей кроме таргета")
        else:
            means = df.groupby(target_col)[numeric_no_target].mean().round(4)
            lines.append("Средние числовых фичей по классам таргета:")
            lines.append(means.to_string())

    return "\n".join(lines)
