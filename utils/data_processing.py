"""Data loading, column guessing, quality checks and safe cleaning.

Everything here works on plain pandas DataFrames and never touches Streamlit,
so it can be unit-tested (or reused) outside the app.
"""
from __future__ import annotations

import pandas as pd

# Keywords used only to PRE-SELECT likely columns in the mapping UI.
# The user can always override the guess, so no column names are assumed.
# The order matters: earlier hints win over later ones.
HINTS = {
    "date": ["date", "day", "time", "period", "month", "order"],
    "sales": ["sales", "revenue", "amount", "total", "income", "turnover", "price"],
    "quantity": ["quantity", "qty", "units", "volume", "count", "demand"],
    "product": ["product", "item", "sku", "name", "description"],
    "category": ["category", "type", "segment", "department", "class", "group"],
    "region": ["region", "state", "country", "city", "area", "zone", "location", "market"],
}


def load_csv(file) -> pd.DataFrame:
    """Read a CSV, trying a few common encodings and auto-detecting the separator."""
    last_error = None
    # FIX: "utf-8-sig" must come FIRST. It reads files with or without a BOM
    # (Excel's "CSV UTF-8" adds one). With plain "utf-8" first, the BOM was kept
    # and the first column was called "\ufeffOrder_Date"; "utf-8-sig" was never reached.
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            file.seek(0)  # rewind, because a failed attempt may have consumed the stream
            df = pd.read_csv(file, encoding=enc, sep=None, engine="python")
            df.columns = [str(c).strip() for c in df.columns]  # tidy header whitespace
            if df.empty:
                raise ValueError("The file has no rows.")
            return df
        except UnicodeDecodeError as e:
            last_error = e  # try the next encoding
    raise ValueError(f"Could not read the file ({last_error}).")


def _mostly_numeric(series: pd.Series, threshold: float = 0.8) -> bool:
    """True if the column is numeric, or if most of its values can be read as numbers.

    This lets text columns such as "$1,200" count as numeric, which is exactly
    what clean_data() is able to convert later.
    """
    if pd.api.types.is_numeric_dtype(series):
        return True
    sample = series.dropna().head(200)  # a sample is enough and keeps this fast
    if sample.empty:
        return False
    return bool(_to_number(sample).notna().mean() >= threshold)


def guess_column(df: pd.DataFrame, field: str, numeric: bool = False) -> str | None:
    """Best-effort guess of which column holds `field` (date, sales, product ...)."""
    for hint in HINTS[field]:
        for col in df.columns:
            if hint in col.lower():
                # FIX: the old check used is_numeric_dtype only, so a "Revenue" column
                # holding "$1,200" strings was rejected even though the README says
                # currency symbols are supported. Use the looser "mostly numeric" test.
                if numeric and not _mostly_numeric(df[col]):
                    continue
                return col
    return None


def column_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Column types, missing values and unique counts for the preview tab."""
    return pd.DataFrame({
        "Column": df.columns,
        "Type": [str(t) for t in df.dtypes],
        "Missing": df.isnull().sum().values,
        "Missing %": (df.isnull().mean() * 100).round(1).values,
        "Unique": df.nunique().values,
    })


def quality_warnings(df: pd.DataFrame) -> list[str]:
    """Plain-language warnings about the raw data."""
    warnings = []
    dups = int(df.duplicated().sum())
    if dups:
        warnings.append(f"{dups:,} duplicate rows found (they will be removed).")
    missing = df.isnull().sum()
    for col, n in missing[missing > 0].items():
        warnings.append(f"Column '{col}' has {n:,} missing values ({n / len(df):.1%}).")
    if len(df) < 30:
        warnings.append("Fewer than 30 rows: analysis and forecasts will be limited.")
    return warnings


def clean_data(df, mapping: dict) -> tuple[pd.DataFrame, list[str]]:
    """Return a standardised dataframe and a log of what was cleaned.

    Output columns: date, sales, quantity, product, category, region
    (only those the user mapped).
    """
    log = []
    n0 = len(df)

    # FIX (data loss): remove duplicates on the RAW rows, before keeping only the
    # mapped columns. The old code de-duplicated after selecting just date + sales
    # (+ whatever else was mapped), so two genuine orders on the same day with the
    # same amount were thrown away as "duplicates". Comparing complete raw rows only
    # removes rows that are identical in every column.
    df = df.drop_duplicates()
    if n0 - len(df):
        log.append(f"Removed {n0 - len(df):,} duplicate rows.")

    out = pd.DataFrame(index=df.index)

    # utc=True makes mixed time zones (e.g. "+05:30" and "Z") parse instead of crash;
    # tz_localize(None) then drops the zone so later steps (resample, Prophet) get plain dates.
    out["date"] = pd.to_datetime(df[mapping["date"]], errors="coerce", utc=True).dt.tz_localize(None)
    out["sales"] = _to_number(df[mapping["sales"]])
    if mapping.get("quantity"):
        out["quantity"] = _to_number(df[mapping["quantity"]])
    for field in ("product", "category", "region"):
        if mapping.get(field):
            out[field] = df[mapping[field]].astype("string").str.strip()

    # Rows without a usable date or sales amount cannot be analysed, so they are dropped.
    # Count the two reasons separately so the log never double counts a row.
    bad_dates = int(out["date"].isna().sum())
    bad_sales = int((out["sales"].isna() & out["date"].notna()).sum())
    out = out.dropna(subset=["date", "sales"])
    if bad_dates:
        log.append(f"Removed {bad_dates:,} rows with invalid or missing dates.")
    if bad_sales:
        log.append(f"Removed {bad_sales:,} rows with non-numeric or missing sales.")

    # Optional columns are filled rather than dropped, so a gap does not cost a sale.
    if "quantity" in out:
        missing_q = int(out["quantity"].isna().sum())
        if missing_q:
            out["quantity"] = out["quantity"].fillna(0)
            log.append(f"Filled {missing_q:,} missing quantity values with 0.")

    for field in ("product", "category", "region"):
        if field in out:
            missing = int(out[field].isna().sum())
            if missing:
                out[field] = out[field].fillna("Unknown")
                log.append(f"Labelled {missing:,} missing {field} values as 'Unknown'.")

    out = out.sort_values("date").reset_index(drop=True)
    log.append(f"Kept {len(out):,} of {n0:,} rows after cleaning.")
    return out, log


def _to_number(series: pd.Series) -> pd.Series:
    """Convert to numbers, tolerating currency symbols and thousand separators."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    # Keep only digits, the decimal point and a minus sign: "$1,200.50" -> "1200.50"
    cleaned = series.astype("string").str.replace(r"[^\d.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")  # anything unreadable becomes NaN
