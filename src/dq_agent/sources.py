"""Loading a dataset from whatever the user uploads.

Everything is read as text, literally, with pandas' own missing-value
conversion switched off. Both matter:

- Letting pandas infer types would repair a mixed-type column on read, so the
  detector would never see the evidence.
- Letting pandas apply its default `na_values` would turn "N/A", "null", "-"
  and a dozen other spellings into NaN on the way in, hiding the fact that a
  column uses placeholder text at all. `detectors.profile` decides what counts
  as blank instead, in one place, and says so in the report.
"""

from __future__ import annotations

import io

import pandas as pd


class LoadError(ValueError):
    pass


def sniff_separator(text: str) -> str:
    """Pick the delimiter from the header row, or default to a comma.

    pandas' own sniffer (`sep=None`) guesses from the characters present, so a
    single-column file headed "total" is split on the letter "t" into three
    nonsense columns. Counting known delimiters cannot make that mistake.
    """
    header = text.lstrip().splitlines()[0] if text.strip() else ""
    counts = {sep: header.count(sep) for sep in (",", "\t", ";", "|")}
    best = max(counts, key=lambda s: counts[s])
    return best if counts[best] > 0 else ","


MAX_ROWS = 200_000


def load_dataframe(filename: str, data: bytes) -> pd.DataFrame:
    lowered = filename.lower()
    try:
        if lowered.endswith((".xlsx", ".xlsm")):
            df = pd.read_excel(io.BytesIO(data), dtype=str, keep_default_na=False)
        elif lowered.endswith((".csv", ".tsv", ".txt")):
            text = data.decode("utf-8-sig", errors="replace")
            if not text.strip():
                raise LoadError("That file is empty.")
            sep = "\t" if lowered.endswith(".tsv") else sniff_separator(text)
            df = pd.read_csv(
                io.StringIO(text),
                dtype=str,
                sep=sep,
                engine="python",
                skip_blank_lines=True,
                keep_default_na=False,
            )
        else:
            raise LoadError("Unsupported file type. Upload a .csv, .tsv or .xlsx file.")
    except LoadError:
        raise
    except Exception as exc:
        raise LoadError(f"Could not read that file: {exc}") from exc

    if df.empty:
        raise LoadError("That file has no rows.")
    if len(df) > MAX_ROWS:
        raise LoadError(f"That file has {len(df):,} rows. The limit is {MAX_ROWS:,}.")
    df.columns = [str(c).strip() if str(c).strip() else f"column_{i + 1}" for i, c in enumerate(df.columns)]
    return df


def preview(df: pd.DataFrame, rows: int = 20) -> dict:
    """A JSON-safe slice for the browser."""
    head = df.head(rows)
    return {
        "columns": [str(c) for c in df.columns],
        "rows": [["" if pd.isna(v) else str(v) for v in row] for row in head.itertuples(index=False)],
        "total_rows": len(df),
    }


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
