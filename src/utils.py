# src/utils.py
# Helper utilities used by train.py

import pandas as pd

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column headers to a clean snake_case:
    - strip leading/trailing spaces
    - lowercase
    - replace any non-alphanumeric sequence with a single underscore
    - strip leading/trailing underscores
    """
    df = df.copy()
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r'[^a-z0-9]+', '_', regex=True)
          .str.strip('_')
    )
    return df