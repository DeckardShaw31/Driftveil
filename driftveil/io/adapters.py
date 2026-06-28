import pandas as pd
from typing import Any, Optional

def to_pandas(df: Any) -> Optional[pd.DataFrame]:
    """Convert a supported DataFrame (like Polars) to a Pandas DataFrame."""
    if df is None:
        return None
        
    # Check if the DataFrame has a to_pandas method (e.g. Polars)
    if hasattr(df, "to_pandas") and callable(getattr(df, "to_pandas")):
        return df.to_pandas()
        
    return df
