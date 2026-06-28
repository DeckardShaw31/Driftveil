import os
import sys
import argparse
import pandas as pd
from driftveil import DriftVeil

def load_data(path: str) -> pd.DataFrame:
    """Load a dataset from CSV, Parquet, or JSON based on file extension."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
        
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    elif ext == ".json":
        return pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Driftveil CLI supports: .csv, .parquet, .json")

def main():
    parser = argparse.ArgumentParser(
        description="Driftveil CLI: Enforce statistical data contracts on DataFrames."
    )
    parser.add_argument(
        "data",
        help="Path to the new data file to check (CSV, Parquet, or JSON)."
    )
    parser.add_argument(
        "--pact",
        required=True,
        help="Path to the saved Driftveil pact JSON file."
    )
    parser.add_argument(
        "--raise-on-fail",
        action="store_true",
        help="If set, exit with code 1 if any contract fails. Otherwise exit with 0."
    )
    
    args = parser.parse_args()
    
    try:
        # Load pact
        if not os.path.exists(args.pact):
            print(f"Error: Pact file not found at {args.pact}", file=sys.stderr)
            sys.exit(1)
            
        pact = DriftVeil.load(args.pact)
        
        # Load dataset
        new_df = load_data(args.data)
        
        # Enforce
        report = pact.enforce(new_df)
        
        # Print summary
        print(report.summary())
        
        # Determine exit code
        if not report.passed and args.raise_on_fail:
            print("\n✗ Contract enforcement FAILED.", file=sys.stderr)
            sys.exit(1)
            
        print("\n✓ Contract enforcement completed successfully.")
        sys.exit(0)
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
