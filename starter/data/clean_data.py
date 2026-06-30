"""
Clean the raw census.csv file.

The raw file (as provided by the Census Bureau / UCI repository) has stray
whitespace after every comma, which makes pandas read the column names and
the categorical values incorrectly (e.g. " Bachelors" instead of
"Bachelors"). This script strips that whitespace and writes out a clean
copy of the file, leaving the original census.csv untouched.

Usage:
    cd starter/data
    python clean_data.py
"""
import pandas as pd

RAW_PATH = "census.csv"
CLEAN_PATH = "census_clean.csv"


def clean_data(input_path: str = RAW_PATH, output_path: str = CLEAN_PATH) -> pd.DataFrame:
    """Load the raw census data, strip whitespace from headers/values, and save it."""
    df = pd.read_csv(input_path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    cleaned = clean_data()
    print(f"Cleaned data written to {CLEAN_PATH} with shape {cleaned.shape}")
