"""
PRC Data Challenge — Taxi-Out Time Prediction
Data Cleaning & Preprocessing script
Author: Kamran (Data Cleaning & Preprocessing Lead)

What this does:
1. Loads each monthly training parquet file
2. Adds 3 quality flags (never deletes rows — "flag, don't delete")
3. Combines all months into one dataset
4. Runs sanity checks (duplicates, row counts)
5. Saves the final clean dataset

Flags added:
- has_flight_plan_data : False if the flight-plan (_flt) columns are missing for this row
- taxi_time_valid       : False if TAXITIME_SEC_mvt is <= 0 (physically impossible)
- taxi_time_outlier     : True if TAXITIME_SEC_mvt is > 12 hours (43200s) — almost certainly a data error

Design decision: both DEP and ARR rows are KEPT (not filtered to DEP-only).
PHASE_mvt stays as a clear label so the taxi-out target is trained on DEP rows
only, while ARR rows remain available as a potential feature (e.g. airport
congestion signal).
"""

import glob
import pandas as pd

# ---- Settings -----------------------------------------------------------
INPUT_FOLDER = "/content"                 # where the monthly parquet files live
OUTPUT_FILE = "/content/cleaned_full_data.parquet"
OUTLIER_THRESHOLD_SECONDS = 43200         # 12 hours


def clean_month(filepath: str) -> pd.DataFrame:
    """Load one monthly parquet file and add the 3 quality flags."""
    df = pd.read_parquet(filepath)

    df["has_flight_plan_data"] = df["ADEP_flt"].notna()
    df["taxi_time_valid"] = df["TAXITIME_SEC_mvt"] > 0
    df["taxi_time_outlier"] = df["TAXITIME_SEC_mvt"] > OUTLIER_THRESHOLD_SECONDS

    return df


def build_clean_dataset(input_folder: str = INPUT_FOLDER) -> pd.DataFrame:
    """Find all monthly training files, clean each, and stack them into one table."""
    files = sorted(glob.glob(f"{input_folder}/training_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No training_*.parquet files found in {input_folder}. "
            "Upload the 12 monthly files first."
        )

    all_months = []
    for f in files:
        cleaned = clean_month(f)
        all_months.append(cleaned)
        print(f"{f}: {cleaned.shape}")

    full_data = pd.concat(all_months, ignore_index=True)
    return full_data


def run_sanity_checks(full_data: pd.DataFrame) -> None:
    """Print a summary so cleaning results can be checked at a glance."""
    print("\n--- Sanity checks ---")
    print("Total rows:", full_data.shape[0])
    print("Duplicate MVT_ID_mvt:", full_data["MVT_ID_mvt"].duplicated().sum())
    print("Missing flight-plan data:", (~full_data["has_flight_plan_data"]).sum())
    print("Invalid (<=0) taxi times:", (~full_data["taxi_time_valid"]).sum())
    print("Outlier (>12hr) taxi times:", full_data["taxi_time_outlier"].sum())
    print("\nRows by phase:")
    print(full_data["PHASE_mvt"].value_counts())


if __name__ == "__main__":
    full_data = build_clean_dataset()
    run_sanity_checks(full_data)

    full_data.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved cleaned dataset to {OUTPUT_FILE}")
