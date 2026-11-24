import json
import pandas as pd
import numpy as np

def audit_dataset():
    print("=== TAHAP A: DATA AUDIT & UNDERSTANDING ===")
    
    # 1. Read metadata
    with open("metadata.json", "r") as f:
        meta = json.load(f)
    print("\n--- METADATA SUMMARY ---")
    print(f"Dataset Name: {meta.get('name')}")
    print(f"Num Instances in metadata: {meta.get('num_instances')}")
    print(f"Num Features in metadata: {meta.get('num_features')}")
    print(f"Target col: {meta.get('target_col')}")
    print(f"Index col: {meta.get('index_col')}")

    # 2. Read CSV
    df = pd.read_csv("Bike_Sharing.csv")
    print("\n--- CSV SUMMARY ---")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes}")
    
    # Missing values
    missing = df.isnull().sum()
    print(f"\nMissing values per column:\n{missing[missing > 0] if (missing > 0).any() else 'None (0 missing values)'}")
    
    # Duplicates
    full_duplicates = df.duplicated().sum()
    instant_duplicates = df["instant"].duplicated().sum()
    print(f"\nFull row duplicates: {full_duplicates}")
    print(f"Instant duplicates: {instant_duplicates}")
    print(f"Instant min: {df['instant'].min()}, max: {df['instant'].max()}, is monotonic: {df['instant'].is_monotonic_increasing}")
    
    # Check target consistency: cnt = casual + registered
    calc_cnt = df["casual"] + df["registered"]
    matches = (calc_cnt == df["cnt"]).all()
    diff_count = (calc_cnt != df["cnt"]).sum()
    print(f"\nVerification cnt == casual + registered: {matches} (Mismatches: {diff_count})")
    
    # Timestamp parsing
    df["dteday_dt"] = pd.to_datetime(df["dteday"])
    df["timestamp"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    print(f"\nDate range: {df['dteday'].min()} to {df['dteday'].max()}")
    print(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Timestamp duplicates: {df['timestamp'].duplicated().sum()}")
    print(f"Is timestamp strictly monotonic increasing: {df['timestamp'].is_monotonic_increasing}")
    
    # Observations per day
    day_counts = df.groupby("dteday").size()
    print(f"\nTotal calendar days in dataset: {len(day_counts)}")
    print(f"Days with exactly 24 observations: {(day_counts == 24).sum()}")
    incomplete_days = day_counts[day_counts < 24]
    print(f"Days with < 24 observations: {len(incomplete_days)}")
    print("Sample incomplete days:\n", incomplete_days.head(10))
    print(f"Distribution of hourly counts per day: {day_counts.value_counts().sort_index()}")
    
    # Expected hours across date range
    expected_hours = (pd.to_datetime('2012-12-31 23:00:00') - pd.to_datetime('2011-01-01 00:00:00')).total_seconds() / 3600 + 1
    print(f"Expected total hourly slots between 2011-01-01 and 2012-12-31: {int(expected_hours)}")
    print(f"Missing hourly slots: {int(expected_hours) - len(df)}")
    
    # Summary statistics
    print("\n--- DESCRIPTIVE STATISTICS FOR TARGET & NUMERICS ---")
    num_cols = ["cnt", "casual", "registered", "temp", "atemp", "hum", "windspeed"]
    print(df[num_cols].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]])
    
    # Categorical distributions
    cat_cols = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
    print("\n--- CATEGORICAL / DISCRETE VALUE COUNTS ---")
    for col in cat_cols:
        print(f"{col} unique values: {sorted(df[col].unique())}")
    
    # Correlation between temp and atemp
    corr_temp_atemp = df["temp"].corr(df["atemp"])
    print(f"\nPearson correlation temp vs atemp: {corr_temp_atemp:.5f}")
    
    # Comparison with metadata.json
    print("\n--- COMPARISON METADATA vs CSV ---")
    print(f"Metadata instances: {meta.get('num_instances')} vs CSV rows: {len(df)} (Difference: {meta.get('num_instances') - len(df)})")
    print(f"Metadata features: {meta.get('num_features')} vs CSV columns: {len(df.columns) - 2} (Note: CSV has 17 cols including instant, dteday, casual, registered, cnt)")

if __name__ == "__main__":
    audit_dataset()
