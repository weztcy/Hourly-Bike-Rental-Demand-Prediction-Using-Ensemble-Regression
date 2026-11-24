"""
Data Preparation and Feature Engineering Module
Hourly Bike Rental Demand Prediction via Ensemble Regression
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def inverse_nonnegative(y):
    """Inverse transform for log1p predictions with non-negative lower bound."""
    return np.clip(np.expm1(y), 0, None)

def load_data(csv_path="Bike_Sharing.csv"):
    """
    Loads dataset, parses chronological timestamp, and sorts observations.
    Performs data leakage prevention audit.
    """
    df = pd.read_csv(csv_path)
    
    # Verify leakage & target equality
    assert (df["casual"] + df["registered"] == df["cnt"]).all(), "Target mismatch: cnt != casual + registered!"
    
    # Parse timestamps
    df["dteday"] = pd.to_datetime(df["dteday"])
    df["timestamp"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    return df

class BikeFeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Custom transformer to engineer cyclical and domain features cleanly in a Pipeline.
    """
    def __init__(self, drop_atemp=True):
        self.drop_atemp = drop_atemp
        self.feature_names_out_ = None
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_out = X.copy()
        
        # Cyclical transformations
        X_out["hr_sin"] = np.sin(2 * np.pi * X_out["hr"] / 24.0)
        X_out["hr_cos"] = np.cos(2 * np.pi * X_out["hr"] / 24.0)
        X_out["mnth_sin"] = np.sin(2 * np.pi * X_out["mnth"] / 12.0)
        X_out["mnth_cos"] = np.cos(2 * np.pi * X_out["mnth"] / 12.0)
        X_out["weekday_sin"] = np.sin(2 * np.pi * X_out["weekday"] / 7.0)
        X_out["weekday_cos"] = np.cos(2 * np.pi * X_out["weekday"] / 7.0)
        
        # Domain interaction: Rush hour commute
        X_out["is_rush_hour"] = (((X_out["hr"].isin([8, 9, 17, 18])) & (X_out["workingday"] == 1))).astype(int)
        
        # Weather comfort interaction: temp / humidity ratio
        X_out["temp_hum_ratio"] = X_out["temp"] / (X_out["hum"] + 1e-4)
        
        # Drop extreme collinear feature if requested
        if self.drop_atemp and "atemp" in X_out.columns:
            X_out = X_out.drop(columns=["atemp"])
            
        self.feature_names_out_ = list(X_out.columns)
        return X_out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)

def prepare_features_and_target(df, drop_leakage=True):
    """
    Splits DataFrame into features X and target y.
    Strictly removes target leakage: 'instant', 'casual', 'registered', 'cnt', 'timestamp', 'dteday'.
    """
    leakage_cols = ["instant", "casual", "registered", "cnt", "dteday", "timestamp"]
    feature_cols = [c for c in df.columns if c not in leakage_cols]
    
    X = df[feature_cols].copy()
    y = df["cnt"].copy()
    
    return X, y, df["timestamp"]

def chronological_split(df, train_frac=0.70, val_frac=0.15):
    """
    Splits data chronologically into Train (70%), Validation (15%), and Test (15%).
    Never shuffles to preserve real-world forecasting validity.
    """
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()
    
    splits_info = {
        "train": {
            "n_samples": len(train_df),
            "start_time": str(train_df["timestamp"].min()),
            "end_time": str(train_df["timestamp"].max()),
            "target_mean": float(train_df["cnt"].mean()),
            "target_std": float(train_df["cnt"].std())
        },
        "val": {
            "n_samples": len(val_df),
            "start_time": str(val_df["timestamp"].min()),
            "end_time": str(val_df["timestamp"].max()),
            "target_mean": float(val_df["cnt"].mean()),
            "target_std": float(val_df["cnt"].std())
        },
        "test": {
            "n_samples": len(test_df),
            "start_time": str(test_df["timestamp"].min()),
            "end_time": str(test_df["timestamp"].max()),
            "target_mean": float(test_df["cnt"].mean()),
            "target_std": float(test_df["cnt"].std())
        }
    }
    
    return train_df, val_df, test_df, splits_info

if __name__ == "__main__":
    df = load_data()
    train_df, val_df, test_df, info = chronological_split(df)
    print("Dataset split summary:")
    import json
    print(json.dumps(info, indent=2))
