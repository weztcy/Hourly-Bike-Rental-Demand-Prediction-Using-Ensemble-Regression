"""
Model Training and Validation Module
Hourly Bike Rental Demand Prediction via Ensemble Regression
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    GradientBoostingRegressor,
    VotingRegressor
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.data_preparation import load_data, chronological_split, prepare_features_and_target, BikeFeatureEngineering, inverse_nonnegative

def build_model_pipeline(regressor, is_linear=False):
    """
    Builds a full end-to-end pipeline including feature engineering,
    optional scaling/one-hot encoding for linear models, and log1p target transformation.
    """
    if is_linear:
        cat_cols = ["season", "yr", "holiday", "weekday", "workingday", "weathersit"]
        num_cols = ["temp", "hum", "windspeed", "hr_sin", "hr_cos", "mnth_sin", "mnth_cos", "weekday_sin", "weekday_cos", "is_rush_hour", "temp_hum_ratio"]
        encoder_scaler = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
        ])
        pipe = Pipeline([
            ("engineer", BikeFeatureEngineering(drop_atemp=True)),
            ("preprocess", encoder_scaler),
            ("regressor", regressor)
        ])
    else:
        pipe = Pipeline([
            ("engineer", BikeFeatureEngineering(drop_atemp=True)),
            ("regressor", regressor)
        ])
        
    return TransformedTargetRegressor(
        regressor=pipe,
        func=np.log1p,
        inverse_func=inverse_nonnegative,
        check_inverse=False
    )

def plot_model_comparison(results_df, output_path="visualizations/model_comparison.png"):
    """
    Generates high-resolution publication-quality model comparison plots.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")
    
    # Filter out models with extreme negative R2 for clean display, or show all
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    
    order = results_df.sort_values("Val MAE", ascending=False)["Model"].tolist()
    colors = ["#7f7f7f" if "Dummy" in m else "#aec7e8" if "Linear" in m or "Ridge" in m else "#1f77b4" if "Ensemble" in m else "#2ca02c" for m in order]
    
    # MAE Plot
    sns.barplot(data=results_df, y="Model", x="Val MAE", order=order, palette=colors, ax=axes[0])
    axes[0].set_title("Validation MAE (Lower is Better)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Mean Absolute Error (bikes/hour)", fontsize=11)
    for p in axes[0].patches:
        width = p.get_width()
        axes[0].annotate(f"{width:.1f}", (width + 2, p.get_y() + p.get_height() / 2.),
                         va='center', fontsize=9, fontweight='bold')
    
    # RMSE Plot
    sns.barplot(data=results_df, y="Model", x="Val RMSE", order=order, palette=colors, ax=axes[1])
    axes[1].set_title("Validation RMSE (Lower is Better)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Root Mean Squared Error", fontsize=11)
    axes[1].set_ylabel("")
    axes[1].set_yticklabels([])
    for p in axes[1].patches:
        width = p.get_width()
        axes[1].annotate(f"{width:.1f}", (width + 3, p.get_y() + p.get_height() / 2.),
                         va='center', fontsize=9, fontweight='bold')
        
    # R2 Plot (Clipped at -0.1 for readable scaling)
    r2_data = results_df.copy()
    r2_data["Val R2 Display"] = np.clip(r2_data["Val R2"], -0.1, 1.0)
    sns.barplot(data=r2_data, y="Model", x="Val R2 Display", order=order, palette=colors, ax=axes[2])
    axes[2].set_title("Validation R² (Higher is Better)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("R² Score (Clipped at -0.1 for visibility)", fontsize=11)
    axes[2].set_ylabel("")
    axes[2].set_yticklabels([])
    for p in axes[2].patches:
        width = p.get_width()
        axes[2].annotate(f"{width:.3f}", (max(width, 0) + 0.02, p.get_y() + p.get_height() / 2.),
                         va='center', fontsize=9, fontweight='bold')

    plt.suptitle("Comparative Performance of Candidate Regression Models on Validation Set", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")

def train_and_validate():
    print("="*80)
    print("STEP E & F: MODEL DEVELOPMENT & VALIDATION SELECTION")
    print("="*80)
    
    # 1. Load data & chronological partition
    df = load_data("Bike_Sharing.csv")
    train_df, val_df, test_df, split_info = chronological_split(df)
    
    # 2. Extract features and target
    X_train, y_train, _ = prepare_features_and_target(train_df)
    X_val, y_val, _ = prepare_features_and_target(val_df)
    
    print(f"Training samples:   {len(X_train)} ({split_info['train']['start_time']} to {split_info['train']['end_time']})")
    print(f"Validation samples: {len(X_val)} ({split_info['val']['start_time']} to {split_info['val']['end_time']})")
    
    # Base tuned models
    lgb_base = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.07,
        max_depth=8,
        num_leaves=35,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    
    hgb_base = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.07,
        max_depth=8,
        min_samples_leaf=15,
        l2_regularization=0.1,
        random_state=42
    )
    
    xgb_base = XGBRegressor(
        n_estimators=200,
        learning_rate=0.07,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    
    voting_ensemble = VotingRegressor([
        ("lgbm", lgb_base),
        ("hgb", hgb_base),
        ("xgb", xgb_base)
    ], weights=[0.40, 0.35, 0.25])
    
    # Candidate model dictionary
    candidate_models = {
        "Dummy (Mean)": (DummyRegressor(strategy="mean"), False, False),
        "Dummy (Median)": (DummyRegressor(strategy="median"), False, False),
        "Linear Regression": (LinearRegression(), True, True),
        "Ridge Regression": (Ridge(alpha=10.0), True, True),
        "Decision Tree": (DecisionTreeRegressor(max_depth=10, min_samples_leaf=10, random_state=42), False, True),
        "Random Forest": (RandomForestRegressor(n_estimators=150, max_depth=18, min_samples_split=5, min_samples_leaf=2, max_features=0.8, random_state=42, n_jobs=-1), False, True),
        "Gradient Boosting": (GradientBoostingRegressor(n_estimators=180, learning_rate=0.07, max_depth=6, random_state=42), False, True),
        "HistGradientBoosting": (hgb_base, False, True),
        "XGBoost": (xgb_base, False, True),
        "LightGBM": (lgb_base, False, True),
        "Voting Ensemble (Ensemble Regressor)": (voting_ensemble, False, True)
    }
    
    results = []
    trained_pipelines = {}
    
    for name, (reg, is_linear, use_target_transform) in candidate_models.items():
        print(f"Training: {name} ...")
        
        if not use_target_transform:
            # Baseline dummy models without target transform
            pipe = Pipeline([
                ("engineer", BikeFeatureEngineering(drop_atemp=True)),
                ("regressor", reg)
            ])
            pipe.fit(X_train, y_train)
            pred_train = np.clip(pipe.predict(X_train), 0, None)
            pred_val = np.clip(pipe.predict(X_val), 0, None)
            model_obj = pipe
        else:
            model_obj = build_model_pipeline(reg, is_linear=is_linear)
            model_obj.fit(X_train, y_train)
            pred_train = np.clip(model_obj.predict(X_train), 0, None)
            pred_val = np.clip(model_obj.predict(X_val), 0, None)
            
        train_mae = mean_absolute_error(y_train, pred_train)
        train_rmse = root_mean_squared_error(y_train, pred_train)
        train_r2 = r2_score(y_train, pred_train)
        
        val_mae = mean_absolute_error(y_val, pred_val)
        val_rmse = root_mean_squared_error(y_val, pred_val)
        val_r2 = r2_score(y_val, pred_val)
        
        gap_mae = val_mae - train_mae
        
        results.append({
            "Model": name,
            "Train MAE": round(train_mae, 2),
            "Train RMSE": round(train_rmse, 2),
            "Train R2": round(train_r2, 4),
            "Val MAE": round(val_mae, 2),
            "Val RMSE": round(val_rmse, 2),
            "Val R2": round(val_r2, 4),
            "Overfitting Gap (MAE)": round(gap_mae, 2)
        })
        
        trained_pipelines[name] = model_obj
        print(f" -> Val MAE: {val_mae:6.2f} | Val RMSE: {val_rmse:6.2f} | Val R2: {val_r2:6.4f} (Train MAE: {train_mae:6.2f})")

    results_df = pd.DataFrame(results)
    
    # Save validation results JSON and CSV
    os.makedirs("models", exist_ok=True)
    results_df.to_csv("models/validation_results.csv", index=False)
    with open("models/validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved models/validation_results.json and models/validation_results.csv")
    
    # Plot model comparison
    plot_model_comparison(results_df, "visualizations/model_comparison.png")
    
    # Select best model based on validation MAE (and RMSE)
    best_candidate = results_df.sort_values(by=["Val MAE", "Val RMSE"]).iloc[0]
    best_model_name = best_candidate["Model"]
    print(f"\n{'*'*70}\nBEST MODEL SELECTED: {best_model_name}\nVal MAE: {best_candidate['Val MAE']} | Val RMSE: {best_candidate['Val RMSE']} | Val R2: {best_candidate['Val R2']}\n{'*'*70}")
    
    # Save best model artifact
    best_model_obj = trained_pipelines[best_model_name]
    best_model_path = "models/best_model.joblib"
    joblib.dump(best_model_obj, best_model_path)
    print(f"Saved best model pipeline to {best_model_path}")
    
    return best_model_name, results_df

if __name__ == "__main__":
    train_and_validate()
