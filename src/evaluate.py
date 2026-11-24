"""
Final Test Evaluation and Model Interpretation Module
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

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

from src.data_preparation import load_data, chronological_split, prepare_features_and_target, BikeFeatureEngineering, inverse_nonnegative

def evaluate_model():
    print("="*80)
    print("STEP F & G: FINAL TEST EVALUATION & MODEL INTERPRETATION")
    print("="*80)
    
    # 1. Load data & chronological partition
    df = load_data("Bike_Sharing.csv")
    train_df, val_df, test_df, split_info = chronological_split(df)
    
    # Extract Test set
    X_test, y_test, ts_test = prepare_features_and_target(test_df)
    
    print(f"Test samples: {len(X_test)} ({split_info['test']['start_time']} to {split_info['test']['end_time']})")
    print(f"Test Target 'cnt' stats: Mean={y_test.mean():.2f}, Std={y_test.std():.2f}, Min={y_test.min()}, Max={y_test.max()}")
    
    # 2. Load best model
    best_model_path = "models/best_model.joblib"
    assert os.path.exists(best_model_path), f"Model file not found: {best_model_path}"
    model = joblib.load(best_model_path)
    print(f"Loaded best model from {best_model_path}")
    
    # 3. Predict on Test set
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)
    
    # Also evaluate Baseline Dummy (Mean of Train) on Test set
    train_mean = split_info["train"]["target_mean"]
    dummy_pred = np.full_like(y_test, train_mean)
    
    # Overall Test Metrics
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = root_mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    dummy_mae = mean_absolute_error(y_test, dummy_pred)
    dummy_rmse = root_mean_squared_error(y_test, dummy_pred)
    dummy_r2 = r2_score(y_test, dummy_pred)
    
    # Bias / Directional Error Analysis
    residuals = y_test - y_pred  # positive means underpredicted (actual > predicted), negative means overpredicted
    mean_bias_error = np.mean(y_pred - y_test)
    underpredict_pct = (residuals > 0).mean() * 100
    overpredict_pct = (residuals < 0).mean() * 100
    exact_pct = (residuals == 0).mean() * 100
    
    print("\n--- OVERALL TEST METRICS ---")
    print(f"Best Model Test MAE:   {test_mae:.2f} bikes/hr (vs Dummy: {dummy_mae:.2f}, {(dummy_mae-test_mae)/dummy_mae*100:.1f}% reduction)")
    print(f"Best Model Test RMSE:  {test_rmse:.2f} bikes/hr (vs Dummy: {dummy_rmse:.2f}, {(dummy_rmse-test_rmse)/dummy_rmse*100:.1f}% reduction)")
    print(f"Best Model Test R²:    {test_r2:.4f} (vs Dummy: {dummy_r2:.4f})")
    print(f"Mean Bias Error (MBE): {mean_bias_error:.2f} bikes/hr")
    print(f"Underprediction rate:  {underpredict_pct:.1f}% of hours (Actual > Predicted)")
    print(f"Overprediction rate:   {overpredict_pct:.1f}% of hours (Actual < Predicted)")

    # 4. Sliced Performance Breakdown
    eval_df = test_df.copy()
    eval_df["y_true"] = y_test.values
    eval_df["y_pred"] = y_pred
    eval_df["abs_err"] = np.abs(eval_df["y_true"] - eval_df["y_pred"])
    eval_df["residual"] = eval_df["y_true"] - eval_df["y_pred"]
    
    # Slice by Working Day vs Weekend/Holiday
    slice_workingday = eval_df.groupby("workingday").agg(
        count=("cnt", "count"),
        mae=("abs_err", "mean"),
        rmse=("residual", lambda r: np.sqrt(np.mean(r**2)))
    ).reset_index()
    
    # Slice by Weather
    slice_weather = eval_df.groupby("weathersit").agg(
        count=("cnt", "count"),
        mae=("abs_err", "mean"),
        rmse=("residual", lambda r: np.sqrt(np.mean(r**2)))
    ).reset_index()
    
    # Slice by Peak Hours (Rush vs Non-Rush)
    eval_df["is_rush"] = ((eval_df["hr"].isin([8, 9, 17, 18])) & (eval_df["workingday"] == 1)).astype(int)
    slice_rush = eval_df.groupby("is_rush").agg(
        count=("cnt", "count"),
        mae=("abs_err", "mean"),
        rmse=("residual", lambda r: np.sqrt(np.mean(r**2)))
    ).reset_index()

    # Save Test Results JSON
    test_results_data = {
        "test_metrics": {
            "MAE": round(test_mae, 2),
            "RMSE": round(test_rmse, 2),
            "R2": round(test_r2, 4),
            "MBE": round(mean_bias_error, 2),
            "underpredict_pct": round(underpredict_pct, 2),
            "overpredict_pct": round(overpredict_pct, 2)
        },
        "dummy_baseline_metrics": {
            "MAE": round(dummy_mae, 2),
            "RMSE": round(dummy_rmse, 2),
            "R2": round(dummy_r2, 4)
        },
        "slice_workingday": slice_workingday.to_dict(orient="records"),
        "slice_weather": slice_weather.to_dict(orient="records"),
        "slice_rush": slice_rush.to_dict(orient="records")
    }
    with open("models/test_evaluation_results.json", "w") as f:
        json.dump(test_results_data, f, indent=2)
    print("\nSaved models/test_evaluation_results.json")

    # 5. Visualizations
    sns.set_theme(style="whitegrid", font="sans-serif")
    os.makedirs("visualizations", exist_ok=True)
    
    # 5A. Actual vs Predicted Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    # Scatter plot
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.35, color="#1f77b4", edgecolor=None, s=25, ax=axes[0])
    max_val = max(y_test.max(), y_pred.max()) + 20
    axes[0].plot([0, max_val], [0, max_val], color="crimson", linestyle="--", linewidth=2, label="Ideal 1:1 Reference Line")
    axes[0].set_title(f"Actual vs Predicted Hourly Rental Demand on Test Set\n(R² = {test_r2:.4f}, MAE = {test_mae:.2f})", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Actual Rental Demand ('cnt')", fontsize=11)
    axes[0].set_ylabel("Predicted Rental Demand (bikes/hour)", fontsize=11)
    axes[0].set_xlim(0, max_val)
    axes[0].set_ylim(0, max_val)
    axes[0].legend(frameon=True)
    
    # Continuous Time Series Slice (First 14 days of Test set)
    sample_slice = eval_df.iloc[:24*14].copy()
    axes[1].plot(sample_slice["timestamp"], sample_slice["y_true"], label="Actual Demand", color="#2c3e50", linewidth=1.8, marker="o", markersize=3)
    axes[1].plot(sample_slice["timestamp"], sample_slice["y_pred"], label="Predicted Demand", color="#e74c3c", linewidth=1.8, linestyle="--", marker="s", markersize=3)
    axes[1].set_title("14-Day Hourly Time-Series Tracking on Test Partition", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Date & Time", fontsize=11)
    axes[1].set_ylabel("Hourly Rental Count", fontsize=11)
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].legend(frameon=True)
    
    plt.suptitle("Test Set Performance: Actual vs Predicted Rental Demand", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig("visualizations/actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved visualizations/actual_vs_predicted.png")

    # 5B. Residual Analysis Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
    
    # Residual vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.3, color="#2b5c8f", s=20)
    axes[0, 0].axhline(0, color="crimson", linestyle="--", linewidth=1.8)
    axes[0, 0].set_title("Residuals vs Predicted Values", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Predicted Demand (bikes/hour)", fontsize=11)
    axes[0, 0].set_ylabel("Residual (Actual - Predicted)", fontsize=11)
    
    # Residual Distribution
    sns.histplot(residuals, kde=True, ax=axes[0, 1], color="#17a2b8", bins=40)
    axes[0, 1].axvline(0, color="red", linestyle="--", linewidth=1.5, label=f"Mean Residual: {residuals.mean():.2f}")
    axes[0, 1].set_title(f"Residual Distribution (Std: {residuals.std():.2f}, Skew: {residuals.skew():.2f})", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Residual (bikes/hour)", fontsize=11)
    axes[0, 1].set_ylabel("Frequency", fontsize=11)
    axes[0, 1].legend(frameon=True)
    
    # Residual by Hour of Day
    sns.boxplot(data=eval_df, x="hr", y="residual", ax=axes[1, 0], palette="coolwarm")
    axes[1, 0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_title("Residual Distribution Across Hours of Day (0–23)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Hour of Day", fontsize=11)
    axes[1, 0].set_ylabel("Residual (Actual - Predicted)", fontsize=11)
    axes[1, 0].set_xticks(range(0, 24, 2))
    
    # Residual by Weather Situation
    weather_labels = {1: "Clear / Few Clouds", 2: "Mist / Cloudy", 3: "Light Rain / Snow"}
    eval_df["weather_label"] = eval_df["weathersit"].map(weather_labels)
    sns.boxplot(data=eval_df, x="weather_label", y="residual", ax=axes[1, 1], palette="Set2")
    axes[1, 1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_title("Residual Distribution Across Weather Conditions", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Weather Condition", fontsize=11)
    axes[1, 1].set_ylabel("Residual (Actual - Predicted)", fontsize=11)

    plt.suptitle("In-Depth Residual Diagnostics and Error Patterns", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig("visualizations/residual_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved visualizations/residual_analysis.png")

    # 5C. Feature Importance Plot
    print("\nComputing Permutation Importance on Test Set (10 repeats)...")
    perm_importance = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=42,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    
    sorted_idx = perm_importance.importances_mean.argsort()[::-1]
    feature_names = np.array(X_test.columns)[sorted_idx]
    importance_means = perm_importance.importances_mean[sorted_idx]
    importance_stds = perm_importance.importances_std[sorted_idx]
    
    # Extract underlying tree feature importances
    pipeline_regressor = model.regressor_
    underlying_reg = pipeline_regressor.named_steps["regressor"]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    # Permutation Importance (Test Set)
    axes[0].barh(range(len(feature_names)), importance_means, xerr=importance_stds, color="#1f77b4", capsize=3, align="center")
    axes[0].set_yticks(range(len(feature_names)))
    axes[0].set_yticklabels(feature_names, fontsize=10)
    axes[0].invert_yaxis()
    axes[0].set_title("Permutation Importance on Test Set\n(Increase in MAE when feature is permuted)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Mean MAE Loss (+ bikes/hour)", fontsize=11)
    
    # Internal Feature Importance
    if hasattr(underlying_reg, "feature_importances_"):
        # Transformed feature names after FeatureEngineering
        feat_eng = pipeline_regressor.named_steps["engineer"]
        X_sample_trans = feat_eng.transform(X_test.iloc[:5])
        trans_cols = list(X_sample_trans.columns)
        tree_importances = underlying_reg.feature_importances_
        
        tree_df = pd.DataFrame({"Feature": trans_cols, "Importance": tree_importances}).sort_values("Importance", ascending=False)
        sns.barplot(data=tree_df.head(15), x="Importance", y="Feature", palette="viridis", ax=axes[1])
        axes[1].set_title("Underlying Model Feature Importance (Split Gain / Gini)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Importance Weight", fontsize=11)
    else:
        axes[1].text(0.5, 0.5, "Ensemble Voting Regressor uses multiple estimators.\nSee Permutation Importance on left.",
                     ha="center", va="center", fontsize=12)
        axes[1].set_axis_off()

    plt.suptitle("Feature Importance and Model Interpretability Analysis", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig("visualizations/feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved visualizations/feature_importance.png")

    # 6. Test Single Inference Reload
    print("\n--- INFERENCE RELOAD VALIDATION ---")
    sample_input = pd.DataFrame([{
        "season": 3,
        "yr": 1,
        "mnth": 8,
        "hr": 17,
        "holiday": 0,
        "weekday": 3,
        "workingday": 1,
        "weathersit": 1,
        "temp": 0.70,
        "atemp": 0.65,
        "hum": 0.50,
        "windspeed": 0.15
    }])
    sample_pred = model.predict(sample_input)[0]
    print(f"Sample Input (Summer Thursday 5 PM, Clear, 25°C):")
    print(f"Predicted Demand: {sample_pred:.1f} bikes/hour")

if __name__ == "__main__":
    evaluate_model()
