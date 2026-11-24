# Portfolio Project Data — Hourly Bike Rental Demand Prediction Using Ensemble Regression

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/hourly_bike_rental_demand/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image10.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "hourly-bike-rental-demand-prediction-using-ensemble-regression",
  "code": "PRO-19 // ML-04",
  "title": "Hourly Bike Rental Demand Prediction Using Ensemble Regression",
  "category": "machine-learning",
  "categoryLabel": "Supervised Learning",
  "year": "2023",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Hourly-Bike-Rental-Demand-Prediction-Using-Ensemble-Regression",
  "license": "MIT License",
  "banner": "/images/projects/hourly_bike_rental_demand/banner.webp",
  "summary": "End-to-end ensemble regression pipeline predicting hourly bike rental demand with LightGBM, cyclical features, and chronological validation.",
  "projectDescription": "A supervised regression system forecasting hourly Capital Bikeshare rental demand across Washington, D.C. Combining cyclical trigonometric encoding, commuter interaction features, and log-target LightGBM modeling under strict chronological splits, it achieves a 49.17 test MAE.",
  "architectureSummary": "Chronological Split -> Leakage Filtering -> BikeFeatureEngineering -> Log1p TransformedTargetRegressor -> LightGBM Pipeline -> Streamlit Dashboard",
  "techStack": [
    "Python",
    "Scikit-learn",
    "LightGBM",
    "XGBoost",
    "Pandas",
    "NumPy",
    "Matplotlib",
    "Seaborn",
    "Streamlit",
    "Joblib"
  ],
  "telemetryMatrix": {
    "projectType": "Ensemble Regression Pipeline",
    "targetPlatform": "Python & Streamlit Dashboard",
    "primaryDomain": "Urban Micromobility Forecasting",
    "architecturePipeline": "FeatureEng + Log1p + LightGBM",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2023"
  },
  "mlProjectInfo": {
    "projectType": "Supervised Demand Forecasting Pipeline",
    "learningParadigm": "Supervised Learning",
    "modelDomain": "Urban Micromobility Time Series",
    "projectStatus": "Completed",
    "developmentYear": "2023",
    "capabilityBadges": [
      "Cyclical Feature Encoding",
      "Chronological Temporal Split",
      "Interactive Forecast Dashboard"
    ]
  },
  "mlProblemContext": "Bike-sharing operators face continuous fleet imbalances between morning and evening commuter surges and low-demand overnight hours. Forecasting hourly demand from calendar and weather signals enables proactive dock rebalancing and maintenance scheduling while avoiding future-data leakage and collinear thermal predictors.",
  "mlDatasetOverview": {
    "profileBadge": "UCI Tabular Time Series",
    "source": "UCI Machine Learning Repository — Capital Bikeshare Dataset (Washington, D.C.)",
    "modality": "Tabular Time Series",
    "scale": "17,379 Hourly Observations (731 Calendar Days, 2011–2012)",
    "inputShape": "17,379 x 11 Retained Raw Predictors (18 After Feature Engineering)",
    "dataFormat": "CSV",
    "description": "Two years of hourly bike rental records partitioned chronologically without shuffling into 12,165 training hours (70.00%), 2,606 validation hours (15.00%), and 2,608 holdout test hours (15.01%), with zero missing values and 165 unrecorded operational hours left unimputed.",
    "provenanceNote": "Published by Hadi Fanaee-T and João Gama (2013); verified 100% identity consistency (cnt = casual + registered) across all 17,379 CSV records."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "Temporal & Weather Space",
    "spaceNote": "11 retained calendar/weather inputs expanded to 18 engineered features after dropping collinear atemp.",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "cnt (Hourly Total Bike Rentals)",
      "type": "Continuous / Discrete Scalar Count",
      "distribution": "Range 1–977 bikes/hour (Mean: 189.46, Median: 142, Skewness: 1.28)"
    },
    "inputSignals": [
      {
        "name": "hr & Cyclical Encodings (hr_sin, hr_cos)",
        "type": "Numerical / Cyclical",
        "importanceOrRole": "Top Driver (>130 MAE Permutation Loss)",
        "description": "Hour of day (0–23) and trigonometric sine/cosine pairs capturing continuous diurnal transitions across midnight."
      },
      {
        "name": "temp & temp_hum_ratio",
        "type": "Numerical",
        "importanceOrRole": "Primary Thermal Signal (r = 0.40 with cnt)",
        "description": "Normalized ambient Celsius temperature retained after dropping atemp (r = 0.9877), paired with thermal comfort ratio."
      },
      {
        "name": "workingday & is_rush_hour",
        "type": "Binary Indicator",
        "importanceOrRole": "Regime Separator",
        "description": "Distinguishes bimodal weekday commuter peaks (08:00, 09:00, 17:00, 18:00) from unimodal weekend leisure riding."
      },
      {
        "name": "yr, season, mnth & weekday",
        "type": "Temporal / Cyclical",
        "importanceOrRole": "Growth & Seasonality",
        "description": "Captures multi-year ridership expansion (2011 to 2012) alongside cyclic monthly and weekly demand oscillations."
      },
      {
        "name": "weathersit, hum & windspeed",
        "type": "Categorical / Numerical",
        "importanceOrRole": "Adverse Weather Penalty",
        "description": "Encodes clear-to-storm weather categories, normalized relative humidity (r = -0.32), and wind speed."
      }
    ],
    "dataCharacteristicsSummary": "Hourly demand exhibits strong right-skewness (1.28), bimodal weekday commuter peaks at 08:00 and 17:00, and unimodal weekend leisure peaks around midday. Ambient and feels-like temperature show 0.9877 Pearson correlation, while year-over-year ridership nearly doubles from 2011 to 2012."
  },
  "mlLearningApproach": {
    "paradigm": "Supervised Learning — Tabular Ensemble Regression",
    "methodologyDescription": "Eleven models spanning Dummy baselines, linear models, single trees, bagging, gradient boosting variants, and a weighted Voting Regressor were trained on log-transformed targets. Every candidate shared identical BikeFeatureEngineering pipelines and strict chronological train-validation-test splits.",
    "justification": "Histogram-based gradient boosting with log1p target transformation handles nonlinear hour-weather interactions and skewed count distributions far better than linear models, which collapsed to negative R² under 2012 ridership growth."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Target Leakage Exclusion",
      "description": "Strictly removed casual and registered target components (cnt = casual + registered), sequential instant index, and raw date strings from feature matrix X.",
      "badge": "Leakage Guard"
    },
    {
      "step": "02",
      "title": "Chronological Partitioning",
      "description": "Sorted records monotonically by timestamp and split 70% Train (12,165), 15% Validation (2,606), and 15% Test (2,608) without random shuffling.",
      "badge": "Temporal Split"
    },
    {
      "step": "03",
      "title": "Multicollinearity & Feature Engineering",
      "description": "Dropped collinear atemp (r = 0.98767 with temp) and generated 6 cyclical trig features, is_rush_hour, and temp_hum_ratio inside BikeFeatureEngineering.",
      "badge": "Domain Transform"
    },
    {
      "step": "04",
      "title": "Log1p Target Transformation",
      "description": "Wrapped pipelines in TransformedTargetRegressor using log1p training targets (reducing skewness from 1.28 to -0.82) and non-negative expm1 inverse clipping.",
      "badge": "Skew Normalization"
    }
  ],
  "mlDataPreparationNote": "All feature transformations and linear scalers are encapsulated inside Scikit-learn Pipeline objects fit exclusively on the Training partition to eliminate temporal and statistical data leakage.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "LGBMRegressor (TransformedTargetRegressor Pipeline)",
    "architectureSummary": "Histogram-based leaf-wise gradient boosted decision tree ensemble of 200 trees integrated directly after the deterministic BikeFeatureEngineering transformer and wrapped in log1p target scaling.",
    "configurationBadge": "Deterministic Seed 42",
    "hyperparameters": [
      {
        "name": "n_estimators",
        "value": "200",
        "description": "Total sequential boosting trees fitted on log1p residuals."
      },
      {
        "name": "learning_rate",
        "value": "0.07",
        "description": "Shrinkage factor balancing convergence speed and generalization."
      },
      {
        "name": "max_depth / num_leaves",
        "value": "8 / 35",
        "description": "Limits tree depth and leaf count to prevent over-fragmentation."
      },
      {
        "name": "subsample / colsample_bytree",
        "value": "0.8 / 0.8",
        "description": "Stochastic row and feature subsampling ratios per boosting round."
      },
      {
        "name": "reg_alpha / reg_lambda",
        "value": "0.1 / 1.0",
        "description": "L1 and L2 leaf weight regularization terms."
      }
    ],
    "objectiveCriterion": "Mean Squared Error on log1p(cnt) Target",
    "optimizationSolver": "Leaf-Wise Histogram Gradient Boosting (LightGBM)",
    "trainingProcess": "Models were fit exclusively on 12,165 chronological training hours (Jan 2011 to May 2012) using deterministic seed 42 and log1p target transformation. Validation across 2,606 subsequent hours selected LightGBM over Voting Ensemble and XGBoost based on lowest MAE and minimal overfitting gap."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "Chronological Holdout Validation & Single-Pass Test Evaluation",
    "description": "Candidate selection occurred solely on the May–September 2012 validation window using MAE, RMSE, R², and train-validation MAE gap. The winning LightGBM pipeline was evaluated once on the isolated September–December 2012 test window alongside operational slices.",
    "metrics": [
      {
        "label": "TEST MAE",
        "value": "49.17",
        "benchmark": "Baseline < 169.03"
      },
      {
        "label": "TEST RMSE",
        "value": "73.81",
        "benchmark": "Baseline < 228.42"
      },
      {
        "label": "TEST R2",
        "value": "0.8823",
        "benchmark": "Target > 0.85"
      },
      {
        "label": "VAL MAE",
        "value": "39.70",
        "benchmark": "Candidate < 40.12"
      },
      {
        "label": "TEST MBE",
        "value": "-31.04",
        "benchmark": "Baseline: -75.85"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Streamlit Dashboard — Forecast & Performance Overview",
      "badge": "Inference UI",
      "imageUrl": "/images/projects/hourly_bike_rental_demand/image01.webp",
      "telemetryNote": "Displays 734 bikes/hr Surge Demand prediction alongside Test MAE 49.17 (-70.9% vs Baseline) and Test R² 0.8823."
    },
    {
      "title": "Streamlit Dashboard — Normalized Input Feature Breakdown",
      "badge": "Feature Inspector",
      "imageUrl": "/images/projects/hourly_bike_rental_demand/image02.webp",
      "telemetryNote": "Verifies real-time UCI normalization of temperature, humidity, and windspeed inputs without leakage columns."
    },
    {
      "title": "Streamlit Dashboard — System Architecture & Compliance",
      "badge": "System Audit",
      "imageUrl": "/images/projects/hourly_bike_rental_demand/image03.webp",
      "telemetryNote": "Documents LightGBM core, casual/registered exclusion, non-negative log1p clipping, and seed 42 determinism."
    },
    {
      "title": "Test Actual vs. Predicted Tracking & Calendar Patterns",
      "badge": "Regression Fit",
      "imageUrl": "/images/projects/hourly_bike_rental_demand/image04.webp",
      "telemetryNote": "Shows tight 1:1 scatter clustering (R² = 0.8823) and 14-day hourly tracking paired with seasonal demand bars."
    },
    {
      "title": "In-Depth Residual Diagnostics & Error Patterns",
      "badge": "Error Analysis",
      "imageUrl": "/images/projects/hourly_bike_rental_demand/image05.webp",
      "telemetryNote": "Residual distribution (Std 66.98, Skew 0.31) highlighting widened rush-hour variance and positive mean residual (31.04)."
    },
    {
      "title": "Impact of Weather Conditions on Hourly Rental Demand",
      "badge": "Weather EDA",
      "imageUrl": "/images/projects/hourly_bike_rental_demand/image06.webp",
      "telemetryNote": "Demonstrates demand drop across severe weather categories and opposing linear slopes for temperature vs. humidity."
    },
    {
      "title": "Hourly Diurnal Dynamics & Log-Target Normalization",
      "badge": "Diurnal & Target EDA",
      "imageUrl": "/images/projects/hourly_bike_rental_demand/image07.webp",
      "telemetryNote": "Contrasts bimodal commuter peaks (08:00, 17:00) vs. weekend midday peaks and raw skewness (1.28) vs. log1p (-0.82)."
    },
    {
      "title": "Two-Year Weekly Demand Trend & Chronological Splits",
      "badge": "Temporal Drift",
      "imageUrl": "/images/projects/hourly_bike_rental_demand/image08.webp",
      "telemetryNote": "Visualizes 2011–2012 ridership growth across exact 70% Train, 15% Validation, and 15% Test cutoffs."
    },
    {
      "title": "Feature Importance & 11-Model Validation Benchmark",
      "badge": "Explainability & Leaderboard",
      "imageUrl": "/images/projects/hourly_bike_rental_demand/image09.webp",
      "telemetryNote": "Ranks hr, temp, workingday, and temp_hum_ratio as top drivers while LightGBM leads validation (MAE 39.7, R² 0.923)."
    },
    {
      "title": "Pearson Correlation Matrix of Numeric & Encoded Features",
      "badge": "Collinearity Audit",
      "imageUrl": "/images/projects/hourly_bike_rental_demand/image10.webp",
      "telemetryNote": "Confirms r = 0.99 multicollinearity between temp and atemp alongside cnt associations with temp (0.40) and hum (-0.32)."
    }
  ],
  "mlVisualizationTelemetryNote": "Validation and holdout test diagnostics confirm consistent ranking across gradient boosting models, with non-rush test MAE at 41.08 bikes/hr and higher absolute error concentrated in extreme commuter surges.",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Permutation importance and split-gain diagnostics confirm hour-of-day, ambient temperature, workingday status, and temp-humidity ratio dominate predictions. Residual analysis shows accurate non-rush tracking (41.08 MAE) with moderate underprediction (-31.04 MBE) during late-2012 ridership growth.",
    "keyDrivers": [
      {
        "factor": "Hour of Day (hr & hr_sin / hr_cos)",
        "impact": "Shuffling hr increases test MAE by >130 bikes/hr; drives bimodal commuter vs. leisure diurnal curves."
      },
      {
        "factor": "Ambient Temperature & Thermal Ratio",
        "impact": "Second strongest driver (+28 MAE permutation impact); temp_hum_ratio ranks #2 in LightGBM split gain."
      },
      {
        "factor": "Working Day & Rush Hour Flags",
        "impact": "Separates weekday peak surges (08:00 and 17:00–18:00) from midday weekend demand patterns."
      },
      {
        "factor": "Relative Humidity & Weather Situation",
        "impact": "Applies downward demand correction as humidity rises or precipitation occurs (MAE rises to 66.18 in rain/snow)."
      }
    ],
    "limitations": [
      {
        "title": "Underprediction Under Growth Concept Drift",
        "description": "73.47% of test predictions underestimate actual rentals (MBE -31.04) because late-2012 membership exceeded 2011–early-2012 training levels."
      },
      {
        "title": "Rare Severe Weather Sample Scarcity",
        "description": "Extreme weather category 4 (heavy rain/snow) appears only 3 times across all 17,379 hours, limiting variance reliability in storms."
      },
      {
        "title": "City-Wide Spatial Aggregation",
        "description": "Forecasts aggregate system demand rather than station-level dock depletion or localized neighborhood surpluses."
      }
    ],
    "futureImprovements": [
      {
        "title": "Sliding-Window Periodic Retraining",
        "target": "Mitigate -31.04 MBE concept drift via monthly or quarterly rolling-window updates."
      },
      {
        "title": "Causal Autoregressive Lag Features",
        "target": "Reduce rush-hour MAE (113.35) using strictly past t-24h and t-168h demand lags in live serving."
      },
      {
        "title": "Station-Level Hierarchical Forecasting",
        "target": "Disaggregate city-wide predictions to individual dock hubs for targeted truck rebalancing."
      }
    ],
    "discoveryNote": "Permutation importance reflects predictive reliance under temporal split conditions rather than causal meteorological effect."
  },
  "mlModelArtifacts": [
    {
      "name": "models/best_model.joblib",
      "format": "JOBLIB"
    },
    {
      "name": "models/validation_results.json",
      "format": "JSON"
    },
    {
      "name": "models/validation_results.csv",
      "format": "CSV"
    },
    {
      "name": "models/test_evaluation_results.json",
      "format": "JSON"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Supervised Learning",
    "modelArchitecture": "LightGBM Gradient Boosting",
    "coreFramework": "Scikit-learn and LightGBM",
    "validationMethod": "Chronological Holdout Split"
  }
}
```
