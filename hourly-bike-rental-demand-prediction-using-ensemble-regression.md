# Portfolio Project Data — Hourly Bike Rental Demand Prediction Using Ensemble Regression

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/hourly_bike_rental_demand/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image10.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "hourly-bike-rental-demand-prediction-using-ensemble-regression",
  "code": "PRO-21 // ML-04",
  "title": "Hourly Bike Rental Demand Prediction Using Ensemble Regression",
  "category": "machine-learning",
  "categoryLabel": "Supervised Learning",
  "year": "2024",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Hourly-Bike-Rental-Demand-Prediction-Using-Ensemble-Regression",
  "license": "MIT License",
  "banner": "/images/projects/hourly_bike_rental_demand/banner.webp",
  "summary": "Leak-free LightGBM regression pipeline forecasting hourly bike demand with chronological validation and a Streamlit inference dashboard.",
  "projectDescription": "End-to-end supervised regression system predicting hourly Capital Bikeshare rentals across 17,379 records. Combines leakage prevention, cyclical time encoding, multicollinearity removal, and strict chronological validation across 11 models, deploying a tuned LightGBM pipeline via Streamlit.",
  "architectureSummary": "Chronological Data Split -> Cyclical & Rush-Hour Feature Engineering -> Log1p TransformedTargetRegressor -> Tuned LightGBM Pipeline -> Joblib & Streamlit Inference",
  "techStack": [
    "Python",
    "LightGBM",
    "Scikit-Learn",
    "XGBoost",
    "Pandas",
    "NumPy",
    "Streamlit",
    "Joblib",
    "Matplotlib",
    "Seaborn"
  ],
  "telemetryMatrix": {
    "projectType": "Supervised Ensemble Regression Pipeline",
    "targetPlatform": "Python CLI & Streamlit Web Dashboard",
    "primaryDomain": "Urban Micromobility Demand Forecasting",
    "architecturePipeline": "Cyclical Features + Log1p LightGBM Pipeline",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2024"
  },
  "mlProjectInfo": {
    "projectType": "Predictive Tabular Regression Pipeline",
    "learningParadigm": "Supervised Learning",
    "modelDomain": "Urban Micromobility Demand",
    "projectStatus": "Completed",
    "developmentYear": "2024",
    "capabilityBadges": [
      "Chronological Validation",
      "Cyclical Time Encoding",
      "Leakage Prevention"
    ]
  },
  "mlProblemContext": "Urban bike-sharing operators require accurate hourly demand forecasts to balance dock inventory, deploy rebalancing trucks before commuter surges, and schedule maintenance overnight. Because ridership shifts non-linearly across rush hours, weather, and multi-year fleet growth, linear models fail under temporal holdouts.",
  "mlDatasetOverview": {
    "profileBadge": "UCI Capital Bikeshare",
    "source": "UCI Machine Learning Repository (Bike Sharing Dataset, Capital Bikeshare Washington, D.C.)",
    "modality": "Tabular Time-Indexed Observational Data",
    "scale": "17,379 Hourly Observations (731 Calendar Days, 2011–2012)",
    "inputShape": "17,379 x 11 Core Predictors (18 Engineered Features)",
    "dataFormat": "CSV",
    "description": "Two years of hourly Capital Bikeshare rental records from 2011-01-01 to 2012-12-31 with zero missing values or duplicates, partitioned chronologically into 12,165 train, 2,606 validation, and 2,608 test hours without shuffling.",
    "provenanceNote": "Published by Hadi Fanaee-T and Joao Gama (2013), UCI ID 275; 165 unrecorded operational hours are preserved without synthetic imputation."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "Temporal & Weather Space",
    "spaceNote": "11 raw predictors transformed into 18 cyclical and domain-interaction features after dropping casual, registered, instant, and collinear atemp.",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "cnt (Total Hourly Rental Count)",
      "type": "Continuous Non-Negative Scalar",
      "distribution": "Range 1–977 bikes/hour; Mean 189.46, Median 142, Skewness 1.28"
    },
    "inputSignals": [
      {
        "name": "hr (and hr_sin / hr_cos)",
        "type": "Cyclical Temporal",
        "importanceOrRole": "Dominant Predictor (+130+ Test MAE on permutation)",
        "description": "Hour of day (0–23) encoded as sine/cosine harmonics to capture bimodal weekday commuter surges and midnight troughs."
      },
      {
        "name": "temp & temp_hum_ratio",
        "type": "Numerical Meteorological",
        "importanceOrRole": "Top Weather Driver (Pearson r = 0.40 with cnt)",
        "description": "Normalized ambient Celsius temperature and temperature-to-humidity comfort ratio after removing collinear atemp (r = 0.9877)."
      },
      {
        "name": "workingday & is_rush_hour",
        "type": "Binary Interaction",
        "importanceOrRole": "Regime Separator (Weekday vs. Leisure)",
        "description": "Distinguishes bimodal weekday commuter curves (8–9 AM, 5–6 PM peaks) from unimodal weekend leisure peaks."
      },
      {
        "name": "yr, mnth, weekday, season",
        "type": "Calendar & Structural Trend",
        "importanceOrRole": "Multi-Year Growth & Seasonality",
        "description": "Captures 2011-to-2012 ridership expansion (~130 to ~240 mean bikes/hr) and seasonal cycles via cyclical sin/cos encoding."
      },
      {
        "name": "hum & weathersit",
        "type": "Meteorological Penalty",
        "importanceOrRole": "Weather Attenuation Signal",
        "description": "Relative humidity (r = -0.32) and four-tier weather severity rating governing sharp drops during rain or snow."
      }
    ],
    "dataCharacteristicsSummary": "Hourly demand exhibits strong right-skewness (1.28) and bimodal weekday spikes at 08:00 and 17:00–18:00 versus unimodal weekend midday peaks. Temperature correlates positively (0.40) while humidity correlates negatively (-0.32), with 0.9877 collinearity between temp and atemp."
  },
  "mlLearningApproach": {
    "paradigm": "Supervised Tabular Regression under Strict Chronological Holdout",
    "methodologyDescription": "Formulates hourly forecasting as supervised tabular regression across 11 candidates spanning naive Dummy baselines, OLS/Ridge linear models, bagged Decision Trees, Gradient Boosting, XGBoost, LightGBM, and a weighted VotingRegressor, evaluated strictly on chronological future windows.",
    "justification": "Non-parametric gradient boosted decision trees paired with a log1p target transformation capture non-linear hour-by-weather interactions and year-over-year volume scaling, whereas linear models yield negative R2 (-0.56) under temporal distribution shift."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Leakage & Chronology Audit",
      "description": "Verified cnt == casual + registered across all 17,379 rows, confirmed strictly monotonic timestamps, and excluded casual, registered, instant, and dteday from feature matrices.",
      "badge": "Anti-Leakage"
    },
    {
      "step": "02",
      "title": "Chronological 70/15/15 Partitioning",
      "description": "Split records sequentially without shuffling into Train (12,165 rows), Validation (2,606 rows), and Test (2,608 rows) partitions to mirror real-world forecasting.",
      "badge": "Temporal Split"
    },
    {
      "step": "03",
      "title": "Cyclical & Domain Feature Engineering",
      "description": "Computed sine/cosine harmonics for hr, mnth, and weekday, derived is_rush_hour and temp_hum_ratio, and dropped collinear atemp inside BikeFeatureEngineering.",
      "badge": "Feature Pipeline"
    },
    {
      "step": "04",
      "title": "Log1p Target Transformation",
      "description": "Applied np.log1p target scaling via TransformedTargetRegressor (reducing skewness from 1.28 to -0.82) with non-negative expm1 clipping on inverse predictions.",
      "badge": "Target Scaling"
    }
  ],
  "mlDataPreparationNote": "All feature engineering and target transformations reside inside Scikit-learn Pipeline and TransformedTargetRegressor objects fitted exclusively on the training partition to prevent future data leakage.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "LightGBM Regressor (TransformedTargetRegressor Pipeline)",
    "architectureSummary": "200 leaf-wise gradient boosted decision trees (max_depth=8, num_leaves=35) embedded inside a Scikit-learn Pipeline with BikeFeatureEngineering and log1p/expm1 target wrapping, outperforming 10 peer models including a weighted Voting Ensemble.",
    "configurationBadge": "200 Trees / Log1p",
    "hyperparameters": [
      {
        "name": "n_estimators",
        "value": "200",
        "description": "Sequential boosting iterations balancing predictive capacity and fast sub-5 ms inference."
      },
      {
        "name": "learning_rate",
        "value": "0.07",
        "description": "Conservative shrinkage step size paired with 200 estimators."
      },
      {
        "name": "max_depth / num_leaves",
        "value": "8 / 35",
        "description": "Bounds tree interaction depth and leaf complexity to control variance."
      },
      {
        "name": "subsample / colsample_bytree",
        "value": "0.8 / 0.8",
        "description": "Stochastic row and feature subsampling ratios per boosting iteration."
      },
      {
        "name": "reg_alpha / reg_lambda",
        "value": "0.1 / 1.0",
        "description": "L1 and L2 leaf-weight regularization penalties preventing extreme leaf values."
      },
      {
        "name": "random_state",
        "value": "42",
        "description": "Fixed deterministic seed across candidate training and permutation importance."
      }
    ],
    "objectiveCriterion": "Mean Squared Error on log1p-Transformed Target (L2 Loss)",
    "optimizationSolver": "Leaf-Wise Gradient Boosting (LightGBM Histogram Tree Builder)",
    "trainingProcess": "All 11 candidate models were fitted strictly on the 12,165-row training set (Jan 2011–May 2012) with random_state=42. Candidates were ranked by MAE and RMSE on the 2,606-row chronological validation set (May–Sep 2012), selecting LightGBM with the lowest validation MAE (39.70) and smallest overfitting gap (+20.25)."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "Chronological Validation & Single-Pass Out-of-Sample Test Holdout",
    "description": "Model selection relied exclusively on the 15% validation partition (2,606 rows), followed by a single final evaluation on the isolated 15% test partition (2,608 rows from Sep–Dec 2012) with sliced MAE/RMSE audits across working days, weather, and rush hours.",
    "metrics": [
      {
        "label": "TEST MAE",
        "value": "49.17",
        "benchmark": "Baseline: 169.03"
      },
      {
        "label": "TEST RMSE",
        "value": "73.81",
        "benchmark": "Baseline: 228.42"
      },
      {
        "label": "TEST R2",
        "value": "0.8823",
        "benchmark": "Target > 0.85"
      },
      {
        "label": "VAL MAE",
        "value": "39.70",
        "benchmark": "Ensemble: 40.12"
      },
      {
        "label": "VAL R2",
        "value": "0.9232",
        "benchmark": "Baseline: -0.2939"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Streamlit Dashboard — Validation & Test Performance",
      "badge": "Inference UI",
      "imageUrl": "/images/projects/hourly_bike_rental_demand/image01.webp",
      "telemetryNote": "Displays 734 bikes/hr surge forecast and benchmark KPIs (Test MAE 49.17, R2 0.8823)."
    },
    {
      "title": "Streamlit Dashboard — Input Feature Breakdown",
      "badge": "Feature Inspector",
      "imageUrl": "/images/projects/hourly_bike_rental_demand/image02.webp",
      "telemetryNote": "Shows live UCI-normalized weather/calendar vector alongside physical units."
    },
    {
      "title": "Streamlit Dashboard — System & Reproducibility",
      "badge": "Architecture Tab",
      "imageUrl": "/images/projects/hourly_bike_rental_demand/image03.webp",
      "telemetryNote": "Documents leakage removal, non-negative clipping, and deterministic seed=42."
    },
    {
      "title": "Actual vs Predicted & Calendar Demand Patterns",
      "badge": "Regression Fit",
      "imageUrl": "/images/projects/hourly_bike_rental_demand/image04.webp",
      "telemetryNote": "Test scatter (R2 0.8823) and 14-day hourly trace alongside seasonal/monthly averages."
    },
    {
      "title": "In-Depth Residual Diagnostics & Error Patterns",
      "badge": "Residual Analysis",
      "imageUrl": "/images/projects/hourly_bike_rental_demand/image05.webp",
      "telemetryNote": "Residual histogram (Std 66.98, Skew 0.31) and hourly/weather error spreads."
    },
    {
      "title": "Weather Impact on Hourly Rental Demand",
      "badge": "EDA Weather",
      "imageUrl": "/images/projects/hourly_bike_rental_demand/image06.webp",
      "telemetryNote": "Contrasts clear vs. rain/snow demand and temperature/humidity regression trends."
    },
    {
      "title": "Hourly Diurnal Dynamics & Target Distribution",
      "badge": "EDA Target & Hour",
      "imageUrl": "/images/projects/hourly_bike_rental_demand/image07.webp",
      "telemetryNote": "Illustrates bimodal rush-hour curves and skewness reduction via log1p (1.28 to -0.82)."
    },
    {
      "title": "Two-Year Demand Trend & Chronological Splits",
      "badge": "Temporal Trend",
      "imageUrl": "/images/projects/hourly_bike_rental_demand/image08.webp",
      "telemetryNote": "Maps 2011–2012 growth across 70% Train, 15% Validation, and 15% Test windows."
    },
    {
      "title": "Feature Importance & Candidate Model Comparison",
      "badge": "Model Benchmark",
      "imageUrl": "/images/projects/hourly_bike_rental_demand/image09.webp",
      "telemetryNote": "Test permutation importance and validation MAE/RMSE/R2 across all 11 models."
    },
    {
      "title": "Correlation Matrix of Numeric & Encoded Features",
      "badge": "Correlation Audit",
      "imageUrl": "/images/projects/hourly_bike_rental_demand/image10.webp",
      "telemetryNote": "Confirms r = 0.99 temp/atemp redundancy and linear associations with cnt."
    }
  ],
  "mlVisualizationTelemetryNote": "LightGBM achieved 19.45 Train MAE, 39.70 Validation MAE (+20.25 gap), and 49.17 Test MAE under Q4 2012 ridership expansion.",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Test-set permutation importance and split-gain weights show LightGBM relies primarily on hour of day, ambient temperature, and working-day/rush-hour interactions to separate bimodal commuter peaks from weekend leisure demand, while humidity and weather severity apply corrective penalties.",
    "keyDrivers": [
      {
        "factor": "hr (Hour of Day & Sin/Cos)",
        "impact": "Highest permutation importance (>130 bikes/hr MAE increase when shuffled)"
      },
      {
        "factor": "temp & temp_hum_ratio",
        "impact": "Second largest driver; demand scales strongly in 20–30°C comfort bands"
      },
      {
        "factor": "workingday & is_rush_hour",
        "impact": "Separates bimodal weekday peaks (41.08 off-peak vs. 113.35 rush MAE)"
      },
      {
        "factor": "hum & weathersit",
        "impact": "Reduces predictions under high humidity and rain/snow (66.18 MAE in light rain)"
      }
    ],
    "limitations": [
      {
        "title": "Late-2012 Concept Drift & Underprediction",
        "description": "73.47% of test predictions underpredict actual counts (MBE -31.04 bikes/hr) because ridership grew steadily beyond 2011 training levels."
      },
      {
        "title": "Extreme Weather Sample Scarcity",
        "description": "Severe weather category 4 (heavy rain/snow) occurs in only 3 hours across the entire 2-year dataset, increasing prediction uncertainty."
      },
      {
        "title": "City-Wide Aggregation Granularity",
        "description": "Predicts aggregate Washington, D.C. hourly demand rather than individual station dock availability or spatial rebalancing routes."
      }
    ],
    "futureImprovements": [
      {
        "title": "Sliding-Window Periodic Retraining",
        "target": "Eliminate -31.04 MBE drift via quarterly rolling refits"
      },
      {
        "title": "Short-Lag Autoregressive Telemetry",
        "target": "Reduce rush-hour MAE (113.35) using causal t-1/t-24 demand lags"
      },
      {
        "title": "Station-Level Spatial Disaggregation",
        "target": "Forecast dock-level rebalancing needs across commuter hubs"
      }
    ],
    "discoveryNote": "Permutation importance reflects predictive sensitivity within the chronological test distribution rather than causal inference."
  },
  "mlModelArtifacts": [
    {
      "name": "best_model.joblib",
      "format": "JOBLIB"
    },
    {
      "name": "validation_results.json",
      "format": "JSON"
    },
    {
      "name": "validation_results.csv",
      "format": "CSV"
    },
    {
      "name": "test_evaluation_results.json",
      "format": "JSON"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Supervised Learning",
    "modelArchitecture": "LightGBM Ensemble Regression",
    "coreFramework": "LightGBM and Scikit-Learn",
    "validationMethod": "Chronological Temporal Holdout Split"
  }
}
```
