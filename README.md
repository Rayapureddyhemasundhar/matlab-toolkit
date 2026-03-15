# MATLAB Data Preprocessing & Analysis Toolkit

A modular, production-ready pipeline for preprocessing tabular and time-series data
and training regression models in MATLAB.

---

## Project Structure

```
matlab-data-preprocessing-toolkit/
│
├── src/
│   ├── main_pipeline.m          Master pipeline controller
│   ├── config.m                 Central configuration
│   ├── logger.m                 Logging system
│   ├── safe_normalization.m     Robust feature scaling
│   ├── feature_selection.m      Multiple selection methods
│   ├── feature_importance.m     Post-training importance plots
│   ├── feature_engineering.m    Lag, rolling, derivative features
│   ├── load_real_dataset.m      Dataset loader / generator
│   ├── handle_missing_values.m  Multi-method imputation
│   ├── detect_outliers.m        Consensus outlier detection
│   ├── train_models.m           Model training and evaluation
│   └── hyperparameter_tuning.m  Bayesian hyperparameter search
│
├── utils/
│   ├── export_results.m         CSV / MAT file export
│   └── generate_report.m        HTML report generator
│
├── tests/
│   └── unit_tests.m             Full test suite
│
├── data/                        Input datasets
├── outputs/                     Exported results and reports
├── logs/                        Log files
├── figures/                     Saved plots
├── benchmarks/                  Execution time records
│
├── example_run.m                Quick-start entry point
└── README.md                    This file
```

---

## Quick Start

```matlab
% Quick demonstration on synthetic data
example_run

% Full pipeline on a real dataset
main_pipeline
```

---

## Configuration

All settings are in `src/config.m`. Key options:

| Parameter                  | Default          | Description                                      |
|----------------------------|------------------|--------------------------------------------------|
| `cfg.dataset`              | `'air_quality'`  | Dataset: air_quality, boston_housing, weather, stock, synthetic |
| `cfg.train_ratio`          | `0.8`            | Proportion of data used for training             |
| `cfg.random_seed`          | `42`             | Seed for reproducibility                         |
| `cfg.feature_selection_method` | `'pca'`      | pca, correlation, variance, mutual_info, rfe, none |
| `cfg.ml_models`            | (7 models)       | Subset of: Linear, Ridge, Lasso, Tree, RandomForest, SVM, Ensemble |
| `cfg.hyperparameter_tuning`| `true`           | Enable Bayesian optimization                     |
| `cfg.cv_folds`             | `5`              | Cross-validation folds                           |

---

## Pipeline Steps

1. Load dataset (real or synthetic)
2. Handle missing values (9 methods, auto-select by RMSE)
3. Detect and treat outliers (5 methods, consensus voting, winsorization)
4. Normalize features (StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler)
5. Engineer features (rolling stats, lags, derivatives, interactions)
6. Select features (PCA, correlation filter, variance, mutual info, RFE)
7. Train/test split
8. Train models (7 regression algorithms)
9. Bayesian hyperparameter tuning (optional)
10. Feature importance analysis
11. Export results (CSV, MAT)
12. Generate HTML report

---

## Running Tests

```matlab
cd tests
unit_tests
```

---

## Requirements

- MATLAB R2021a or later
- Statistics and Machine Learning Toolbox
- Signal Processing Toolbox (optional, for certain smoothing methods)

---

## Outputs

After running the pipeline, the `outputs/` directory contains:

- `cleaned_data.csv`          Imputed and outlier-treated data
- `engineered_features.csv`   Full expanded feature matrix
- `selected_features.csv`     Post-selection feature matrix
- `model_comparison.csv`      RMSE, R2, MAE, and time for each model
- `pipeline_results.mat`      Full MATLAB workspace
- `pipeline_report.html`      Self-contained HTML report

Figures are saved to `figures/` and logs to `logs/pipeline.log`.

---

## License

MIT License.
