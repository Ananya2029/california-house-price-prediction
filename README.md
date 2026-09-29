# California House Price Prediction

End-to-end regression project predicting median house values for California districts, comparing linear and tree-based models with cross-validation and hyperparameter tuning.

## Data

[California Housing Prices (Kaggle)](https://www.kaggle.com/datasets/camnugent/california-housing-prices) — 20,640 districts, 9 features (location, housing age, rooms, population, income, ocean proximity).
Download `housing.csv` into `data/` before running the notebook.

## Approach

1. **EDA** — distributions, outliers, correlations (median income is the strongest predictor, r = 0.69), the 500,001 price cap.
2. **Preprocessing pipeline** — `ColumnTransformer` with median imputation + scaling for numeric features and one-hot encoding for `ocean_proximity`, inside a scikit-learn `Pipeline` so there is no train/test leakage.
3. **Model comparison** — 5-fold cross-validation of Linear Regression, Ridge, Lasso, Random Forest and HistGradientBoosting.
4. **Tuning** — `GridSearchCV` over 243 HistGradientBoosting configurations.
5. **Evaluation** — hold-out test set, residual analysis, and a reusable `predict_house_price()` function.

## Results (hold-out test set, 4,128 districts)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression (baseline) | $70,059 | $50,670 | 0.625 |
| **Tuned HistGradientBoosting** | **$47,170** | **$30,948** | **0.830** |

The tuned model cuts error by about a third versus the linear baseline.

## Files

| File | Description |
|---|---|
| `15_4_house_price_prediction.ipynb` | Main project notebook (California housing) |
| `v1-fs-...-feature_selection-v1.ipynb` | Earlier feature-selection experiment on the Kaggle Ames dataset (`house-prices-advanced-regression-techniques.zip`); needs `df_for_feature_engineering.csv` from a previous feature-engineering step |

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn
