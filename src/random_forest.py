"""
Random forest regression for hourly Seoul bike rentals.

Uses the team's shared pipeline so the results are directly comparable
with the linear models:
  - preprocessing.prepare_dataset  -> same cleaning and chronological split
  - models.py                      -> same target, feature lists and baseline
  - MAE and RMSE on October 2018   -> same metrics as the linear models
  - visualisation.py               -> same chart functions and style

Run from the project folder:
    python random_forest.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from preprocessing import prepare_dataset
from models import (
    TARGET,
    WEATHER_FEATURES,
    CALENDAR_FEATURES_WITHOUT_SEASONS,
    CALENDAR_FEATURES_WITH_SEASONS,
    fit_hourly_baseline,
    predict_hourly_baseline,
)
from visualisation import (
    plot_model_performance,
    plot_prediction_week,
    plot_residuals,
)

# ---------- Settings ----------
PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "data" / "SeoulBikeData.csv"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
N_TREES = 300
N_SPLITS = 5

param_grid = {
    "max_features": [1.0, 0.5, 0.33],
    "min_samples_leaf": [1, 5, 10],
    "max_depth": [None, 20],
}

# Same four feature sets as the linear models
feature_sets = {
    "Calendar": CALENDAR_FEATURES_WITHOUT_SEASONS,
    "Calendar + Weather": CALENDAR_FEATURES_WITHOUT_SEASONS + WEATHER_FEATURES,
    "Calendar with Seasons": CALENDAR_FEATURES_WITH_SEASONS,
    "Calendar with Seasons + Weather":
        CALENDAR_FEATURES_WITH_SEASONS + WEATHER_FEATURES,
}
CATEGORICAL = ["Hour", "Weekday", "Seasons", "Holiday"]


def calculate_metrics(y_true, y_pred):
    """MAE and RMSE, computed exactly as in the team's evaluation code."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def make_X(data, features):
    """One-hot encode the categorical calendar features (weather stays numeric)."""
    return pd.get_dummies(data[features],
                          columns=[c for c in CATEGORICAL if c in features])


# ---------- 1. Data: the team's shared preprocessing ----------
train, validation, test, info = prepare_dataset(DATA_PATH)
# The test set (November) is deliberately not used here.

y_train = train[TARGET]
y_val = validation[TARGET]

# ---------- 2. Hourly mean baseline (the team's function) ----------
baseline = fit_hourly_baseline(train)
base_mae, base_rmse = calculate_metrics(
    y_val, predict_hourly_baseline(baseline, validation))

rows = [{"Model": "Hourly Mean Baseline", "Features": "Hour",
         "MAE": base_mae, "RMSE": base_rmse}]

# ---------- 3. Tune, train and evaluate the random forest ----------
predictions = {}
fitted = {}

for name, features in feature_sets.items():
    X_train = make_X(train, features)
    X_val = make_X(validation, features).reindex(columns=X_train.columns,
                                                 fill_value=0)

    search = GridSearchCV(
        RandomForestRegressor(n_estimators=N_TREES, random_state=SEED, n_jobs=-1),
        param_grid,
        cv=TimeSeriesSplit(n_splits=N_SPLITS),
        scoring="neg_mean_absolute_error",
    )
    search.fit(X_train, y_train)

    pred = search.best_estimator_.predict(X_val)
    mae, rmse = calculate_metrics(y_val, pred)

    predictions[name] = pred
    fitted[name] = (search.best_estimator_, X_train.columns)
    rows.append({"Model": "Random Forest", "Features": name,
                 "MAE": mae, "RMSE": rmse,
                 "CV MAE": -search.best_score_, **search.best_params_})

results = pd.DataFrame(rows)

print("\nOctober 2018 validation results "
      f"({len(y_val)} observations)\n")
print(results.round(2).to_string(index=False))

# ---------- 4. Weather improvement (same formula as the team's) ----------
lookup = results.set_index("Features")
print()
for without, with_weather in [
        ("Calendar", "Calendar + Weather"),
        ("Calendar with Seasons", "Calendar with Seasons + Weather")]:
    m0, r0 = lookup.loc[without, ["MAE", "RMSE"]]
    m1, r1 = lookup.loc[with_weather, ["MAE", "RMSE"]]
    print(f"Weather improvement over '{without}':")
    print(f"  MAE:  {m0 - m1:.2f} ({(m0 - m1) / m0 * 100:.2f}%)")
    print(f"  RMSE: {r0 - r1:.2f} ({(r0 - r1) / r0 * 100:.2f}%)")

results.to_csv(RESULTS_DIR / "rf_results.csv", index=False)

# ---------- 5. Figures: the team's visualisation functions ----------
CAL, WEA = "Calendar with Seasons", "Calendar with Seasons + Weather"
baseline_predictions = predict_hourly_baseline(baseline, validation)

# Error comparison: baseline vs random forest (same chart style as the team's)
performance = pd.DataFrame({
    "Model": ["Hourly Mean Baseline", "RF Calendar", "RF Calendar + Weather"],
    "MAE": [base_mae, lookup.loc[CAL, "MAE"], lookup.loc[WEA, "MAE"]],
    "RMSE": [base_rmse, lookup.loc[CAL, "RMSE"], lookup.loc[WEA, "RMSE"]],
})
plot_model_performance(performance)

# One week of actual vs predicted (lines labelled "Calendar" and
# "Calendar + Weather" are the random forest predictions here)
plot_prediction_week(
    validation,
    baseline_predictions,
    predictions[CAL],
    predictions[WEA],
    start_date="2018-10-15",
)

# Residuals of the random forest
plot_residuals(y_val, predictions[WEA],
               "Random Forest Residuals: Calendar + Weather")

# Feature importance (random-forest specific, so not in the team's file)
model, cols = fitted[WEA]
importance = pd.Series(model.feature_importances_, index=cols).nlargest(10)
plt.figure(figsize=(10, 6))
plt.barh(importance.index[::-1], importance.values[::-1])
plt.xlabel("Importance")
plt.title("Random Forest: Top 10 Features (Calendar + Weather)")
plt.tight_layout()
plt.show()

print(f"\nResults table saved to {RESULTS_DIR / 'rf_results.csv'}")
