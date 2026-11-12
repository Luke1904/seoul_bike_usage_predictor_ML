import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


DATA_FILE = "SeoulBikeData.csv"
TARGET = "Rented Bike Count"

SEED = 42
N_TREES = 100
N_SPLITS = 5

param_grid = {
    "max_features": [1.0, 0.5, 0.33],
    "min_samples_leaf": [1, 5, 10],
    "max_depth": [None, 20],
}

CALENDAR_FEATURES_WITHOUT_SEASONS = ["Hour", "Weekday", "Holiday"]
CALENDAR_FEATURES_WITH_SEASONS = ["Hour", "Weekday", "Seasons", "Holiday"]
WEATHER_FEATURES = [
    "Temperature(°C)",
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Dew point temperature(°C)",
    "Solar Radiation (MJ/m2)",
    "Rainfall(mm)",
    "Snowfall (cm)",
]
CATEGORICAL = ["Hour", "Weekday", "Seasons", "Holiday"]

feature_sets = {
    "Calendar": CALENDAR_FEATURES_WITHOUT_SEASONS,
    "Calendar + Weather": CALENDAR_FEATURES_WITHOUT_SEASONS + WEATHER_FEATURES,
    "Calendar with Seasons": CALENDAR_FEATURES_WITH_SEASONS,
    "Calendar with Seasons + Weather":
        CALENDAR_FEATURES_WITH_SEASONS + WEATHER_FEATURES,
}

df = pd.read_csv(DATA_FILE, encoding="latin1")
df.columns = df.columns.str.strip()

df = df.rename(columns={c: "Temperature(°C)" for c in df.columns
                        if c.startswith("Temperature")})
df = df.rename(columns={c: "Dew point temperature(°C)" for c in df.columns
                        if c.startswith("Dew point temperature")})

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df["Weekday"] = df["Date"].dt.day_name()
df["Timestamp"] = df["Date"] + pd.to_timedelta(df["Hour"], unit="h")

print(f"Rows: {len(df)}")
print(f"Missing values: {int(df.isna().sum().sum())}")
print(f"Duplicate rows: {int(df.duplicated().sum())}")

df = df.sort_values(["Date", "Hour"]).reset_index(drop=True)

non_functioning = df["Functioning Day"].astype(str).str.strip().str.lower() == "no"
print(f"Non-functioning observations excluded: {int(non_functioning.sum())}")
df = df.loc[~non_functioning].reset_index(drop=True)

train = df[df["Date"] < "2018-10-01"].copy()
validation = df[(df["Date"] >= "2018-10-01") & (df["Date"] < "2018-11-01")].copy()

print(f"Training: {len(train)} observations (expected 7152)")
print(f"Validation: {len(validation)} observations (expected 665)")

y_train = train[TARGET]
y_val = validation[TARGET]

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def make_X(data, features):
    """One-hot encode the calendar features; weather stays numeric."""
    return pd.get_dummies(data[features],
                          columns=[c for c in CATEGORICAL if c in features])


hourly_means = train.groupby("Hour")[TARGET].mean()
baseline_predictions = validation["Hour"].map(hourly_means).to_numpy()
base_mae, base_rmse = calculate_metrics(y_val, baseline_predictions)

rows = [{"Model": "Hourly Mean Baseline", "Features": "Hour",
         "MAE": base_mae, "RMSE": base_rmse}]


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
print(f"\nOctober 2018 validation results ({len(y_val)} observations)\n")
print(results.round(2).to_string(index=False))
results.to_csv("rf_results.csv", index=False)


lookup = results.set_index("Features")
for without, with_weather in [
        ("Calendar", "Calendar + Weather"),
        ("Calendar with Seasons", "Calendar with Seasons + Weather")]:
    m0, r0 = lookup.loc[without, ["MAE", "RMSE"]]
    m1, r1 = lookup.loc[with_weather, ["MAE", "RMSE"]]
    print(f"\nWeather improvement over '{without}':")
    print(f"  MAE:  {m0 - m1:.2f} ({(m0 - m1) / m0 * 100:.2f}%)")
    print(f"  RMSE: {r0 - r1:.2f} ({(r0 - r1) / r0 * 100:.2f}%)")


CAL, WEA = "Calendar with Seasons", "Calendar with Seasons + Weather"


labels = ["Hourly Mean Baseline", "RF Calendar", "RF Calendar + Weather"]
mae_vals = [base_mae, lookup.loc[CAL, "MAE"], lookup.loc[WEA, "MAE"]]
rmse_vals = [base_rmse, lookup.loc[CAL, "RMSE"], lookup.loc[WEA, "RMSE"]]
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(12, 7))
plt.bar(x - width / 2, mae_vals, width, label="MAE")
plt.bar(x + width / 2, rmse_vals, width, label="RMSE")
plt.xticks(x, labels, rotation=15)
plt.ylabel("Prediction Error (bikes per hour)")
plt.title("Random Forest: October Validation Performance")
plt.legend()
plt.tight_layout()
plt.show()

mask = ((validation["Timestamp"] >= "2018-10-15")
        & (validation["Timestamp"] < "2018-10-22")).to_numpy()
times = validation["Timestamp"].to_numpy()[mask]

plt.figure(figsize=(14, 7))
plt.plot(times, y_val.to_numpy()[mask], label="Actual", color="black", linewidth=2)
plt.plot(times, baseline_predictions[mask], label="Baseline")
plt.plot(times, predictions[CAL][mask], label="RF Calendar")
plt.plot(times, predictions[WEA][mask], label="RF Calendar + Weather")
plt.xlabel("Date")
plt.ylabel("Rented Bike Count")
plt.title("Random Forest: Actual vs Predicted, 15-21 October 2018")
plt.legend()
plt.tight_layout()
plt.show()

residuals = y_val.to_numpy() - predictions[WEA]
plt.figure(figsize=(10, 6))
plt.scatter(predictions[WEA], residuals, alpha=0.4)
plt.axhline(0)
plt.xlabel("Predicted Bike Demand")
plt.ylabel("Residual")
plt.title("Random Forest Residuals: Calendar + Weather")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

model, cols = fitted[WEA]
importance = pd.Series(model.feature_importances_, index=cols).nlargest(10)
plt.figure(figsize=(10, 6))
plt.barh(importance.index[::-1], importance.values[::-1])
plt.xlabel("Importance")
plt.title("Random Forest: Top 10 Features (Calendar + Weather)")
plt.tight_layout()
plt.show()

print("\nResults table saved to rf_results.csv")
