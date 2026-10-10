import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error


WEATHER_FEATURES = [
    "Temperature(°C)",
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Dew point temperature(°C)",
    "Solar Radiation (MJ/m2)",
    "Rainfall(mm)",
    "Snowfall (cm)"
]


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    return mae, rmse


def evaluate_models(
    y_true,
    baseline_predictions,
    calendar_predictions,
    weather_predictions
):
    baseline_mae, baseline_rmse = calculate_metrics(
        y_true,
        baseline_predictions
    )

    calendar_mae, calendar_rmse = calculate_metrics(
        y_true,
        calendar_predictions
    )

    weather_mae, weather_rmse = calculate_metrics(
        y_true,
        weather_predictions
    )

    results = pd.DataFrame({
        "Model": [
            "Hourly Mean Baseline",
            "Calendar Linear Regression",
            "Calendar + Weather Linear Regression"
        ],
        "MAE": [
            baseline_mae,
            calendar_mae,
            weather_mae
        ],
        "RMSE": [
            baseline_rmse,
            calendar_rmse,
            weather_rmse
        ],
        "Validation Observations": [
            len(y_true),
            len(y_true),
            len(y_true)
        ]
    })

    return results


def calculate_weather_improvement(results):
    calendar = results[
        results["Model"] == "Calendar Linear Regression"
    ].iloc[0]

    weather = results[
        results["Model"]
        == "Calendar + Weather Linear Regression"
    ].iloc[0]

    mae_difference = calendar["MAE"] - weather["MAE"]
    rmse_difference = calendar["RMSE"] - weather["RMSE"]

    mae_percentage = (
        mae_difference / calendar["MAE"]
    ) * 100

    rmse_percentage = (
        rmse_difference / calendar["RMSE"]
    ) * 100

    improvement = {
        "MAE difference": mae_difference,
        "MAE percentage": mae_percentage,
        "RMSE difference": rmse_difference,
        "RMSE percentage": rmse_percentage
    }

    return improvement


def compare_weather_distributions(
    train,
    validation,
    weather_features=WEATHER_FEATURES
):
    comparison = pd.DataFrame({
        "Training Mean": train[weather_features].mean(),
        "Validation Mean": validation[weather_features].mean(),
        "Training Std": train[weather_features].std(),
        "Validation Std": validation[weather_features].std(),
        "Training Min": train[weather_features].min(),
        "Validation Min": validation[weather_features].min(),
        "Training Max": train[weather_features].max(),
        "Validation Max": validation[weather_features].max()
    })

    return comparison


def print_results(results):
    print(results.to_string(index=False))


def print_weather_improvement(improvement):
    print(
        f"MAE improvement: "
        f"{improvement['MAE difference']:.2f} "
        f"({improvement['MAE percentage']:.2f}%)"
    )

    print(
        f"RMSE improvement: "
        f"{improvement['RMSE difference']:.2f} "
        f"({improvement['RMSE percentage']:.2f}%)"
    )