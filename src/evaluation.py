import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models import WEATHER_FEATURES


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def evaluate_prediction_map(y_true, predictions):
    """Evaluate named regression predictions against one matching target."""
    rows = []
    for model_name, model_predictions in predictions.items():
        mae, rmse = calculate_metrics(y_true, model_predictions)
        rows.append({
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "Validation Observations": len(y_true),
        })
    return pd.DataFrame(rows)


def compare_weather_distributions(
    train,
    validation,
    weather_features=WEATHER_FEATURES,
):
    return pd.DataFrame({
        "Training Mean": train[weather_features].mean(),
        "Validation Mean": validation[weather_features].mean(),
        "Training Std": train[weather_features].std(),
        "Validation Std": validation[weather_features].std(),
        "Training Min": train[weather_features].min(),
        "Validation Min": validation[weather_features].min(),
        "Training Max": train[weather_features].max(),
        "Validation Max": validation[weather_features].max(),
    })
