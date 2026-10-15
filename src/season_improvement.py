import numpy as np
import pandas as pd

from pathlib import Path

from preprocessing import prepare_dataset

from sklearn.metrics import mean_absolute_error, mean_squared_error

from models import (
    TARGET,
    WEATHER_FEATURES,
    fit_calendar_model,
    predict_calendar_model,
    fit_weather_model,
    predict_weather_model,
    fit_calendar_model_with_seasons,
    predict_calendar_model_with_seasons,
    fit_weather_model_with_seasons,
    predict_weather_model_with_seasons
)



data_path = (
        Path(__file__).parent
        / "data"
        / "SeoulBikeData.csv"
    )

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    return mae, rmse

def evaluate_models_seasons(
    y_true,
    calendar_predictions,
    calendar_predictions_with_seasons,
    weather_predictions,
    weather_predictions_with_seasons
):

    calendar_mae, calendar_rmse = calculate_metrics(
        y_true,
        calendar_predictions
    )

    calendar_mae_with_seasons, calendar_rmse_with_seasons = calculate_metrics(
        y_true,
        calendar_predictions_with_seasons
    )

    weather_mae, weather_rmse = calculate_metrics(
        y_true,
        weather_predictions
    )

    weather_mae_with_seasons, weather_rmse_with_seasons = calculate_metrics(
        y_true,
        weather_predictions_with_seasons
    )

    return f"Calendar MAE:{calendar_mae}", f"Calendar RMSE: {calendar_rmse}", f"Calendar MAE with Seasons: {calendar_mae_with_seasons}", f"Calendar RMSE with Seasons: {calendar_rmse_with_seasons}", f"Weather MAE: {weather_mae}", f"Weather RMSE: {weather_rmse}", f"Weather RMSE with Seasons: {weather_rmse_with_seasons}", f"Weather MAE with Seasons: {weather_mae_with_seasons}"

train, validation, test, info = prepare_dataset(
        data_path
    )

y_validation = validation[TARGET]

calendar_model = fit_calendar_model(train)

calendar_predictions = predict_calendar_model(
    calendar_model,
    validation
)

calendar_model_with_seasons = fit_calendar_model_with_seasons(train)

calendar_predictions_with_seasons = predict_calendar_model_with_seasons(
    calendar_model_with_seasons,
    validation
)

weather_model = fit_weather_model(train)

weather_predictions = predict_weather_model(
    weather_model,
    validation
)

weather_model_with_seasons = fit_weather_model_with_seasons(train)

weather_predictions_with_seasons = predict_weather_model_with_seasons(
    weather_model_with_seasons,
    validation
)

results = evaluate_models_seasons(
    y_validation,
    calendar_predictions,
    calendar_predictions_with_seasons,
    weather_predictions,
    weather_predictions_with_seasons
)

print(results)