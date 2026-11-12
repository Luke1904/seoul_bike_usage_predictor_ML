import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Extract metrics from results
calendar_mae = mean_absolute_error(y_validation, calendar_predictions)
calendar_rmse = np.sqrt(mean_squared_error(y_validation, calendar_predictions))
calendar_mae_with_seasons = mean_absolute_error(y_validation, calendar_predictions_with_seasons)
calendar_rmse_with_seasons = np.sqrt(mean_squared_error(y_validation, calendar_predictions_with_seasons))
weather_mae = mean_absolute_error(y_validation, weather_predictions)
weather_rmse = np.sqrt(mean_squared_error(y_validation, weather_predictions))
weather_mae_with_seasons = mean_absolute_error(y_validation, weather_predictions_with_seasons)
weather_rmse_with_seasons = np.sqrt(mean_squared_error(y_validation, weather_predictions_with_seasons))

# Calculate percentage improvements
mae_improvements = [
    ((calendar_mae - calendar_mae_with_seasons) / calendar_mae) * 100,
    ((weather_mae - weather_mae_with_seasons) / weather_mae) * 100
]

rmse_improvements = [
    ((calendar_rmse - calendar_rmse_with_seasons) / calendar_rmse) * 100,
    ((weather_rmse - weather_rmse_with_seasons) / weather_rmse) * 100
]

# Create percentage improvement graph
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

model_names = ['Calendar Model', 'Weather Model']
x = np.arange(len(model_names))
width = 0.35

# MAE improvements
bars1 = ax1.bar(x, mae_improvements, width, color=['#2E86AB', '#E63946'], alpha=0.8)
ax1.set_ylabel('Improvement (%)', fontsize=12)
ax1.set_title('MAE Improvement with Seasons Feature', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names)
ax1.grid(axis='y', alpha=0.3)

for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')

# RMSE improvements
bars2 = ax2.bar(x, rmse_improvements, width, color=['#2E86AB', '#E63946'], alpha=0.8)
ax2.set_ylabel('Improvement (%)', fontsize=12)
ax2.set_title('RMSE Improvement with Seasons Feature', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(model_names)
ax2.grid(axis='y', alpha=0.3)

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('seasonal_percentage_improvement.png', dpi=300, bbox_inches='tight')
plt.show()