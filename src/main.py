"""Interactive exploratory run for the Seoul bike-demand project."""

from pathlib import Path

from evaluation import compare_weather_distributions
from experiment import BikeDemandExperiment
from models import (
    TARGET,
    WEATHER_FEATURES,
    HOURLY_BASELINE_NAME,
    CALENDAR_CYCLICAL_NAME,
    CALENDAR_WEATHER_CYCLICAL_NAME,
    CALENDAR_WEATHER_SEASONS_CYCLICAL_NAME,
)
from visualization import (
    plot_hourly_demand,
    plot_weekday_demand,
    plot_hour_weekday_demand,
    plot_seasonal_demand,
    plot_temperature_demand,
    plot_rainfall_demand,
    plot_rain_vs_no_rain,
    plot_weather_correlations,
    plot_temperature_shift,
    plot_model_performance,
    plot_predictions,
    plot_prediction_week,
    plot_residuals,
)


SRC_DIR = Path(__file__).resolve().parent
DATA_PATH = SRC_DIR / "data" / "SeoulBikeData.csv"


def print_weather_improvement(results):
    lookup = results.set_index("Model")
    old = lookup.loc[CALENDAR_CYCLICAL_NAME]
    new = lookup.loc[CALENDAR_WEATHER_CYCLICAL_NAME]
    print(
        f"MAE improvement: {old['MAE'] - new['MAE']:.2f} "
        f"({(old['MAE'] - new['MAE']) / old['MAE'] * 100:.2f}%)"
    )
    print(
        f"RMSE improvement: {old['RMSE'] - new['RMSE']:.2f} "
        f"({(old['RMSE'] - new['RMSE']) / old['RMSE'] * 100:.2f}%)"
    )


def main():
    experiment = BikeDemandExperiment(DATA_PATH)
    results, predictions = experiment.run_linear_models()

    print("\nModel Results")
    print(results.to_string(index=False))

    print("\nValue of Weather Information")
    print_weather_improvement(results)

    print("\nTraining vs October Weather")
    weather_comparison = compare_weather_distributions(
        experiment.train,
        experiment.validation,
    )
    print(weather_comparison.round(2))

    plot_hourly_demand(experiment.train)
    plot_weekday_demand(experiment.train)
    plot_hour_weekday_demand(experiment.train)
    plot_seasonal_demand(experiment.train)
    plot_temperature_demand(experiment.train)
    plot_rainfall_demand(experiment.train)
    plot_rain_vs_no_rain(experiment.train)
    plot_weather_correlations(
        experiment.train,
        WEATHER_FEATURES,
    )
    plot_temperature_shift(
        experiment.train,
        experiment.validation,
    )
    plot_model_performance(results)

    baseline = predictions[HOURLY_BASELINE_NAME]
    calendar = predictions[CALENDAR_CYCLICAL_NAME]
    weather = predictions[
        CALENDAR_WEATHER_SEASONS_CYCLICAL_NAME
    ]

    plot_predictions(
        experiment.validation,
        baseline,
        calendar,
        weather,
    )
    plot_prediction_week(
        experiment.validation,
        baseline,
        calendar,
        weather,
    )
    plot_residuals(
        experiment.y_validation,
        calendar,
        "Calendar-Only Cyclical Linear Regression Residuals",
    )
    plot_residuals(
        experiment.y_validation,
        weather,
        "Calendar + Weather + Seasons Cyclical Linear Regression Residuals",
    )


if __name__ == "__main__":
    main()
