from pathlib import Path

from preprocessing import prepare_dataset

from models import (
    TARGET,
    WEATHER_FEATURES,
    fit_hourly_baseline,
    predict_hourly_baseline,
    fit_calendar_model,
    predict_calendar_model,
    fit_weather_model,
    predict_weather_model
)

from evaluation import (
    evaluate_models,
    calculate_weather_improvement,
    compare_weather_distributions,
    print_results,
    print_weather_improvement
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
    plot_residuals
)


def main():
    data_path = (
        Path(__file__).parent
        / "data"
        / "SeoulBikeData.csv"
    )
    
    train, validation, test, info = prepare_dataset(
        data_path
    )

    y_validation = validation[TARGET]

    baseline_model = fit_hourly_baseline(train)

    baseline_predictions = predict_hourly_baseline(
        baseline_model,
        validation
    )

    calendar_model = fit_calendar_model(train)

    calendar_predictions = predict_calendar_model(
        calendar_model,
        validation
    )

    weather_model = fit_weather_model(train)

    weather_predictions = predict_weather_model(
        weather_model,
        validation
    )

    results = evaluate_models(
        y_validation,
        baseline_predictions,
        calendar_predictions,
        weather_predictions
    )

    improvement = calculate_weather_improvement(
        results
    )

    weather_comparison = compare_weather_distributions(
        train,
        validation
    )

    print("\nModel Results")
    print_results(results)

    print("\nValue of Weather Information")
    print_weather_improvement(improvement)

    print("\nTraining vs October Weather")
    print(weather_comparison.round(2))

    plot_hourly_demand(train)

    plot_weekday_demand(train)

    plot_hour_weekday_demand(train)

    plot_seasonal_demand(train)

    plot_temperature_demand(train)

    plot_rainfall_demand(train)

    plot_rain_vs_no_rain(train)

    plot_weather_correlations(
        train,
        WEATHER_FEATURES
    )

    plot_temperature_shift(
        train,
        validation
    )

    plot_model_performance(
        results
    )

    plot_predictions(
        validation,
        baseline_predictions,
        calendar_predictions,
        weather_predictions
    )

    plot_prediction_week(
        validation,
        baseline_predictions,
        calendar_predictions,
        weather_predictions
    )

    plot_residuals(
        y_validation,
        calendar_predictions,
        "Calendar Model Residuals"
    )

    plot_residuals(
        y_validation,
        weather_predictions,
        "Calendar + Weather Model Residuals"
    )


if __name__ == "__main__":
    main()