"""Reusable experiment object for the Seoul bike-demand project."""

from pathlib import Path

import pandas as pd

from evaluation import evaluate_prediction_map
from models import (
    TARGET,
    HOURLY_BASELINE_NAME,
    WEEKDAY_CYCLICAL_NAME,
    WEEKEND_CYCLICAL_NAME,
    CALENDAR_CYCLICAL_NAME,
    CALENDAR_SEASONS_CYCLICAL_NAME,
    CALENDAR_WEATHER_CYCLICAL_NAME,
    CALENDAR_WEATHER_SEASONS_CYCLICAL_NAME,
    fit_hourly_baseline,
    predict_hourly_baseline,
    fit_periodic_day_type_model,
    fit_calendar_model,
    predict_calendar_model,
    fit_calendar_model_with_seasons,
    predict_calendar_model_with_seasons,
    fit_weather_model,
    predict_weather_model,
    fit_weather_model_with_seasons,
    predict_weather_model_with_seasons,
)
from preprocessing import prepare_dataset


class BikeDemandExperiment:
    """Own the shared data split and regression experiment lifecycle."""

    def __init__(self, data_path):
        self.data_path = Path(data_path)
        (
            self.train,
            self.validation,
            self.test,
            self.data_info,
        ) = prepare_dataset(self.data_path)

    @property
    def y_validation(self):
        return self.validation[TARGET]

    def _run_day_type_models(self):
        model = fit_periodic_day_type_model(self.train)
        weekend = self.validation["IsWeekend"].astype(bool)

        weekday_results = evaluate_prediction_map(
            self.validation.loc[~weekend, TARGET],
            {
                WEEKDAY_CYCLICAL_NAME:
                    model.predict_weekdays(self.validation)
            },
        )
        weekend_results = evaluate_prediction_map(
            self.validation.loc[weekend, TARGET],
            {
                WEEKEND_CYCLICAL_NAME:
                    model.predict_weekends(self.validation)
            },
        )
        return pd.concat(
            [weekday_results, weekend_results],
            ignore_index=True,
        )

    def run_linear_models(self):
        baseline_model = fit_hourly_baseline(self.train)
        baseline_predictions = {
            HOURLY_BASELINE_NAME: predict_hourly_baseline(
                baseline_model,
                self.validation,
            )
        }

        calendar_model = fit_calendar_model(self.train)
        calendar_seasons_model = fit_calendar_model_with_seasons(
            self.train
        )
        weather_model = fit_weather_model(self.train)
        weather_seasons_model = fit_weather_model_with_seasons(
            self.train
        )

        regression_predictions = {
            CALENDAR_CYCLICAL_NAME: predict_calendar_model(
                calendar_model,
                self.validation,
            ),
            CALENDAR_SEASONS_CYCLICAL_NAME:
                predict_calendar_model_with_seasons(
                    calendar_seasons_model,
                    self.validation,
                ),
            CALENDAR_WEATHER_CYCLICAL_NAME: predict_weather_model(
                weather_model,
                self.validation,
            ),
            CALENDAR_WEATHER_SEASONS_CYCLICAL_NAME:
                predict_weather_model_with_seasons(
                    weather_seasons_model,
                    self.validation,
                ),
        }

        baseline_results = evaluate_prediction_map(
            self.y_validation,
            baseline_predictions,
        )
        baseline_results.insert(1, "Family", "Baseline")

        day_type_results = self._run_day_type_models()
        day_type_results.insert(1, "Family", "Cyclical linear regression")

        regression_results = evaluate_prediction_map(
            self.y_validation,
            regression_predictions,
        )
        regression_results.insert(1, "Family", "Cyclical linear regression")

        results = pd.concat(
            [
                baseline_results,
                day_type_results,
                regression_results,
            ],
            ignore_index=True,
        )
        predictions = {
            **baseline_predictions,
            **regression_predictions,
        }
        return results, predictions
