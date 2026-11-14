"""Tune random-forest regressors for continuous bike-demand prediction."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from evaluation import calculate_metrics
from experiment import BikeDemandExperiment
from models import (
    TARGET,
    HOURLY_BASELINE_NAME,
    RANDOM_FOREST_MODEL_FAMILY,
    RF_CALENDAR_FEATURES_NAME,
    RF_CALENDAR_WEATHER_FEATURES_NAME,
    RF_CALENDAR_SEASONS_FEATURES_NAME,
    RF_CALENDAR_WEATHER_SEASONS_FEATURES_NAME,
    CATEGORICAL_FEATURES,
    CALENDAR_FEATURES_WITHOUT_SEASONS,
    CALENDAR_FEATURES_WITH_SEASONS,
    WEATHER_FEATURES,
    fit_hourly_baseline,
    predict_hourly_baseline,
)


SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
DATA_FILE = SRC_DIR / "data" / "SeoulBikeData.csv"
RESULTS_DIR = ROOT_DIR / "results"

SEED = 42
N_TREES = 100
N_SPLITS = 5

PARAM_GRID = {
    "max_features": [1.0, 0.5, 0.33],
    "min_samples_leaf": [1, 5, 10],
    "max_depth": [None, 20],
}

FEATURE_SETS = {
    RF_CALENDAR_FEATURES_NAME:
        CALENDAR_FEATURES_WITHOUT_SEASONS,
    RF_CALENDAR_WEATHER_FEATURES_NAME:
        CALENDAR_FEATURES_WITHOUT_SEASONS + WEATHER_FEATURES,
    RF_CALENDAR_SEASONS_FEATURES_NAME:
        CALENDAR_FEATURES_WITH_SEASONS,
    RF_CALENDAR_WEATHER_SEASONS_FEATURES_NAME:
        CALENDAR_FEATURES_WITH_SEASONS + WEATHER_FEATURES,
}


def make_design_matrix(data, features):
    categorical = [
        feature for feature in CATEGORICAL_FEATURES
        if feature in features
    ]
    return pd.get_dummies(
        data[features],
        columns=categorical,
    )


def tune_random_forest(train, validation, features):
    x_train = make_design_matrix(train, features)
    x_validation = make_design_matrix(
        validation,
        features,
    ).reindex(columns=x_train.columns, fill_value=0)

    search = GridSearchCV(
        RandomForestRegressor(
            n_estimators=N_TREES,
            random_state=SEED,
            n_jobs=-1,
        ),
        PARAM_GRID,
        cv=TimeSeriesSplit(n_splits=N_SPLITS),
        scoring="neg_mean_absolute_error",
    )
    search.fit(x_train, train[TARGET])
    predictions = search.best_estimator_.predict(x_validation)
    return search, predictions


def print_weather_improvement(results):
    lookup = results.set_index("Features")
    comparisons = [
        (
            RF_CALENDAR_FEATURES_NAME,
            RF_CALENDAR_WEATHER_FEATURES_NAME,
        ),
        (
            RF_CALENDAR_SEASONS_FEATURES_NAME,
            RF_CALENDAR_WEATHER_SEASONS_FEATURES_NAME,
        ),
    ]
    for calendar, weather in comparisons:
        old_mae, old_rmse = lookup.loc[
            calendar, ["MAE", "RMSE"]
        ]
        new_mae, new_rmse = lookup.loc[
            weather, ["MAE", "RMSE"]
        ]
        print(f"\nWeather improvement over '{calendar}':")
        print(
            f"  MAE: {old_mae - new_mae:.2f} "
            f"({(old_mae - new_mae) / old_mae * 100:.2f}%)"
        )
        print(
            f"  RMSE: {old_rmse - new_rmse:.2f} "
            f"({(old_rmse - new_rmse) / old_rmse * 100:.2f}%)"
        )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    experiment = BikeDemandExperiment(DATA_FILE)
    train = experiment.train
    validation = experiment.validation
    y_validation = experiment.y_validation

    baseline = fit_hourly_baseline(train)
    baseline_predictions = predict_hourly_baseline(
        baseline,
        validation,
    )
    baseline_mae, baseline_rmse = calculate_metrics(
        y_validation,
        baseline_predictions,
    )

    rows = [{
        "Model": HOURLY_BASELINE_NAME,
        "Features": "Hour",
        "MAE": baseline_mae,
        "RMSE": baseline_rmse,
    }]

    for feature_name, features in FEATURE_SETS.items():
        search, predictions = tune_random_forest(
            train,
            validation,
            features,
        )
        mae, rmse = calculate_metrics(
            y_validation,
            predictions,
        )
        rows.append({
            "Model": RANDOM_FOREST_MODEL_FAMILY,
            "Features": feature_name,
            "MAE": mae,
            "RMSE": rmse,
            "CV MAE": -search.best_score_,
            **search.best_params_,
        })

    results = pd.DataFrame(rows)
    output_path = RESULTS_DIR / "rf_results.csv"
    results.to_csv(output_path, index=False)

    print(
        f"\nOctober 2018 validation results "
        f"({len(y_validation)} observations)\n"
    )
    print(results.round(2).to_string(index=False))
    print_weather_improvement(
        results.loc[
            results["Model"] == RANDOM_FOREST_MODEL_FAMILY
        ]
    )
    print(f"\nResults table saved to {output_path}")


if __name__ == "__main__":
    main()
