"""Generate the tables and figures required by the SWIM report outline."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment import BikeDemandExperiment
from models import (
    TARGET,
    HOURLY_BASELINE_NAME,
    WEEKDAY_CYCLICAL_NAME,
    WEEKEND_CYCLICAL_NAME,
    CALENDAR_CYCLICAL_NAME,
    CALENDAR_SEASONS_CYCLICAL_NAME,
    CALENDAR_WEATHER_CYCLICAL_NAME,
    CALENDAR_WEATHER_SEASONS_CYCLICAL_NAME,
    RANDOM_FOREST_MODEL_FAMILY,
    RF_CALENDAR_FEATURES_NAME,
    RF_CALENDAR_WEATHER_FEATURES_NAME,
    RF_CALENDAR_SEASONS_FEATURES_NAME,
    RF_RESULT_NAMES,
)


ROOT = Path(__file__).resolve().parent.parent
SRC = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
DATA_PATH = SRC / "data" / "SeoulBikeData.csv"

LINEAR_MODELS = [
    HOURLY_BASELINE_NAME,
    WEEKDAY_CYCLICAL_NAME,
    WEEKEND_CYCLICAL_NAME,
    CALENDAR_CYCLICAL_NAME,
    CALENDAR_SEASONS_CYCLICAL_NAME,
    CALENDAR_WEATHER_CYCLICAL_NAME,
    CALENDAR_WEATHER_SEASONS_CYCLICAL_NAME,
]

RF_FEATURES = [
    RF_CALENDAR_FEATURES_NAME,
    RF_CALENDAR_WEATHER_FEATURES_NAME,
    RF_CALENDAR_SEASONS_FEATURES_NAME,
]


def reduction(old, new):
    return (old - new) / old * 100


def select_linear_models(linear):
    return (
        linear
        .set_index("Model")
        .loc[LINEAR_MODELS]
        .reset_index()
    )


def combine_results(linear, validation_size):
    rf = pd.read_csv(RESULTS / "rf_results.csv")
    rf = rf.loc[
        (rf["Model"] == RANDOM_FOREST_MODEL_FAMILY)
        & rf["Features"].isin(RF_FEATURES),
        ["Features", "MAE", "RMSE"],
    ]
    rf["Model"] = rf.pop("Features").map(RF_RESULT_NAMES)
    rf["Family"] = "Cyclical random forest regression"
    rf["Validation Observations"] = validation_size

    columns = [
        "Model",
        "Family",
        "MAE",
        "RMSE",
        "Validation Observations",
    ]
    table = pd.concat([linear, rf[columns]], ignore_index=True)
    table = table.sort_values("MAE").reset_index(drop=True)
    table.insert(0, "MAE Rank", np.arange(1, len(table) + 1))
    return table


def plot_comparison(table):
    table = table.sort_values("MAE")
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(y - 0.18, table["MAE"], 0.36, label="MAE")
    ax.barh(y + 0.18, table["RMSE"], 0.36, label="RMSE")
    ax.set_yticks(y, table["Model"])
    ax.set_xlabel("Prediction error (bikes per hour; lower is better)")
    ax.set_title("October 2018 Validation Performance")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIGURES / "model_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_weather_value(linear):
    lookup = linear.set_index("Model")
    pairs = [
        (
            "Without seasons",
            CALENDAR_CYCLICAL_NAME,
            CALENDAR_WEATHER_CYCLICAL_NAME,
        ),
        (
            "With seasons",
            CALENDAR_SEASONS_CYCLICAL_NAME,
            CALENDAR_WEATHER_SEASONS_CYCLICAL_NAME,
        ),
    ]

    labels, maes, rmses = [], [], []
    for label, calendar, weather in pairs:
        labels.append(label)
        maes.append(
            reduction(
                lookup.loc[calendar, "MAE"],
                lookup.loc[weather, "MAE"],
            )
        )
        rmses.append(
            reduction(
                lookup.loc[calendar, "RMSE"],
                lookup.loc[weather, "RMSE"],
            )
        )

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 6))
    mae_bars = ax.bar(
        x - 0.17,
        maes,
        0.34,
        label="MAE reduction",
    )
    rmse_bars = ax.bar(
        x + 0.17,
        rmses,
        0.34,
        label="RMSE reduction",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Error reduction after adding weather (%)")
    ax.set_title("Value of Weather Information on October Validation Data")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.bar_label(mae_bars, fmt="%.2f%%", padding=3)
    ax.bar_label(rmse_bars, fmt="%.2f%%", padding=3)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "weather_error_reduction.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_week(validation, predictions):
    mask = (
        (validation["Timestamp"] >= "2018-10-15")
        & (validation["Timestamp"] < "2018-10-22")
    )
    selected = [
        HOURLY_BASELINE_NAME,
        CALENDAR_WEATHER_SEASONS_CYCLICAL_NAME,
    ]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(
        validation.loc[mask, "Timestamp"],
        validation.loc[mask, TARGET],
        label="Actual",
        color="black",
        linewidth=2,
    )
    for label in selected:
        ax.plot(
            validation.loc[mask, "Timestamp"],
            predictions[label][mask.to_numpy()],
            label=label,
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Rented bike count")
    ax.set_title("Actual and Predicted Demand, 15-21 October 2018")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(
        FIGURES / "october_prediction_week.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)

    experiment = BikeDemandExperiment(DATA_PATH)
    linear_all, predictions = experiment.run_linear_models()
    linear = select_linear_models(linear_all)
    comparison = combine_results(
        linear,
        len(experiment.validation),
    )

    comparison.to_csv(
        RESULTS / "model_comparison.csv",
        index=False,
    )
    linear.to_csv(
        RESULTS / "linear_model_results.csv",
        index=False,
    )

    plot_comparison(comparison)
    plot_weather_value(linear)
    plot_week(experiment.validation, predictions)

    print(comparison.round(2).to_string(index=False))
    print(f"\nResults written to {RESULTS}")


if __name__ == "__main__":
    main()
