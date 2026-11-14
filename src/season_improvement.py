"""Plot the effect of adding the season feature to linear regressions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiment import BikeDemandExperiment
from models import (
    CALENDAR_CYCLICAL_NAME,
    CALENDAR_SEASONS_CYCLICAL_NAME,
    CALENDAR_WEATHER_CYCLICAL_NAME,
    CALENDAR_WEATHER_SEASONS_CYCLICAL_NAME,
)


SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
DATA_PATH = SRC_DIR / "data" / "SeoulBikeData.csv"
FIGURES_DIR = ROOT_DIR / "results" / "figures"

def percentage_reduction(old_value, new_value):
    return (old_value - new_value) / old_value * 100


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    experiment = BikeDemandExperiment(DATA_PATH)
    results, _ = experiment.run_linear_models()
    lookup = results.set_index("Model")

    comparisons = [
        ("Calendar Model", CALENDAR_CYCLICAL_NAME, CALENDAR_SEASONS_CYCLICAL_NAME),
        ("Weather Model", CALENDAR_WEATHER_CYCLICAL_NAME, CALENDAR_WEATHER_SEASONS_CYCLICAL_NAME),
    ]
    model_names = []
    mae_improvements = []
    rmse_improvements = []

    for label, without_season, with_season in comparisons:
        model_names.append(label)
        mae_improvements.append(
            percentage_reduction(
                lookup.loc[without_season, "MAE"],
                lookup.loc[with_season, "MAE"],
            )
        )
        rmse_improvements.append(
            percentage_reduction(
                lookup.loc[without_season, "RMSE"],
                lookup.loc[with_season, "RMSE"],
            )
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(model_names))
    colors = ["#2E86AB", "#E63946"]

    mae_bars = ax1.bar(x, mae_improvements, color=colors, alpha=0.8)
    ax1.set_ylabel("Improvement (%)")
    ax1.set_title("MAE Improvement with Seasons Feature", fontweight="bold")
    ax1.set_xticks(x, model_names)
    ax1.grid(axis="y", alpha=0.3)
    ax1.bar_label(mae_bars, fmt="%.2f%%", padding=3, fontweight="bold")

    rmse_bars = ax2.bar(x, rmse_improvements, color=colors, alpha=0.8)
    ax2.set_ylabel("Improvement (%)")
    ax2.set_title("RMSE Improvement with Seasons Feature", fontweight="bold")
    ax2.set_xticks(x, model_names)
    ax2.grid(axis="y", alpha=0.3)
    ax2.bar_label(rmse_bars, fmt="%.2f%%", padding=3, fontweight="bold")

    output_path = FIGURES_DIR / "seasonal_percentage_improvement.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(results.round(2).to_string(index=False))
    print(f"\nSeasonal comparison saved to {output_path}")


if __name__ == "__main__":
    main()
