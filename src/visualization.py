import numpy as np
import matplotlib.pyplot as plt


TARGET = "Rented Bike Count"


def plot_hourly_demand(df):
    hourly = df.groupby("Hour")[TARGET].mean()

    plt.figure(figsize=(12, 7))
    plt.plot(hourly.index, hourly.values, marker="o")
    plt.xlabel("Hour")
    plt.ylabel("Average Rented Bike Count")
    plt.title("Average Bike Demand by Hour")
    plt.xticks(range(24))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_weekday_demand(df):
    order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    weekday = (
        df.groupby("Weekday")[TARGET]
        .mean()
        .reindex(order)
    )

    plt.figure(figsize=(10, 6))
    plt.bar(weekday.index, weekday.values)
    plt.xlabel("Weekday")
    plt.ylabel("Average Rented Bike Count")
    plt.title("Average Bike Demand by Weekday")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def plot_hour_weekday_demand(df):
    order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    grouped = (
        df.groupby(["Hour", "Weekday"])[TARGET]
        .mean()
        .unstack()
    )

    plt.figure(figsize=(12, 7))

    for day in order:
        plt.plot(
            grouped.index,
            grouped[day],
            label=day
        )

    plt.xlabel("Hour")
    plt.ylabel("Average Rented Bike Count")
    plt.title("Hourly Bike Demand by Weekday")
    plt.xticks(range(24))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_seasonal_demand(df):
    seasonal = df.groupby("Seasons")[TARGET].mean()

    plt.figure(figsize=(9, 6))
    plt.bar(seasonal.index, seasonal.values)
    plt.xlabel("Season")
    plt.ylabel("Average Rented Bike Count")
    plt.title("Average Bike Demand by Season")
    plt.tight_layout()
    plt.show()


def plot_temperature_demand(df):
    plt.figure(figsize=(10, 7))

    plt.scatter(
        df["Temperature(°C)"],
        df[TARGET],
        alpha=0.25
    )

    plt.xlabel("Temperature (°C)")
    plt.ylabel("Rented Bike Count")
    plt.title("Temperature vs Bike Demand")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_rainfall_demand(df):
    plt.figure(figsize=(10, 7))

    plt.scatter(
        df["Rainfall(mm)"],
        df[TARGET],
        alpha=0.3
    )

    plt.xlabel("Rainfall (mm)")
    plt.ylabel("Rented Bike Count")
    plt.title("Rainfall vs Bike Demand")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_rain_vs_no_rain(df):
    data = df.copy()

    data["Rain"] = data["Rainfall(mm)"] > 0

    grouped = (
        data.groupby("Rain")[TARGET]
        .mean()
        .reindex([False, True])
    )

    plt.figure(figsize=(8, 6))

    plt.bar(
        ["No Rain", "Rain"],
        grouped.values
    )

    plt.ylabel("Average Rented Bike Count")
    plt.title("Bike Demand During Rain vs No Rain")
    plt.tight_layout()
    plt.show()


def plot_weather_correlations(df, weather_features):
    columns = weather_features + [TARGET]

    correlation = df[columns].corr()

    plt.figure(figsize=(11, 9))

    plt.imshow(
        correlation,
        aspect="auto"
    )

    plt.colorbar()

    plt.xticks(
        range(len(columns)),
        columns,
        rotation=90
    )

    plt.yticks(
        range(len(columns)),
        columns
    )

    plt.title("Weather and Bike Demand Correlations")
    plt.tight_layout()
    plt.show()


def plot_temperature_shift(train, validation):
    plt.figure(figsize=(10, 6))

    plt.hist(
        train["Temperature(°C)"],
        bins=30,
        alpha=0.5,
        density=True,
        label="Training"
    )

    plt.hist(
        validation["Temperature(°C)"],
        bins=30,
        alpha=0.5,
        density=True,
        label="October Validation"
    )

    plt.xlabel("Temperature (°C)")
    plt.ylabel("Density")
    plt.title("Temperature Distribution: Training vs October")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_model_performance(results):
    x = np.arange(len(results))
    width = 0.35

    plt.figure(figsize=(12, 7))

    plt.bar(
        x - width / 2,
        results["MAE"],
        width,
        label="MAE"
    )

    plt.bar(
        x + width / 2,
        results["RMSE"],
        width,
        label="RMSE"
    )

    plt.xticks(
        x,
        results["Model"],
        rotation=15
    )

    plt.ylabel("Prediction Error")
    plt.title("October Validation Performance")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_predictions(
    validation,
    baseline_predictions,
    calendar_predictions,
    weather_predictions
):
    plt.figure(figsize=(14, 7))

    plt.plot(
        validation["Timestamp"],
        validation[TARGET],
        label="Actual"
    )

    plt.plot(
        validation["Timestamp"],
        baseline_predictions,
        label="Baseline",
        alpha=0.8
    )

    plt.plot(
        validation["Timestamp"],
        calendar_predictions,
        label="Calendar",
        alpha=0.8
    )

    plt.plot(
        validation["Timestamp"],
        weather_predictions,
        label="Calendar + Weather",
        alpha=0.8
    )

    plt.xlabel("Date")
    plt.ylabel("Rented Bike Count")
    plt.title("Actual vs Predicted Bike Demand")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_prediction_week(
    validation,
    baseline_predictions,
    calendar_predictions,
    weather_predictions,
    start_date="2018-10-01"
):
    start_date = np.datetime64(start_date)
    end_date = start_date + np.timedelta64(7, "D")

    mask = (
        (validation["Timestamp"] >= start_date)
        & (validation["Timestamp"] < end_date)
    )

    plt.figure(figsize=(14, 7))

    plt.plot(
        validation.loc[mask, "Timestamp"],
        validation.loc[mask, TARGET],
        label="Actual"
    )

    plt.plot(
        validation.loc[mask, "Timestamp"],
        baseline_predictions[mask],
        label="Baseline"
    )

    plt.plot(
        validation.loc[mask, "Timestamp"],
        calendar_predictions[mask],
        label="Calendar"
    )

    plt.plot(
        validation.loc[mask, "Timestamp"],
        weather_predictions[mask],
        label="Calendar + Weather"
    )

    plt.xlabel("Date")
    plt.ylabel("Rented Bike Count")
    plt.title("Actual vs Predicted Bike Demand: One Week")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, predictions, title):
    residuals = np.asarray(y_true) - np.asarray(predictions)

    plt.figure(figsize=(10, 6))

    plt.scatter(
        predictions,
        residuals,
        alpha=0.4
    )

    plt.axhline(0)

    plt.xlabel("Predicted Bike Demand")
    plt.ylabel("Residual")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()