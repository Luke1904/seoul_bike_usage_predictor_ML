import pandas as pd

from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore


TARGET = "Rented Bike Count"

CALENDAR_FEATURES_WITHOUT_SEASONS = [
    "Hour",
    "Weekday",
    "Holiday"
]

CALENDAR_FEATURES_WITH_SEASONS = [
    "Hour",
    "Weekday",
    "Seasons",
    "Holiday"
]

WEATHER_FEATURES = [
    "Temperature(°C)",
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Dew point temperature(°C)",
    "Solar Radiation (MJ/m2)",
    "Rainfall(mm)",
    "Snowfall (cm)"
]


def fit_hourly_baseline(train):
    return train.groupby("Hour")[TARGET].mean()


def predict_hourly_baseline(model, data):
    predictions = data["Hour"].map(model)

    if predictions.isna().any():
        fallback = model.mean()
        predictions = predictions.fillna(fallback)

    return predictions.to_numpy()


def create_calendar_model():
    preprocessor = ColumnTransformer([
        (
            "calendar",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            ),
            CALENDAR_FEATURES_WITHOUT_SEASONS
        )
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])

def create_calendar_model_with_seasons():
    preprocessor = ColumnTransformer([
        (
            "calendar",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            ),
            CALENDAR_FEATURES_WITH_SEASONS
        )
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])

def create_weather_model():
    preprocessor = ColumnTransformer([
        (
            "calendar",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            ),
            CALENDAR_FEATURES_WITHOUT_SEASONS
        ),
        (
            "weather",
            StandardScaler(),
            WEATHER_FEATURES
        )
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])

def create_weather_model_with_seasons():
    preprocessor = ColumnTransformer([
        (
            "calendar",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            ),
            CALENDAR_FEATURES_WITH_SEASONS
        ),
        (
            "weather",
            StandardScaler(),
            WEATHER_FEATURES
        )
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])

def fit_calendar_model(train):
    model = create_calendar_model()

    model.fit(
        train[CALENDAR_FEATURES_WITHOUT_SEASONS],
        train[TARGET]
    )

    return model

def fit_calendar_model_with_seasons(train):
    model = create_calendar_model_with_seasons()

    model.fit(
        train[CALENDAR_FEATURES_WITH_SEASONS],
        train[TARGET]
    )

    return model

def fit_weather_model(train):
    features = CALENDAR_FEATURES_WITHOUT_SEASONS + WEATHER_FEATURES

    model = create_weather_model()

    model.fit(
        train[features],
        train[TARGET]
    )

    return model

def fit_weather_model_with_seasons(train):
    features = CALENDAR_FEATURES_WITH_SEASONS + WEATHER_FEATURES

    model = create_weather_model_with_seasons()

    model.fit(
        train[features],
        train[TARGET]
    )

    return model

def predict_calendar_model(model, data):
    return model.predict(
        data[CALENDAR_FEATURES_WITHOUT_SEASONS]
    )

def predict_calendar_model_with_seasons(model, data):
    return model.predict(
        data[CALENDAR_FEATURES_WITH_SEASONS]
    )

def predict_weather_model(model, data):
    features = CALENDAR_FEATURES_WITHOUT_SEASONS + WEATHER_FEATURES

    return model.predict(
        data[features]
    )

def predict_weather_model_with_seasons(model, data):
    features = CALENDAR_FEATURES_WITH_SEASONS + WEATHER_FEATURES

    return model.predict(
        data[features]
    )