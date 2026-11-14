from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


TARGET = "Rented Bike Count"

HOURLY_BASELINE_NAME = "Hourly Mean Baseline"
WEEKDAY_CYCLICAL_NAME = "Weekday-Only Cyclical Linear Regression"
WEEKEND_CYCLICAL_NAME = "Weekend-Only Cyclical Linear Regression"
CALENDAR_CYCLICAL_NAME = "Calendar-Only Cyclical Linear Regression"
CALENDAR_SEASONS_CYCLICAL_NAME = (
    "Calendar + Seasons Cyclical Linear Regression"
)
CALENDAR_WEATHER_CYCLICAL_NAME = (
    "Calendar + Weather Cyclical Linear Regression"
)
CALENDAR_WEATHER_SEASONS_CYCLICAL_NAME = (
    "Calendar + Weather + Seasons Cyclical Linear Regression"
)

RANDOM_FOREST_MODEL_FAMILY = "Cyclical Random Forest Regressor"
RF_CALENDAR_FEATURES_NAME = "Calendar"
RF_CALENDAR_WEATHER_FEATURES_NAME = "Calendar + Weather"
RF_CALENDAR_SEASONS_FEATURES_NAME = "Calendar + Seasons"
RF_CALENDAR_WEATHER_SEASONS_FEATURES_NAME = (
    "Calendar + Weather + Seasons"
)

RF_RESULT_NAMES = {
    RF_CALENDAR_FEATURES_NAME:
        "Calendar-Only Cyclical Random Forest Regressor",
    RF_CALENDAR_WEATHER_FEATURES_NAME:
        "Calendar + Weather Cyclical Random Forest Regressor",
    RF_CALENDAR_SEASONS_FEATURES_NAME:
        "Calendar + Seasons Cyclical Random Forest Regressor",
    RF_CALENDAR_WEATHER_SEASONS_FEATURES_NAME:
        "Calendar + Weather + Seasons Cyclical Random Forest Regressor",
}

CYCLICAL_HOUR_FEATURES = [
    "HourSin1",
    "HourCos1",
    "HourSin2",
    "HourCos2",
    "HourSin3",
    "HourCos3",
]

CATEGORICAL_FEATURES = ["Weekday", "Seasons", "Holiday"]

CALENDAR_FEATURES_WITHOUT_SEASONS = (
    CYCLICAL_HOUR_FEATURES + ["Weekday", "Holiday"]
)

CALENDAR_FEATURES_WITH_SEASONS = (
    CYCLICAL_HOUR_FEATURES + ["Weekday", "Seasons", "Holiday"]
)

WEATHER_FEATURES = [
    "Temperature(°C)",
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Dew point temperature(°C)",
    "Solar Radiation (MJ/m2)",
    "Rainfall(mm)",
    "Snowfall (cm)",
]


def fit_hourly_baseline(train):
    return train.groupby("Hour")[TARGET].mean()


def predict_hourly_baseline(model, data):
    predictions = data["Hour"].map(model)
    if predictions.isna().any():
        predictions = predictions.fillna(model.mean())
    return predictions.to_numpy()


class PeriodicDayTypeRegressor:
    """Fit distinct cyclical regressions for weekdays and weekends."""

    def __init__(self):
        self.weekday_model = LinearRegression()
        self.weekend_model = LinearRegression()

    def fit(self, data):
        weekend = data["IsWeekend"].astype(bool)
        self.weekday_model.fit(
            data.loc[~weekend, CYCLICAL_HOUR_FEATURES],
            data.loc[~weekend, TARGET],
        )
        self.weekend_model.fit(
            data.loc[weekend, CYCLICAL_HOUR_FEATURES],
            data.loc[weekend, TARGET],
        )
        return self

    def predict_weekdays(self, data):
        weekday_data = data.loc[
            ~data["IsWeekend"].astype(bool),
            CYCLICAL_HOUR_FEATURES,
        ]
        return self.weekday_model.predict(weekday_data)

    def predict_weekends(self, data):
        weekend_data = data.loc[
            data["IsWeekend"].astype(bool),
            CYCLICAL_HOUR_FEATURES,
        ]
        return self.weekend_model.predict(weekend_data)


def fit_periodic_day_type_model(train):
    return PeriodicDayTypeRegressor().fit(train)


def create_linear_model(calendar_features, include_weather=False):
    categorical = [
        feature for feature in calendar_features
        if feature in CATEGORICAL_FEATURES
    ]
    transformers = [
        (
            "calendar_categories",
            OneHotEncoder(handle_unknown="ignore", drop="first"),
            categorical,
        ),
        (
            "cyclical_hour",
            StandardScaler(),
            CYCLICAL_HOUR_FEATURES,
        ),
    ]
    if include_weather:
        transformers.append(
            ("weather", StandardScaler(), WEATHER_FEATURES)
        )

    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers)),
        ("model", LinearRegression()),
    ])


def _fit_linear_model(train, calendar_features, include_weather=False):
    features = list(calendar_features)
    if include_weather:
        features += WEATHER_FEATURES
    model = create_linear_model(calendar_features, include_weather)
    model.fit(train[features], train[TARGET])
    return model


def _predict_linear_model(
    model,
    data,
    calendar_features,
    include_weather=False,
):
    features = list(calendar_features)
    if include_weather:
        features += WEATHER_FEATURES
    return model.predict(data[features])


def fit_calendar_model(train):
    return _fit_linear_model(
        train,
        CALENDAR_FEATURES_WITHOUT_SEASONS,
    )


def fit_calendar_model_with_seasons(train):
    return _fit_linear_model(
        train,
        CALENDAR_FEATURES_WITH_SEASONS,
    )


def fit_weather_model(train):
    return _fit_linear_model(
        train,
        CALENDAR_FEATURES_WITHOUT_SEASONS,
        include_weather=True,
    )


def fit_weather_model_with_seasons(train):
    return _fit_linear_model(
        train,
        CALENDAR_FEATURES_WITH_SEASONS,
        include_weather=True,
    )


def predict_calendar_model(model, data):
    return _predict_linear_model(
        model,
        data,
        CALENDAR_FEATURES_WITHOUT_SEASONS,
    )


def predict_calendar_model_with_seasons(model, data):
    return _predict_linear_model(
        model,
        data,
        CALENDAR_FEATURES_WITH_SEASONS,
    )


def predict_weather_model(model, data):
    return _predict_linear_model(
        model,
        data,
        CALENDAR_FEATURES_WITHOUT_SEASONS,
        include_weather=True,
    )


def predict_weather_model_with_seasons(model, data):
    return _predict_linear_model(
        model,
        data,
        CALENDAR_FEATURES_WITH_SEASONS,
        include_weather=True,
    )
