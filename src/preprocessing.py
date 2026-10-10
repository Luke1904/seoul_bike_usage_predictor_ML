from pathlib import Path
import pandas as pd # type: ignore


DATE_COLUMN = "Date"
HOUR_COLUMN = "Hour"
TARGET_COLUMN = "Rented Bike Count"
FUNCTIONING_COLUMN = "Functioning Day"

TRAIN_END = pd.Timestamp("2018-10-01")
VALIDATION_END = pd.Timestamp("2018-11-01")


def load_data(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path, encoding="latin1")

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


def clean_column_names(df):
    df = df.copy()

    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "Temperature(ï¿½C)": "Temperature(°C)",
        "Dew point temperature(ï¿½C)": "Dew point temperature(°C)"
    })

    return df


def validate_columns(df):
    required_columns = [
        DATE_COLUMN,
        HOUR_COLUMN,
        TARGET_COLUMN,
        FUNCTIONING_COLUMN
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def parse_dates(df):
    df = df.copy()

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        dayfirst=True,
        errors="raise"
    )

    return df


def create_calendar_features(df):
    df = df.copy()

    df["Weekday"] = df[DATE_COLUMN].dt.day_name()

    df["Timestamp"] = (
        df[DATE_COLUMN]
        + pd.to_timedelta(df[HOUR_COLUMN], unit="h")
    )

    return df


def check_missing_values(df):
    missing_values = df.isna().sum()

    missing_columns = missing_values[missing_values > 0]

    if missing_columns.empty:
        print("No missing values found.")
    else:
        print("Missing values:")
        print(missing_columns)

    return missing_values


def check_duplicates(df):
    n_duplicates = df.duplicated().sum()

    print(f"Duplicate rows: {n_duplicates}")

    return n_duplicates


def sort_chronologically(df):
    return (
        df
        .sort_values([DATE_COLUMN, HOUR_COLUMN])
        .reset_index(drop=True)
    )


def filter_functioning_days(df):
    functioning_values = (
        df[FUNCTIONING_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    non_functioning = functioning_values == "no"

    n_excluded = int(non_functioning.sum())

    df = (
        df.loc[~non_functioning]
        .copy()
        .reset_index(drop=True)
    )

    print(f"Non-functioning observations excluded: {n_excluded}")
    print(f"Observations remaining: {len(df)}")

    return df, n_excluded


def chronological_split(df):
    train = df[
        df[DATE_COLUMN] < TRAIN_END
    ].copy()

    validation = df[
        (df[DATE_COLUMN] >= TRAIN_END)
        & (df[DATE_COLUMN] < VALIDATION_END)
    ].copy()

    test = df[
        df[DATE_COLUMN] >= VALIDATION_END
    ].copy()

    return train, validation, test


def validate_split(train, validation, test):
    if train.empty:
        raise ValueError("Training set is empty.")

    if validation.empty:
        raise ValueError("Validation set is empty.")

    if test.empty:
        raise ValueError("Test set is empty.")

    if train[DATE_COLUMN].max() >= validation[DATE_COLUMN].min():
        raise ValueError("Training and validation periods overlap.")

    if validation[DATE_COLUMN].max() >= test[DATE_COLUMN].min():
        raise ValueError("Validation and test periods overlap.")


def print_split_summary(train, validation, test):
    datasets = {
        "Training": train,
        "Validation": validation,
        "Test": test
    }

    for name, data in datasets.items():
        print(
            f"{name}: "
            f"{data[DATE_COLUMN].min().date()} to "
            f"{data[DATE_COLUMN].max().date()}, "
            f"{len(data)} observations"
        )


def preprocess_data(df):
    df = clean_column_names(df)

    validate_columns(df)

    df = parse_dates(df)
    df = create_calendar_features(df)

    missing_values = check_missing_values(df)
    n_duplicates = check_duplicates(df)

    df = sort_chronologically(df)

    df, n_non_functioning = filter_functioning_days(df)

    info = {
        "missing_values": missing_values,
        "duplicate_rows": n_duplicates,
        "non_functioning_excluded": n_non_functioning,
        "observations_after_filtering": len(df)
    }

    return df, info


def prepare_dataset(file_path):
    df = load_data(file_path)

    df, info = preprocess_data(df)

    train, validation, test = chronological_split(df)

    validate_split(train, validation, test)

    print_split_summary(train, validation, test)

    return train, validation, test, info


if __name__ == "__main__":
    data_path = (
        Path(__file__).parent
        / "data"
        / "SeoulBikeData.csv"
    )

    train, validation, test, info = prepare_dataset(data_path)

    print(f"Duplicates: {info['duplicate_rows']}")
    print(
        "Non-functioning observations excluded: "
        f"{info['non_functioning_excluded']}"
    )
    print(
        "Observations after filtering: "
        f"{info['observations_after_filtering']}"
    )