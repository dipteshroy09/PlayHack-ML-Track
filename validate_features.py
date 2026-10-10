"""Validate enhanced PlayHack feature tables and observation-window isolation."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from build_features import (
    LABEL_COLUMNS,
    OBS_DAYS,
    OBS_END_EXCLUSIVE,
    OBS_START,
    _parse_dates,
)

SOURCE_SPECS = (
    ("dailyActivity_merged.csv", "ActivityDate", False, True),
    ("sleepDay_merged.csv", "SleepDay", False, True),
    ("hourlyHeartrate_merged.csv", "ActivityHour", True, True),
    ("hourlyCalories_merged.csv", "ActivityHour", True, True),
    ("hourlyIntensities_merged.csv", "ActivityHour", True, True),
    ("hourlySteps_merged.csv", "ActivityHour", True, True),
    ("training_sessions.csv", "date", False, True),
    ("weightLogInfo_merged.csv", "Date", False, False),
)


def validate_tables() -> None:
    train = pd.read_csv("train_features.csv")
    test = pd.read_csv("test_features.csv")
    feature_columns = [column for column in train if column not in LABEL_COLUMNS]

    assert feature_columns == test.columns.tolist()
    assert len(train) == train["athlete_id"].nunique() == 3000
    assert len(test) == test["athlete_id"].nunique() == 1100
    assert not set(LABEL_COLUMNS).intersection(test.columns)
    assert not train[feature_columns].isna().all().any()
    assert not test.isna().all().any()

    train_numeric = train[feature_columns].select_dtypes(include=[np.number])
    test_numeric = test.select_dtypes(include=[np.number])
    assert not np.isinf(train_numeric.to_numpy()).any()
    assert not np.isinf(test_numeric.to_numpy()).any()

    assert set(train["weight_has_record"].unique()) <= {0, 1}
    assert set(test["weight_has_record"].unique()) <= {0, 1}
    assert ((train["sess_trained_days"] + train["sess_rest_days"]) == OBS_DAYS).all()
    assert ((test["sess_trained_days"] + test["sess_rest_days"]) == OBS_DAYS).all()
    for frame in (train, test):
        assert frame["sess_max_training_streak"].between(0, OBS_DAYS).all()
        assert frame["sess_max_rest_streak"].between(0, OBS_DAYS).all()
        assert frame["sleep_efficiency_mean"].dropna().between(0, 1.2).all()
        assert frame["da_high_load_days_last7"].dropna().between(0, 7).all()

    coverage_columns = [
        column for column in test if column.endswith("hours_observed_total")
    ]
    for column in coverage_columns:
        assert train[column].dropna().between(0, OBS_DAYS * 24).all()
        assert test[column].dropna().between(0, OBS_DAYS * 24).all()

    print("FEATURE_TABLE_VALIDATION=PASS")
    print(
        f"train={train.shape}; test={test.shape}; "
        f"shared_predictors={len(feature_columns) - 1}"
    )
    print(
        "missing_rates="
        f"train:{train[feature_columns].isna().mean().mean():.6f},"
        f"test:{test.isna().mean().mean():.6f}"
    )
    print(
        "top_train_missing=",
        train[feature_columns]
        .isna()
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .round(4)
        .to_dict(),
    )


def validate_source_windows() -> None:
    for folder in ("Dataset", os.path.join("Dataset", "Test data")):
        print(f"SOURCE_WINDOW {folder}")
        for filename, date_column, is_hourly, is_dense in SOURCE_SPECS:
            raw_dates = pd.read_csv(
                os.path.join(folder, filename), usecols=[date_column]
            )[date_column].drop_duplicates()
            dates = _parse_dates(raw_dates).dropna()
            selected = dates[(dates >= OBS_START) & (dates < OBS_END_EXCLUSIVE)]
            assert len(selected) > 0
            assert selected.min() >= OBS_START
            assert selected.max() < OBS_END_EXCLUSIVE

            observation_days = (selected.dt.normalize() - OBS_START).dt.days + 1
            assert observation_days.between(1, OBS_DAYS).all()
            if is_dense:
                assert observation_days.min() == 1
                assert observation_days.max() == OBS_DAYS
            if is_hourly:
                final_day_hours = selected[observation_days == OBS_DAYS].dt.hour
                assert final_day_hours.min() == 0
                assert final_day_hours.max() == 23

            day_30_max_hour = (
                int(selected[observation_days == OBS_DAYS].dt.hour.max())
                if is_hourly
                else "n/a"
            )
            print(
                f"  {filename}: days={int(observation_days.min())}-"
                f"{int(observation_days.max())}, day30_max_hour={day_30_max_hour}"
            )
    print("SOURCE_WINDOW_VALIDATION=PASS")


if __name__ == "__main__":
    validate_tables()
    validate_source_windows()
