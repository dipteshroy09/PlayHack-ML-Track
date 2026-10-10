"""Build one-row-per-athlete PlayHack features from observation days 1-30 only.

The same build_features() function is used for train and test. All temporal
features are athlete-relative and computed strictly inside the observation
window; labels are merged only after training features are complete.

Outputs:
    train_features.csv  (features + three training labels)
    test_features.csv   (the exact same ordered features, no labels)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

OBS_START = pd.Timestamp("2026-01-05")
OBS_DAYS = 30
OBS_END_EXCLUSIVE = OBS_START + pd.Timedelta(days=OBS_DAYS)
RECENT_3_START = 28
RECENT_7_START = 24
RECENT_14_START = 17
PREVIOUS_7_START = 17
PREVIOUS_7_END = 23
PREVIOUS_21_START = 3
PREVIOUS_21_END = 23
EPS = 1e-6
SESSION_TYPES = ("gym", "practice", "scrimmage", "other")
LABEL_COLUMNS = ["injured_in_risk_window", "onset_day_offset", "recovery_duration"]


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse the two date formats used by the raw files via vectorized paths."""
    non_null = series.dropna()
    sample = str(non_null.iloc[0]) if len(non_null) else ""
    if "/" in sample:
        parsed = pd.to_datetime(
            series, format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
        )
    else:
        parsed = pd.to_datetime(series, format="ISO8601", errors="coerce")
    if parsed.isna().any():
        missing = parsed.isna() & series.notna()
        if missing.any():
            parsed.loc[missing] = pd.to_datetime(series.loc[missing], errors="coerce")
    return parsed


def _slice_window(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Slice to the half-open 30-day observation interval before aggregation."""
    datetimes = _parse_dates(df[date_col])
    mask = (datetimes >= OBS_START) & (datetimes < OBS_END_EXCLUSIVE)
    out = df.loc[mask].copy()
    out["_timestamp"] = datetimes.loc[mask]
    out["_date"] = out["_timestamp"].dt.normalize()
    out["_obs_day"] = (out["_date"] - OBS_START).dt.days + 1
    out["_hour"] = out["_timestamp"].dt.hour
    if not out["_obs_day"].between(1, OBS_DAYS).all():
        raise AssertionError(f"{date_col} produced rows outside observation days 1-30")
    return out


def _slope(group: pd.DataFrame, metric: str) -> float:
    clean = group[["_obs_day", metric]].dropna()
    if len(clean) < 2 or clean["_obs_day"].nunique() < 2:
        return np.nan
    return float(np.polyfit(clean["_obs_day"], clean[metric], 1)[0])


def _window_stat(
    daily: pd.DataFrame,
    metric: str,
    start: int,
    end: int,
    stat: str = "mean",
) -> pd.Series:
    selected = daily[daily["_obs_day"].between(start, end)]
    grouped = selected.groupby("athlete_id")[metric]
    if stat == "mean":
        return grouped.mean()
    if stat == "std":
        return grouped.std()
    if stat == "sum":
        return grouped.sum()
    raise ValueError(f"Unsupported stat: {stat}")


def _temporal_summary(
    daily: pd.DataFrame,
    metrics: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Summarize daily signals using recent windows and a disjoint baseline.

    Recent week = days 24-30. Prior baseline = days 3-23 (the immediately
    preceding 21 days). Ratios and z-shifts therefore compare disjoint periods.
    """
    athlete_index = pd.Index(
        sorted(daily["athlete_id"].dropna().unique()), name="athlete_id"
    )
    out = pd.DataFrame(index=athlete_index)
    grouped = daily.groupby("athlete_id", sort=True)

    for metric in metrics:
        mean30 = grouped[metric].mean().reindex(athlete_index)
        std30 = grouped[metric].std().reindex(athlete_index)
        last3 = _window_stat(daily, metric, RECENT_3_START, OBS_DAYS).reindex(athlete_index)
        last7 = _window_stat(daily, metric, RECENT_7_START, OBS_DAYS).reindex(athlete_index)
        last14 = _window_stat(daily, metric, RECENT_14_START, OBS_DAYS).reindex(athlete_index)
        prev7 = _window_stat(
            daily, metric, PREVIOUS_7_START, PREVIOUS_7_END
        ).reindex(athlete_index)
        prev21 = _window_stat(
            daily, metric, PREVIOUS_21_START, PREVIOUS_21_END
        ).reindex(athlete_index)
        prev21_std = _window_stat(
            daily, metric, PREVIOUS_21_START, PREVIOUS_21_END, stat="std"
        ).reindex(athlete_index)

        base = f"{prefix}{metric}"
        out[f"{base}_cv30"] = std30 / (mean30.abs() + EPS)
        out[f"{base}_slope30"] = grouped.apply(
            lambda group, m=metric: _slope(group, m), include_groups=False
        ).reindex(athlete_index)
        out[f"{base}_mean_last3"] = last3
        out[f"{base}_mean_last7"] = last7
        out[f"{base}_mean_last14"] = last14
        out[f"{base}_mean_prev7"] = prev7
        out[f"{base}_mean_prev21"] = prev21
        out[f"{base}_delta7_prev7"] = last7 - prev7
        out[f"{base}_delta7_prev21"] = last7 - prev21
        out[f"{base}_ratio7_prev21"] = last7 / prev21.replace(0, np.nan)
        out[f"{base}_zshift7_prev21"] = (last7 - prev21) / (
            prev21_std.abs() + EPS
        )
    return out


def _assert_block(block: pd.DataFrame, name: str) -> None:
    if block.index.name != "athlete_id":
        raise AssertionError(f"{name} index must be named athlete_id")
    if not block.index.is_unique:
        raise AssertionError(f"{name} has duplicate athlete rows")
    if set(block.columns) & set(LABEL_COLUMNS):
        raise AssertionError(f"{name} unexpectedly contains labels")


# --- Per-source feature builders -------------------------------------------
def feat_daily_activity(folder: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(folder, "dailyActivity_merged.csv"))
    df = _slice_window(df, "ActivityDate").rename(columns={"Id": "athlete_id"})
    df = df.sort_values(["athlete_id", "_obs_day"])

    df["active_minutes"] = (
        df["VeryActiveMinutes"]
        + df["FairlyActiveMinutes"]
        + df["LightlyActiveMinutes"]
    )
    df["moderate_vigorous_minutes"] = (
        df["VeryActiveMinutes"] + df["FairlyActiveMinutes"]
    )
    df["vigorous_share"] = df["VeryActiveMinutes"] / (
        df["active_minutes"] + EPS
    )
    df["sedentary_active_ratio"] = df["SedentaryMinutes"] / (
        df["active_minutes"] + 1.0
    )

    original_metrics = [
        "TotalSteps",
        "TotalDistance",
        "VeryActiveMinutes",
        "FairlyActiveMinutes",
        "LightlyActiveMinutes",
        "SedentaryMinutes",
        "Calories",
    ]
    grouped = df.groupby("athlete_id")
    aggregate = grouped[original_metrics].agg(["mean", "std", "min", "max", "sum"])
    aggregate.columns = [f"da_{metric}_{stat}" for metric, stat in aggregate.columns]

    extra = pd.DataFrame(index=aggregate.index)
    extra["da_active_minutes_mean"] = grouped["active_minutes"].mean()
    extra["da_days_recorded"] = grouped.size()

    temporal = _temporal_summary(
        df,
        [
            "TotalSteps",
            "Calories",
            "active_minutes",
            "moderate_vigorous_minutes",
            "vigorous_share",
            "sedentary_active_ratio",
        ],
        "da_",
    )

    # Preserve established feature names while correcting their definitions.
    extra["da_steps_trend"] = temporal["da_TotalSteps_slope30"]
    extra["da_calories_trend"] = temporal["da_Calories_slope30"]
    extra["da_steps_acwr"] = temporal["da_TotalSteps_ratio7_prev21"]

    # Athlete-relative high-load counts in the final week.
    prior = df[df["_obs_day"].between(PREVIOUS_21_START, PREVIOUS_21_END)]
    prior_stats = prior.groupby("athlete_id")["TotalSteps"].agg(["mean", "std"])
    recent = df[df["_obs_day"].between(RECENT_7_START, OBS_DAYS)].join(
        prior_stats, on="athlete_id"
    )
    recent["is_high_load"] = recent["TotalSteps"] > (
        recent["mean"] + recent["std"].fillna(0)
    )
    extra["da_high_load_days_last7"] = (
        recent.groupby("athlete_id")["is_high_load"].sum().reindex(extra.index)
    )
    return aggregate.join(extra).join(temporal)


def feat_sleep(folder: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(folder, "sleepDay_merged.csv"))
    df = _slice_window(df, "SleepDay").rename(columns={"Id": "athlete_id"})
    df["sleep_efficiency"] = df["TotalMinutesAsleep"] / df[
        "TotalTimeInBed"
    ].replace(0, np.nan)
    df["awake_in_bed_minutes"] = (
        df["TotalTimeInBed"] - df["TotalMinutesAsleep"]
    )
    grouped = df.groupby("athlete_id")
    out = pd.DataFrame(index=grouped.size().index)
    out["sleep_minutes_mean"] = grouped["TotalMinutesAsleep"].mean()
    out["sleep_minutes_std"] = grouped["TotalMinutesAsleep"].std()
    out["sleep_minutes_min"] = grouped["TotalMinutesAsleep"].min()
    out["sleep_efficiency_mean"] = grouped["sleep_efficiency"].mean()
    out["sleep_timeinbed_mean"] = grouped["TotalTimeInBed"].mean()
    out["sleep_days_recorded"] = grouped.size()

    temporal = _temporal_summary(
        df,
        ["TotalMinutesAsleep", "sleep_efficiency", "awake_in_bed_minutes"],
        "sleep_",
    )

    prior = df[df["_obs_day"].between(PREVIOUS_21_START, PREVIOUS_21_END)]
    prior_stats = prior.groupby("athlete_id")["TotalMinutesAsleep"].agg(["mean", "std"])
    recent = df[df["_obs_day"].between(RECENT_7_START, OBS_DAYS)].join(
        prior_stats, on="athlete_id"
    )
    recent["relative_short"] = recent["TotalMinutesAsleep"] < (
        recent["mean"] - recent["std"].fillna(0)
    )
    recent["fixed_short"] = recent["TotalMinutesAsleep"] < 420
    recent["sleep_debt"] = (
        recent["mean"] - recent["TotalMinutesAsleep"]
    ).clip(lower=0)
    out["sleep_relative_short_days_last7"] = recent.groupby("athlete_id")[
        "relative_short"
    ].sum().reindex(out.index)
    out["sleep_under_420_days_last7"] = recent.groupby("athlete_id")[
        "fixed_short"
    ].sum().reindex(out.index)
    out["sleep_debt_minutes_last7"] = recent.groupby("athlete_id")[
        "sleep_debt"
    ].sum().reindex(out.index)
    return out.join(temporal)


def feat_heartrate(folder: str) -> pd.DataFrame:
    df = pd.read_csv(
        os.path.join(folder, "hourlyHeartrate_merged.csv"),
        usecols=["Id", "ActivityHour", "AvgHeartRate", "MinHeartRate", "MaxHeartRate"],
    )
    df = _slice_window(df, "ActivityHour").rename(columns={"Id": "athlete_id"})
    df["hr_range"] = df["MaxHeartRate"] - df["MinHeartRate"]
    grouped = df.groupby("athlete_id")
    out = pd.DataFrame(index=grouped.size().index)
    out["hr_avg_mean"] = grouped["AvgHeartRate"].mean()
    out["hr_avg_std"] = grouped["AvgHeartRate"].std()
    out["hr_max_max"] = grouped["MaxHeartRate"].max()
    out["hr_resting_proxy"] = grouped["AvgHeartRate"].quantile(0.05)
    out["hr_range_mean"] = grouped["hr_range"].mean()
    out["hr_hours_observed_total"] = grouped.size()

    daily_group = df.groupby(["athlete_id", "_obs_day"])
    daily = daily_group.agg(
        daily_avg=("AvgHeartRate", "mean"),
        daily_max=("MaxHeartRate", "max"),
        daily_range=("hr_range", "mean"),
        hours_observed=("AvgHeartRate", "size"),
    ).reset_index()
    daily_rest = daily_group["AvgHeartRate"].quantile(0.05).rename("daily_rest")
    daily = daily.join(daily_rest, on=["athlete_id", "_obs_day"])
    night = (
        df[df["_hour"].between(0, 5)]
        .groupby(["athlete_id", "_obs_day"])["AvgHeartRate"]
        .mean()
        .rename("night_avg")
    )
    daily = daily.join(night, on=["athlete_id", "_obs_day"])

    temporal = _temporal_summary(
        daily, ["daily_avg", "daily_rest", "night_avg"], "hr_"
    )
    daily_coverage = daily.groupby("athlete_id")["hours_observed"]
    out["hr_hours_observed_daily_mean"] = daily_coverage.mean()
    out["hr_hours_observed_daily_min"] = daily_coverage.min()
    out["hr_missing_hours_total"] = OBS_DAYS * 24 - out["hr_hours_observed_total"]
    return out.join(temporal)


def feat_hourly_generic(folder: str, fname: str, col: str, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(
        os.path.join(folder, fname), usecols=["Id", "ActivityHour", col]
    )
    df = _slice_window(df, "ActivityHour").rename(columns={"Id": "athlete_id"})
    grouped = df.groupby("athlete_id")[col]
    athlete_index = pd.Index(
        sorted(df["athlete_id"].unique()), name="athlete_id"
    )
    out = pd.DataFrame(index=athlete_index)
    out[f"{prefix}_hourly_mean"] = grouped.mean()
    out[f"{prefix}_hourly_std"] = grouped.std()
    out[f"{prefix}_hourly_max"] = grouped.max()

    df["active_hour"] = (df[col] > 0).astype(int)
    df["evening_value"] = np.where(df["_hour"].between(18, 23), df[col], 0)
    daily = df.groupby(["athlete_id", "_obs_day"]).agg(
        daily_total=(col, "sum"),
        daily_mean=(col, "mean"),
        daily_max=(col, "max"),
        active_hours=("active_hour", "sum"),
        hours_observed=(col, "size"),
        evening_total=("evening_value", "sum"),
    ).reset_index()
    daily["daily_total_adjusted"] = np.where(
        daily["hours_observed"] >= 18,
        daily["daily_total"] * 24.0 / daily["hours_observed"],
        np.nan,
    )
    daily["evening_share"] = daily["evening_total"] / (
        daily["daily_total"] + EPS
    )

    temporal = _temporal_summary(
        daily, ["daily_total_adjusted", "active_hours"], f"{prefix}_"
    )
    daily_group = daily.groupby("athlete_id")
    out[f"{prefix}_hours_observed_total"] = daily_group["hours_observed"].sum()
    out[f"{prefix}_hours_observed_daily_mean"] = daily_group[
        "hours_observed"
    ].mean()
    out[f"{prefix}_hours_observed_daily_min"] = daily_group[
        "hours_observed"
    ].min()
    out[f"{prefix}_evening_share_mean"] = daily_group["evening_share"].mean()
    out[f"{prefix}_evening_share_last7"] = _window_stat(
        daily, "evening_share", RECENT_7_START, OBS_DAYS
    ).reindex(out.index)
    out[f"{prefix}_evening_share_delta7_prev21"] = (
        out[f"{prefix}_evening_share_last7"]
        - _window_stat(
            daily, "evening_share", PREVIOUS_21_START, PREVIOUS_21_END
        ).reindex(out.index)
    )
    return out.join(temporal)


def _max_run(values: np.ndarray, target: bool) -> int:
    longest = current = 0
    for value in values:
        if bool(value) is target:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def feat_sessions(folder: str, athlete_ids: pd.Index) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(folder, "training_sessions.csv"))
    df = _slice_window(df, "date")
    duration = np.where(
        df["end_hour"] >= df["start_hour"],
        df["end_hour"] - df["start_hour"],
        df["end_hour"] + 24 - df["start_hour"],
    )
    df["duration_h"] = np.clip(duration, 0, 8)
    df["session_type_fixed"] = df["sport_session_type"].where(
        df["sport_session_type"].isin(SESSION_TYPES[:-1]), "other"
    )

    daily_observed = df.groupby(["athlete_id", "_obs_day"]).agg(
        duration_h=("duration_h", "sum"),
        session_count=("session_id", "count"),
    )
    panel_index = pd.MultiIndex.from_product(
        [athlete_ids, range(1, OBS_DAYS + 1)],
        names=["athlete_id", "_obs_day"],
    )
    daily = daily_observed.reindex(panel_index, fill_value=0).reset_index()
    daily["trained"] = daily["session_count"] > 0

    temporal = _temporal_summary(
        daily, ["duration_h", "session_count"], "sess_"
    )
    out = pd.DataFrame(index=athlete_ids.copy())
    out.index.name = "athlete_id"
    daily_group = daily.groupby("athlete_id")
    out["sess_count"] = daily_group["session_count"].sum()
    out["sess_total_hours"] = daily_group["duration_h"].sum()
    out["sess_mean_hours"] = (
        out["sess_total_hours"] / out["sess_count"].replace(0, np.nan)
    )
    out["sess_trained_days"] = daily_group["trained"].sum()
    out["sess_rest_days"] = OBS_DAYS - out["sess_trained_days"]
    out["sess_back_to_back_days"] = daily_group["trained"].apply(
        lambda series: int((series.to_numpy()[1:] & series.to_numpy()[:-1]).sum())
    )
    out["sess_max_training_streak"] = daily_group["trained"].apply(
        lambda series: _max_run(series.to_numpy(), True)
    )
    out["sess_max_rest_streak"] = daily_group["trained"].apply(
        lambda series: _max_run(series.to_numpy(), False)
    )
    daily_hours_mean = daily_group["duration_h"].mean()
    daily_hours_std = daily_group["duration_h"].std()
    out["sess_monotony"] = daily_hours_mean / (daily_hours_std + EPS)
    out["sess_strain"] = out["sess_total_hours"] * out["sess_monotony"]

    counts = df.pivot_table(
        index="athlete_id",
        columns="session_type_fixed",
        values="session_id",
        aggfunc="count",
        fill_value=0,
    ).reindex(index=athlete_ids, columns=SESSION_TYPES, fill_value=0)
    hours = df.pivot_table(
        index="athlete_id",
        columns="session_type_fixed",
        values="duration_h",
        aggfunc="sum",
        fill_value=0,
    ).reindex(index=athlete_ids, columns=SESSION_TYPES, fill_value=0)
    recent = df[df["_obs_day"].between(RECENT_7_START, OBS_DAYS)]
    recent_counts = recent.pivot_table(
        index="athlete_id",
        columns="session_type_fixed",
        values="session_id",
        aggfunc="count",
        fill_value=0,
    ).reindex(index=athlete_ids, columns=SESSION_TYPES, fill_value=0)
    for session_type in SESSION_TYPES:
        out[f"sess_type_{session_type}"] = counts[session_type]
        out[f"sess_type_{session_type}_hours"] = hours[session_type]
        out[f"sess_type_{session_type}_count_last7"] = recent_counts[session_type]

    out = out.join(temporal)
    out["sess_hours_acwr"] = out["sess_duration_h_ratio7_prev21"]
    return out


def feat_weight(folder: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(folder, "weightLogInfo_merged.csv"))
    df = _slice_window(df, "Date").rename(columns={"Id": "athlete_id"})
    latest = (
        df.sort_values(["athlete_id", "_timestamp"])
        .groupby("athlete_id")
        .tail(1)
        .set_index("athlete_id")
    )
    out = pd.DataFrame(index=latest.index)
    out["weight_latest_kg"] = latest["WeightKg"]
    out["weight_latest_bmi"] = latest["BMI"]
    out["weight_latest_fat"] = latest["Fat"]
    out["weight_has_record"] = 1
    return out


def feat_metadata(folder: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(folder, "athlete_metadata.csv"))
    if not df["athlete_id"].is_unique:
        raise AssertionError("athlete_metadata.csv contains duplicate athlete_id values")
    keep = [
        "age",
        "height_cm",
        "weight_kg_baseline",
        "years_playing",
        "prior_season_injury_count",
        "sport",
        "gender",
        "dominant_side",
        "position",
        "team_id",
    ]
    return df.set_index("athlete_id")[keep].copy()


# --- Master builder ---------------------------------------------------------
def build_features(folder: str) -> pd.DataFrame:
    meta = feat_metadata(folder)
    blocks = [
        feat_daily_activity(folder),
        feat_sleep(folder),
        feat_heartrate(folder),
        feat_hourly_generic(
            folder, "hourlyCalories_merged.csv", "Calories", "cal"
        ),
        feat_hourly_generic(
            folder, "hourlyIntensities_merged.csv", "TotalIntensity", "intensity"
        ),
        feat_hourly_generic(folder, "hourlySteps_merged.csv", "StepTotal", "hsteps"),
        feat_sessions(folder, meta.index),
        feat_weight(folder),
    ]

    features = meta.copy()
    for index, block in enumerate(blocks, start=1):
        _assert_block(block, f"feature block {index}")
        duplicate_columns = features.columns.intersection(block.columns)
        if len(duplicate_columns):
            raise AssertionError(f"Duplicate feature columns: {duplicate_columns.tolist()}")
        features = features.join(block, how="left")

    features["weight_has_record"] = features["weight_has_record"].fillna(0).astype(int)

    # Cross-source stress interactions.
    features["interaction_load_minus_sleep"] = (
        features["da_TotalSteps_zshift7_prev21"]
        - features["sleep_TotalMinutesAsleep_zshift7_prev21"]
    )
    features["interaction_load_plus_resting_hr"] = (
        features["da_TotalSteps_zshift7_prev21"]
        + features["hr_daily_rest_zshift7_prev21"]
    )
    features["interaction_session_minus_sleep"] = (
        features["sess_duration_h_zshift7_prev21"]
        - features["sleep_TotalMinutesAsleep_zshift7_prev21"]
    )
    features["interaction_active_sleep_debt"] = (
        features["da_active_minutes_zshift7_prev21"]
        * features["sleep_debt_minutes_last7"]
    )

    if len(features) != len(meta) or not features.index.equals(meta.index):
        raise AssertionError("Feature assembly changed the metadata athlete index")
    result = features.reset_index()
    if not result["athlete_id"].is_unique:
        raise AssertionError("Final feature table has duplicate athletes")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-folder", default="Dataset")
    parser.add_argument("--test-folder", default=os.path.join("Dataset", "Test data"))
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    print("Building enhanced TRAIN features from", args.train_folder)
    train_features = build_features(args.train_folder)
    labels = pd.read_csv(os.path.join(args.train_folder, "train_labels.csv"))
    train = train_features.merge(
        labels, on="athlete_id", how="left", validate="one_to_one"
    )
    if train[LABEL_COLUMNS].isna().all(axis=1).any():
        raise AssertionError("A training athlete is missing every label")

    print("Building enhanced TEST features from", args.test_folder)
    test = build_features(args.test_folder)

    if train_features.columns.tolist() != test.columns.tolist():
        raise AssertionError("Train/test feature columns or order differ")
    if len(train_features) != train_features["athlete_id"].nunique():
        raise AssertionError("Train features are not one row per athlete")
    if len(test) != test["athlete_id"].nunique():
        raise AssertionError("Test features are not one row per athlete")

    train_path = os.path.join(args.outdir, "train_features.csv")
    test_path = os.path.join(args.outdir, "test_features.csv")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"  wrote {train_path}: {train.shape[0]} rows, {train.shape[1]} cols")
    print(f"  wrote {test_path}: {test.shape[0]} rows, {test.shape[1]} cols")
    print("\n--- enhanced feature sanity checks ---")
    print("ordered shared feature columns:", len(test.columns))
    print("train/test columns identical:", train_features.columns.tolist() == test.columns.tolist())
    print("train athletes unique:", train_features["athlete_id"].nunique())
    print("test athletes unique:", test["athlete_id"].nunique())
    print("train label injury rate:", round(train["injured_in_risk_window"].mean(), 3))


if __name__ == "__main__":
    main()
