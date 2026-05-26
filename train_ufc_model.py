from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


DATA_PATH = Path("UFC_full_data_silver.csv")
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "ufc_extra_trees_calibrated.joblib"
METRICS_PATH = ARTIFACT_DIR / "ufc_extra_trees_calibrated_metrics.json"

HIST_STAT_COLUMNS = {
    "knockdowns": "kd_diff_sum",
    "sig_strikes_succ": "sig_strike_diff_sum",
    "total_strikes_succ": "total_strike_diff_sum",
    "takedown_succ": "takedown_diff_sum",
    "submission_att": "submission_att_diff_sum",
    "ctrl_time_sec": "ctrl_time_diff_sum",
}

FEATURE_COLUMNS = [
    "age_diff_years",
    "height_diff_cm",
    "reach_diff_cm",
    "prior_fights_diff",
    "prior_win_rate_diff",
    "prior_finish_win_rate_diff",
    "avg_sig_strike_diff",
    "avg_total_strike_diff",
    "avg_takedown_diff",
    "avg_submission_att_diff",
    "avg_ctrl_time_min_diff",
    "avg_knockdown_diff",
    "same_stance",
    "title_fight",
    "scheduled_rounds",
]

DIFF_FEATURE_COLUMNS = [
    col
    for col in FEATURE_COLUMNS
    if col.endswith("_diff")
    or col.endswith("_diff_years")
    or col.endswith("_diff_cm")
]

FIGHTER_FEATURE_COLUMNS = [
    "age_years",
    "height_cm",
    "reach_cm",
    "prior_fights",
    "prior_win_rate",
    "prior_finish_win_rate",
    "avg_sig_strike_diff",
    "avg_total_strike_diff",
    "avg_takedown_diff",
    "avg_submission_att_diff",
    "avg_ctrl_time_min_diff",
    "avg_knockdown_diff",
]


def fighter_id(df: pd.DataFrame, side: int) -> pd.Series:
    profile_url = df[f"f_{side}_fighter_url"].fillna("").astype(str)
    fight_url = df[f"f_{side}_url"].fillna("").astype(str)
    name = df[f"f_{side}_name"].fillna("").astype(str)
    return np.select(
        [profile_url.ne(""), fight_url.ne("")],
        [profile_url, fight_url],
        default=name,
    )


def parse_finish_win(df: pd.DataFrame, side: int) -> pd.Series:
    won = df["winner"].eq(df[f"f_{side}_name"])
    result = df["result"].fillna("").astype(str)
    decision = result.str.contains("DEC", case=False, regex=False)
    no_contest = result.str.contains("NC", case=False, regex=False)
    return won & ~decision & ~no_contest


def load_clean_fights(data_path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(data_path, low_memory=False)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    valid_target = df["winner"].eq(df["f_1_name"]) | df["winner"].eq(df["f_2_name"])
    df = df.loc[valid_target & df["event_date"].notna()].copy()
    df = df.sort_values(["event_date", "event_name", "fight_url"]).reset_index(drop=True)
    df["target_f1_win"] = df["winner"].eq(df["f_1_name"]).astype(int)
    df["f_1_id"] = fighter_id(df, 1)
    df["f_2_id"] = fighter_id(df, 2)
    return df


def make_long_history(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for side, opp in [(1, 2), (2, 1)]:
        side_data = pd.DataFrame(
            {
                "event_date": df["event_date"],
                "fight_url": df["fight_url"],
                "fighter_id": fighter_id(df, side),
                "win": df["winner"].eq(df[f"f_{side}_name"]).astype(float),
                "finish_win": parse_finish_win(df, side).astype(float),
            }
        )

        for stat_name, out_name in HIST_STAT_COLUMNS.items():
            fighter_stat = pd.to_numeric(
                df[f"f_{side}_{stat_name}"], errors="coerce"
            ).fillna(0.0)
            opponent_stat = pd.to_numeric(
                df[f"f_{opp}_{stat_name}"], errors="coerce"
            ).fillna(0.0)
            side_data[out_name] = fighter_stat - opponent_stat

        frames.append(side_data)

    return pd.concat(frames, ignore_index=True)


def build_historical_features(df: pd.DataFrame, include_current_date: bool) -> pd.DataFrame:
    long_history = make_long_history(df)
    sum_columns = list(HIST_STAT_COLUMNS.values())
    date_level = (
        long_history.groupby(["fighter_id", "event_date"], as_index=False)
        .agg(
            prior_fights=("fight_url", "count"),
            prior_wins=("win", "sum"),
            prior_finish_wins=("finish_win", "sum"),
            **{col: (col, "sum") for col in sum_columns},
        )
        .sort_values(["fighter_id", "event_date"])
    )

    cumulative_columns = ["prior_fights", "prior_wins", "prior_finish_wins"] + sum_columns
    for col in cumulative_columns:
        date_level[col] = date_level.groupby("fighter_id")[col].cumsum()
        if not include_current_date:
            date_level[col] = date_level[col] - date_level.groupby("fighter_id")[col].diff().fillna(date_level[col])

    return date_level


def add_side_history(base: pd.DataFrame, history: pd.DataFrame, side: int) -> pd.DataFrame:
    history_cols = [
        col for col in history.columns if col not in {"fighter_id", "event_date"}
    ]
    renamed = history.rename(columns={col: f"f_{side}_{col}" for col in history_cols})
    return base.merge(
        renamed,
        how="left",
        left_on=[f"f_{side}_id", "event_date"],
        right_on=["fighter_id", "event_date"],
    ).drop(columns=["fighter_id"])


def safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def safe_avg(sum_values: pd.Series, count_values: pd.Series) -> pd.Series:
    return safe_rate(sum_values, count_values)


def add_side_features(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    for side in [1, 2]:
        fights = base[f"f_{side}_prior_fights"].fillna(0.0)
        base[f"f_{side}_prior_win_rate"] = safe_rate(
            base[f"f_{side}_prior_wins"].fillna(0.0), fights
        )
        base[f"f_{side}_prior_finish_win_rate"] = safe_rate(
            base[f"f_{side}_prior_finish_wins"].fillna(0.0), fights
        )
        base[f"f_{side}_avg_sig_strike_diff"] = safe_avg(
            base[f"f_{side}_sig_strike_diff_sum"].fillna(0.0), fights
        )
        base[f"f_{side}_avg_total_strike_diff"] = safe_avg(
            base[f"f_{side}_total_strike_diff_sum"].fillna(0.0), fights
        )
        base[f"f_{side}_avg_takedown_diff"] = safe_avg(
            base[f"f_{side}_takedown_diff_sum"].fillna(0.0), fights
        )
        base[f"f_{side}_avg_submission_att_diff"] = safe_avg(
            base[f"f_{side}_submission_att_diff_sum"].fillna(0.0), fights
        )
        base[f"f_{side}_avg_ctrl_time_min_diff"] = safe_avg(
            base[f"f_{side}_ctrl_time_diff_sum"].fillna(0.0) / 60.0, fights
        )
        base[f"f_{side}_avg_knockdown_diff"] = safe_avg(
            base[f"f_{side}_kd_diff_sum"].fillna(0.0), fights
        )

        dob = pd.to_datetime(base[f"f_{side}_fighter_dob"], errors="coerce")
        base[f"f_{side}_age_years"] = (base["event_date"] - dob).dt.days / 365.25

    return base


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    history = build_historical_features(df, include_current_date=False)
    base = add_side_history(df, history, 1)
    base = add_side_history(base, history, 2)
    base = add_side_features(base)

    return pd.DataFrame(
        {
            "event_date": base["event_date"],
            "event_name": base["event_name"],
            "fight_url": base["fight_url"],
            "f_1_name": base["f_1_name"],
            "f_2_name": base["f_2_name"],
            "winner": base["winner"],
            "target_f1_win": base["target_f1_win"],
            "age_diff_years": base["f_1_age_years"] - base["f_2_age_years"],
            "height_diff_cm": pd.to_numeric(
                base["f_1_fighter_height_cm"], errors="coerce"
            )
            - pd.to_numeric(base["f_2_fighter_height_cm"], errors="coerce"),
            "reach_diff_cm": pd.to_numeric(
                base["f_1_fighter_reach_cm"], errors="coerce"
            )
            - pd.to_numeric(base["f_2_fighter_reach_cm"], errors="coerce"),
            "prior_fights_diff": base["f_1_prior_fights"].fillna(0.0)
            - base["f_2_prior_fights"].fillna(0.0),
            "prior_win_rate_diff": base["f_1_prior_win_rate"]
            - base["f_2_prior_win_rate"],
            "prior_finish_win_rate_diff": base["f_1_prior_finish_win_rate"]
            - base["f_2_prior_finish_win_rate"],
            "avg_sig_strike_diff": base["f_1_avg_sig_strike_diff"]
            - base["f_2_avg_sig_strike_diff"],
            "avg_total_strike_diff": base["f_1_avg_total_strike_diff"]
            - base["f_2_avg_total_strike_diff"],
            "avg_takedown_diff": base["f_1_avg_takedown_diff"]
            - base["f_2_avg_takedown_diff"],
            "avg_submission_att_diff": base["f_1_avg_submission_att_diff"]
            - base["f_2_avg_submission_att_diff"],
            "avg_ctrl_time_min_diff": base["f_1_avg_ctrl_time_min_diff"]
            - base["f_2_avg_ctrl_time_min_diff"],
            "avg_knockdown_diff": base["f_1_avg_knockdown_diff"]
            - base["f_2_avg_knockdown_diff"],
            "same_stance": (
                base["f_1_fighter_stance"].fillna("").astype(str)
                == base["f_2_fighter_stance"].fillna("").astype(str)
            ).astype(int),
            "title_fight": base["title_fight"].astype(bool).astype(int),
            "scheduled_rounds": pd.to_numeric(base["num_rounds"], errors="coerce"),
        }
    )


def build_fighter_table(df: pd.DataFrame) -> pd.DataFrame:
    reference_date = df["event_date"].max()
    long_history = make_long_history(df)
    sum_columns = list(HIST_STAT_COLUMNS.values())
    totals = (
        long_history.groupby("fighter_id", as_index=False)
        .agg(
            prior_fights=("fight_url", "count"),
            prior_wins=("win", "sum"),
            prior_finish_wins=("finish_win", "sum"),
            **{col: (col, "sum") for col in sum_columns},
        )
    )
    fights = totals["prior_fights"].replace(0, np.nan)
    totals["prior_win_rate"] = totals["prior_wins"] / fights
    totals["prior_finish_win_rate"] = totals["prior_finish_wins"] / fights
    totals["avg_sig_strike_diff"] = totals["sig_strike_diff_sum"] / fights
    totals["avg_total_strike_diff"] = totals["total_strike_diff_sum"] / fights
    totals["avg_takedown_diff"] = totals["takedown_diff_sum"] / fights
    totals["avg_submission_att_diff"] = totals["submission_att_diff_sum"] / fights
    totals["avg_ctrl_time_min_diff"] = totals["ctrl_time_diff_sum"] / 60.0 / fights
    totals["avg_knockdown_diff"] = totals["kd_diff_sum"] / fights

    profile_frames = []
    for side in [1, 2]:
        profile_frames.append(
            pd.DataFrame(
                {
                    "event_date": df["event_date"],
                    "fighter_id": fighter_id(df, side),
                    "name": df[f"f_{side}_name"],
                    "height_cm": pd.to_numeric(
                        df[f"f_{side}_fighter_height_cm"], errors="coerce"
                    ),
                    "reach_cm": pd.to_numeric(
                        df[f"f_{side}_fighter_reach_cm"], errors="coerce"
                    ),
                    "stance": df[f"f_{side}_fighter_stance"],
                    "dob": pd.to_datetime(
                        df[f"f_{side}_fighter_dob"], errors="coerce"
                    ),
                }
            )
        )

    profiles = (
        pd.concat(profile_frames, ignore_index=True)
        .sort_values(["fighter_id", "event_date"])
        .drop_duplicates("fighter_id", keep="last")
    )
    profiles["age_years"] = (reference_date - profiles["dob"]).dt.days / 365.25

    fighter_table = profiles.merge(totals, on="fighter_id", how="left")
    fighter_table = fighter_table[
        ["fighter_id", "name", "stance", *FIGHTER_FEATURE_COLUMNS]
    ].copy()
    fighter_table = fighter_table.sort_values("name").reset_index(drop=True)
    return fighter_table


def build_matchup_features(
    fighter_a: pd.Series,
    fighter_b: pd.Series,
    scheduled_rounds: int = 3,
    title_fight: bool = False,
) -> pd.DataFrame:
    row = {
        "age_diff_years": fighter_a["age_years"] - fighter_b["age_years"],
        "height_diff_cm": fighter_a["height_cm"] - fighter_b["height_cm"],
        "reach_diff_cm": fighter_a["reach_cm"] - fighter_b["reach_cm"],
        "prior_fights_diff": fighter_a["prior_fights"] - fighter_b["prior_fights"],
        "prior_win_rate_diff": fighter_a["prior_win_rate"]
        - fighter_b["prior_win_rate"],
        "prior_finish_win_rate_diff": fighter_a["prior_finish_win_rate"]
        - fighter_b["prior_finish_win_rate"],
        "avg_sig_strike_diff": fighter_a["avg_sig_strike_diff"]
        - fighter_b["avg_sig_strike_diff"],
        "avg_total_strike_diff": fighter_a["avg_total_strike_diff"]
        - fighter_b["avg_total_strike_diff"],
        "avg_takedown_diff": fighter_a["avg_takedown_diff"]
        - fighter_b["avg_takedown_diff"],
        "avg_submission_att_diff": fighter_a["avg_submission_att_diff"]
        - fighter_b["avg_submission_att_diff"],
        "avg_ctrl_time_min_diff": fighter_a["avg_ctrl_time_min_diff"]
        - fighter_b["avg_ctrl_time_min_diff"],
        "avg_knockdown_diff": fighter_a["avg_knockdown_diff"]
        - fighter_b["avg_knockdown_diff"],
        "same_stance": int(
            str(fighter_a.get("stance", "")) == str(fighter_b.get("stance", ""))
        ),
        "title_fight": int(title_fight),
        "scheduled_rounds": scheduled_rounds,
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def chronological_split(frame: pd.DataFrame, train_date_fraction: float = 0.8):
    dates = np.array(sorted(frame["event_date"].dropna().unique()))
    split_at = int(len(dates) * train_date_fraction)
    cutoff_date = pd.Timestamp(dates[split_at - 1])
    train_mask = frame["event_date"].le(cutoff_date)
    test_mask = frame["event_date"].gt(cutoff_date)
    return train_mask, test_mask, cutoff_date


def train_calibration_split(frame: pd.DataFrame, fit_date_fraction: float = 0.85):
    dates = np.array(sorted(frame["event_date"].dropna().unique()))
    split_at = int(len(dates) * fit_date_fraction)
    cutoff_date = pd.Timestamp(dates[split_at - 1])
    fit_mask = frame["event_date"].le(cutoff_date)
    calibration_mask = frame["event_date"].gt(cutoff_date)
    return fit_mask, calibration_mask, cutoff_date


def symmetric_training_set(X: pd.DataFrame, y: pd.Series):
    swapped = X.copy()
    swapped[DIFF_FEATURE_COLUMNS] = -swapped[DIFF_FEATURE_COLUMNS]
    return pd.concat([X, swapped], ignore_index=True), pd.concat(
        [y.reset_index(drop=True), 1 - y.reset_index(drop=True)], ignore_index=True
    )


def make_extra_trees_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=500,
                    max_depth=6,
                    min_samples_leaf=20,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )


def fit_temporal_calibrated_model(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_calibration: pd.DataFrame,
    y_calibration: pd.Series,
) -> CalibratedClassifierCV:
    base_model = make_extra_trees_pipeline()
    base_model.fit(X_fit, y_fit)
    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_model),
        method="sigmoid",
        cv=None,
    )
    calibrated_model.fit(X_calibration, y_calibration)
    return calibrated_model


def metric_block(y_true: pd.Series, probabilities: np.ndarray) -> dict:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def evaluate_temporal_holdout(frame: pd.DataFrame) -> tuple[dict, pd.Timestamp, pd.Timestamp]:
    train_mask, test_mask, test_cutoff = chronological_split(frame)
    train_frame = frame.loc[train_mask].copy()
    fit_mask, calibration_mask, calibration_cutoff = train_calibration_split(
        train_frame
    )

    X_fit, y_fit = symmetric_training_set(
        train_frame.loc[fit_mask, FEATURE_COLUMNS],
        train_frame.loc[fit_mask, "target_f1_win"],
    )
    X_cal, y_cal = symmetric_training_set(
        train_frame.loc[calibration_mask, FEATURE_COLUMNS],
        train_frame.loc[calibration_mask, "target_f1_win"],
    )
    X_test = frame.loc[test_mask, FEATURE_COLUMNS]
    y_test = frame.loc[test_mask, "target_f1_win"]

    model = fit_temporal_calibrated_model(X_fit, y_fit, X_cal, y_cal)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = metric_block(y_test, probabilities)
    metrics.update(
        {
            "train_rows": int(train_mask.sum()),
            "fit_rows_after_symmetric_augmentation": int(len(X_fit)),
            "calibration_rows_after_symmetric_augmentation": int(len(X_cal)),
            "test_rows": int(test_mask.sum()),
            "test_cutoff_train_lte": test_cutoff.strftime("%Y-%m-%d"),
            "calibration_cutoff_fit_lte": calibration_cutoff.strftime("%Y-%m-%d"),
            "test_date_min": frame.loc[test_mask, "event_date"].min().strftime("%Y-%m-%d"),
            "test_date_max": frame.loc[test_mask, "event_date"].max().strftime("%Y-%m-%d"),
        }
    )
    return metrics, test_cutoff, calibration_cutoff


def train_final_artifact(frame: pd.DataFrame) -> CalibratedClassifierCV:
    fit_mask, calibration_mask, _ = train_calibration_split(frame, fit_date_fraction=0.85)
    X_fit, y_fit = symmetric_training_set(
        frame.loc[fit_mask, FEATURE_COLUMNS],
        frame.loc[fit_mask, "target_f1_win"],
    )
    X_cal, y_cal = symmetric_training_set(
        frame.loc[calibration_mask, FEATURE_COLUMNS],
        frame.loc[calibration_mask, "target_f1_win"],
    )
    return fit_temporal_calibrated_model(X_fit, y_fit, X_cal, y_cal)


def main() -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    fights = load_clean_fights(DATA_PATH)
    frame = build_training_frame(fights)
    fighter_table = build_fighter_table(fights)

    metrics, test_cutoff, calibration_cutoff = evaluate_temporal_holdout(frame)
    final_model = train_final_artifact(frame)

    X_full_sym, _ = symmetric_training_set(
        frame[FEATURE_COLUMNS], frame["target_f1_win"]
    )
    feature_medians = X_full_sym.median(numeric_only=True).to_dict()

    artifact = {
        "model": final_model,
        "feature_columns": FEATURE_COLUMNS,
        "diff_feature_columns": DIFF_FEATURE_COLUMNS,
        "fighter_table": fighter_table,
        "feature_medians": feature_medians,
        "metadata": {
            "model_name": "calibrated_extra_trees",
            "data_path": str(DATA_PATH),
            "dataset_rows_with_target": int(len(frame)),
            "dataset_min_date": fights["event_date"].min().strftime("%Y-%m-%d"),
            "dataset_max_date": fights["event_date"].max().strftime("%Y-%m-%d"),
            "fighter_count": int(len(fighter_table)),
            "features": FEATURE_COLUMNS,
            "leakage_policy": [
                "No current-fight aggregate/per-round stats are used as model features.",
                "No winner/result/finish fields are used as model features.",
                "No betting odds or ranking snapshots are used.",
                "Historical features for training rows are computed from fights before the current event_date.",
                "The model is trained with swapped fighter rows to reduce f_1/f_2 ordering bias.",
            ],
        },
    }
    joblib.dump(artifact, MODEL_PATH)

    metrics_report = {
        "artifact_path": str(MODEL_PATH),
        "model": "ExtraTreesClassifier + sigmoid probability calibration",
        "feature_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "temporal_holdout_metrics": metrics,
        "artifact_training_note": (
            "The saved artifact fits the base model on the older 85% of available "
            "event dates and calibrates probabilities on the newest 15%, both with "
            "symmetric swapped rows."
        ),
        "test_cutoff_train_lte": test_cutoff.strftime("%Y-%m-%d"),
        "calibration_cutoff_for_holdout_fit_lte": calibration_cutoff.strftime("%Y-%m-%d"),
    }
    METRICS_PATH.write_text(json.dumps(metrics_report, indent=2), encoding="utf-8")
    print(json.dumps(metrics_report, indent=2))


if __name__ == "__main__":
    main()
