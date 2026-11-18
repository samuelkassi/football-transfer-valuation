"""
Trivial baselines on the same time-aware split as models_v2.py.

Each baseline predicts the log market value from training-season rows only:
  * global mean:        one number, the mean target over all training rows
  * per-position mean:  mean target per Transfermarkt `position`
  * per-league mean:    mean target per `league`
Groups that never appear in training fall back to the global training mean.

Run standalone:  python src/baselines.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models_v2 import DATA_PATH, SEASON_COL, TARGET_COL, TEST_SEASON_DEFAULT

BASELINE_GROUPS = {
    "global_mean": None,
    "position_mean": "position",
    "league_mean": "league",
}


def split_train_test(df: pd.DataFrame, test_season: int = TEST_SEASON_DEFAULT):
    df = df.copy()
    df[SEASON_COL] = pd.to_numeric(df[SEASON_COL], errors="coerce")
    return df[df[SEASON_COL] < test_season].copy(), df[df[SEASON_COL] == test_season].copy()


def predict_group_mean(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str | None,
    target_col: str = TARGET_COL,
) -> np.ndarray:
    """Predict each test row with its group's training mean (global mean if no group)."""
    global_mean = float(train_df[target_col].mean())
    if group_col is None:
        return np.full(len(test_df), global_mean)
    means = train_df.groupby(group_col, dropna=False)[target_col].mean()
    return test_df[group_col].map(means).fillna(global_mean).to_numpy(dtype=float)


def run_baselines(
    df: pd.DataFrame,
    test_season: int = TEST_SEASON_DEFAULT,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    train_df, test_df = split_train_test(df, test_season)
    y_test = test_df[target_col].to_numpy()
    rows = []
    for name, group_col in BASELINE_GROUPS.items():
        pred = predict_group_mean(train_df, test_df, group_col, target_col)
        rows.append(
            {
                "Baseline": name,
                "Test R²": float(r2_score(y_test, pred)),
                "Test RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                "Test MAE": float(mean_absolute_error(y_test, pred)),
            }
        )
    return pd.DataFrame(rows)


def print_baselines(df: pd.DataFrame, test_season: int = TEST_SEASON_DEFAULT) -> pd.DataFrame:
    res = run_baselines(df, test_season)
    print(f"\n{'='*80}")
    print("TRIVIAL BASELINES (fit on train seasons, scored on test season)")
    print(f"{'='*80}\n")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return res


if __name__ == "__main__":
    print_baselines(pd.read_parquet(DATA_PATH))
