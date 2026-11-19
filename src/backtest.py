"""Expanding-window out-of-sample predictions for the Phase 3 fee analysis.

For each season s in 2020..2025: fit the preprocessor and an XGBoost model on
transfers from earlier seasons only, then predict every transfer in season s.
Two feature sets are scored: the main model (with modern metrics) and the
buyer-agnostic one (no to_club_level, no club_level_delta). 2019 has no
earlier season in the data and is used only as training rows.

Why XGBoost only: it is the best model in both feature sets, and one model
keeps the runtime and the report readable. The six backtest windows reuse the
tuned parameters of the main run (models/*.joblib) instead of re-tuning per
window, to keep runtime sane. This is a mild lookahead for the earlier
windows: their parameters were chosen by a search that saw later seasons. It
affects only the parameters, not the fits. The preprocessor is always fit on
the window's own training rows only.

Output: data/final/backtest_predictions.parquet, one row per transfer, with
club identity, fee, market value, position, age, previous-season league and
the predicted log value from both feature sets. Per-season R squared goes to
results/backtest_r2.md as a check.

Run with: python -m src.backtest
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from .config import DATASET, FINAL_DIR, MODEL_DIR, RESULTS_DIR, SEASON_COL, TARGET_COL
from .train import FeatureSet, get_feature_cols, make_preprocessor, set_seeds

BACKTEST_OUT = FINAL_DIR / "backtest_predictions.parquet"
BACKTEST_R2 = RESULTS_DIR / "backtest_r2.md"

VARIANTS = [
    ("main", "with_modern", FeatureSet.WITH_MODERN),
    ("buyer_agnostic", "buyer_agnostic", FeatureSet.BUYER_AGNOSTIC),
]

ID_COLS = [
    "player", "player_id", "transfer_date", SEASON_COL,
    "from_club_id", "from_club_name", "to_club_id", "to_club_name",
    "transfer_fee_eur", "market_value_eur_at_transfer",
    "league", "position", "sub_position", "age_at_transfer",
]


def make_xgb(bundle: dict) -> XGBRegressor:
    """An unfitted XGBoost model carrying the tuned parameters of a bundle."""
    return XGBRegressor(**dict(bundle["model"].get_params()))


def run_backtest() -> pd.DataFrame:
    """Expanding-window predictions for both feature sets, seasons 2020..2025."""
    set_seeds()
    df = pd.read_parquet(DATASET)
    df[SEASON_COL] = pd.to_numeric(df[SEASON_COL], errors="coerce")

    seasons = [s for s in sorted(df[SEASON_COL].unique()) if s > df[SEASON_COL].min()]
    print(f"Seasons to score: {seasons}")

    bundles = {
        name: joblib.load(MODEL_DIR / f"xgboost_{suffix}.joblib")
        for name, suffix, _ in VARIANTS
    }

    frames: list[pd.DataFrame] = []
    r2_rows: list[dict] = []
    for season in seasons:
        train = df[df[SEASON_COL] < season]
        window = df[df[SEASON_COL] == season].copy()
        print(f"\nSeason {season}: fit on {len(train)} earlier rows, score {len(window)}")

        out = window[ID_COLS].copy()
        out["actual_log_value"] = window[TARGET_COL].to_numpy()
        for name, _suffix, feature_set in VARIANTS:
            num_cols, cat_cols = get_feature_cols(train, TARGET_COL, feature_set)
            pre = make_preprocessor(num_cols, cat_cols)
            model = make_xgb(bundles[name])
            model.fit(pre.fit_transform(train[num_cols + cat_cols]),
                      train[TARGET_COL].to_numpy())
            pred_log = model.predict(pre.transform(window[num_cols + cat_cols]))
            out[f"pred_log_{name}"] = pred_log
            out[f"pred_eur_{name}"] = np.expm1(pred_log)
            r2 = r2_score(window[TARGET_COL].to_numpy(), pred_log)
            r2_rows.append({"season": season, "variant": name, "n": len(window), "r2": r2})
            print(f"  {name}: R squared {r2:.3f} (n={len(window)})")
        frames.append(out)

    predictions = pd.concat(frames, ignore_index=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(BACKTEST_OUT, index=False)

    r2 = pd.DataFrame(r2_rows)
    lines = ["# Backtest check: expanding-window R squared", "",
             "XGBoost with the tuned parameters of the main run, refit per season "
             "on earlier seasons only, scored on that season.", "",
             "| Season | n | R squared, main | R squared, buyer-agnostic |",
             "|---|---|---|---|"]
    for season in seasons:
        rows = r2[r2["season"] == season]
        n = int(rows["n"].iloc[0])
        r2_main = float(rows.loc[rows["variant"] == "main", "r2"].iloc[0])
        r2_ba = float(rows.loc[rows["variant"] == "buyer_agnostic", "r2"].iloc[0])
        lines.append(f"| {season} | {n} | {r2_main:.3f} | {r2_ba:.3f} |")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BACKTEST_R2.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved {BACKTEST_OUT.name}: {len(predictions)} rows")
    print(f"Saved {BACKTEST_R2.name}")
    return predictions


if __name__ == "__main__":
    run_backtest()
