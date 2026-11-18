"""
Deliberate before/after for the early-stopping leakage issue.

For each feature set, XGBoost hyperparameters are tuned once on the training
seasons (same RandomizedSearchCV as models_v2.py), then two variants are fit:

  leaky   : fit on all training seasons, early-stop on the 2025 TEST season
            (the issue raised in the brief), score on 2025.
  correct : fit on training seasons before 2024, early-stop on 2024, refit on
            all training seasons for that many rounds, score on 2025.
            This is what models_v2.py does.

Run:  python src/early_stopping_check.py
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from models_v2 import (
    DATA_PATH, SEASON_COL, TARGET_COL, TEST_SEASON_DEFAULT, FeatureSet,
    get_feature_cols, make_preprocessor, temporal_holdout_masks, tune_xgboost,
)

FIXED = dict(objective="reg:squarederror", random_state=42, tree_method="hist", eval_metric="rmse")


def run(feature_set: FeatureSet, df: pd.DataFrame) -> dict:
    df = df.copy()
    df[SEASON_COL] = pd.to_numeric(df[SEASON_COL], errors="coerce")
    train_df = df[df[SEASON_COL] < TEST_SEASON_DEFAULT]
    test_df = df[df[SEASON_COL] == TEST_SEASON_DEFAULT]

    num_cols, cat_cols = get_feature_cols(train_df, TARGET_COL, feature_set)
    pre = make_preprocessor(num_cols, cat_cols)
    X_train = pre.fit_transform(train_df[num_cols + cat_cols])
    X_test = pre.transform(test_df[num_cols + cat_cols])
    y_train = train_df[TARGET_COL].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()

    fit_mask, eval_mask = temporal_holdout_masks(train_df)
    # same search input as models_v2.py: seasons before the 2024 holdout
    search_best = tune_xgboost(X_train[fit_mask], y_train[fit_mask])
    params = {k: v for k, v in search_best.get_params().items()
              if k in search_best.get_xgb_params() or k == "n_estimators"}
    params.update(FIXED)
    params["n_estimators"] = 2000

    # leaky: early stopping watches the test set
    leaky = XGBRegressor(**{**params, "early_stopping_rounds": 50})
    leaky.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    r2_leaky = r2_score(y_test, leaky.predict(X_test))

    # correct: early stopping on the last training season, then refit
    probe = XGBRegressor(**{**params, "early_stopping_rounds": 50})
    probe.fit(X_train[fit_mask], y_train[fit_mask],
              eval_set=[(X_train[eval_mask], y_train[eval_mask])], verbose=False)
    final = XGBRegressor(**{**params, "n_estimators": probe.best_iteration + 1})
    final.fit(X_train, y_train, verbose=False)
    r2_correct = r2_score(y_test, final.predict(X_test))

    return {
        "feature_set": feature_set.value,
        "leaky_rounds": leaky.best_iteration + 1,
        "leaky_test_r2": r2_leaky,
        "correct_rounds": probe.best_iteration + 1,
        "correct_test_r2": r2_correct,
    }


if __name__ == "__main__":
    data = pd.read_parquet(DATA_PATH)
    rows = [run(fs, data) for fs in (FeatureSet.TRADITIONAL, FeatureSet.WITH_MODERN)]
    print("\nEARLY STOPPING: test season (leaky) vs last training season (correct)")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
