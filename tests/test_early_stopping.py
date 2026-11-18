"""XGBoost early stopping must only ever look at training seasons."""
import numpy as np
import pandas as pd
import pytest

from src import train as train_mod
from src.train import FeatureSet, SEASON_COL, TARGET_COL, temporal_holdout_masks


def synthetic_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for season in range(2019, 2026):
        # test season gets a different size so eval_set sizes identify it
        for _ in range(80 if season == 2025 else 120):
            age = rng.uniform(18, 34)
            minutes = rng.uniform(450, 3400)
            rows.append({
                SEASON_COL: season,
                "age_at_transfer": age,
                "minutes": minutes,
                "goals": rng.poisson(4),
                "position": rng.choice(["Attack", "Midfield", "Defender", "Goalkeeper"]),
                "league": rng.choice(["Premier League", "La Liga"]),
                TARGET_COL: 14 + 0.001 * minutes - 0.05 * abs(age - 27) + rng.normal(0, 0.3),
            })
    return pd.DataFrame(rows)


def test_holdout_is_last_training_season():
    df = synthetic_frame()
    train = df[df[SEASON_COL] < 2025]
    fit_mask, eval_mask = temporal_holdout_masks(train)
    assert set(train.loc[eval_mask, SEASON_COL]) == {2024}
    assert train.loc[fit_mask, SEASON_COL].max() == 2023


@pytest.mark.parametrize("tune", [False, True])
def test_xgboost_never_sees_test_season(monkeypatch, tune):
    """Record every eval_set passed to XGBoost and check it holds only 2024 rows."""
    seen_eval_sizes = []
    real_fit = train_mod.XGBRegressor.fit

    def spy_fit(self, X, y, *args, eval_set=None, **kwargs):
        if eval_set is not None:
            seen_eval_sizes.extend(len(ys) for _, ys in eval_set)
        return real_fit(self, X, y, *args, eval_set=eval_set, **kwargs)

    monkeypatch.setattr(train_mod.XGBRegressor, "fit", spy_fit)
    if tune:  # keep the search tiny
        real_tune = train_mod.tune_xgboost
        monkeypatch.setattr(train_mod, "tune_xgboost",
                            lambda *a, **k: real_tune(*a, **{**k, "n_iter": 2}))

    df = synthetic_frame()
    train, test = df[df[SEASON_COL] < 2025], df[df[SEASON_COL] == 2025]
    train_mod.train_and_evaluate_model(train, test, "xgboost", FeatureSet.TRADITIONAL,
                                       tune_hyperparams=tune)

    n_2024 = int((train[SEASON_COL] == 2024).sum())
    assert seen_eval_sizes, "early stopping was not used"
    assert all(n == n_2024 for n in seen_eval_sizes)
