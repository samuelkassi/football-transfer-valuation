"""Expanding-window backtest module (Phase 3)."""
from xgboost import XGBRegressor

from src.backtest import ID_COLS, VARIANTS, make_xgb
from src.config import SEASON_COL, TARGET_COL
from src.train import FeatureSet


def test_make_xgb_carries_the_tuned_parameters():
    m = XGBRegressor(n_estimators=123, max_depth=4, learning_rate=0.02, random_state=42)
    out = make_xgb({"model": m})
    assert out.get_params()["n_estimators"] == 123
    assert out.get_params()["max_depth"] == 4
    assert out.get_params()["learning_rate"] == 0.02


def test_backtest_variants_are_valid_feature_sets():
    for name, suffix, feature_set in VARIANTS:
        assert isinstance(feature_set, FeatureSet)
        assert suffix == feature_set.value
        assert name in ("main", "buyer_agnostic")


def test_backtest_id_columns_exist_in_the_dataset():
    import pandas as pd
    from src.config import DATASET
    cols = pd.read_parquet(DATASET, columns=ID_COLS).columns.tolist()
    assert cols == ID_COLS
    assert TARGET_COL not in ID_COLS
    assert SEASON_COL in ID_COLS
