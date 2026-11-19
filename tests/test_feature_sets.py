"""Feature-set definitions in train.py, incl. the Phase 3 buyer-agnostic variant."""
import pandas as pd

from src.train import (
    ALL_STATS,
    BUYER_AGNOSTIC_STATS,
    BUYER_SIDE_COLUMNS,
    FeatureSet,
    get_feature_cols,
)


def test_buyer_agnostic_drops_only_buyer_side_club_features():
    assert set(BUYER_AGNOSTIC_STATS) == set(ALL_STATS) - set(BUYER_SIDE_COLUMNS)
    # the selling club's level is fair context and stays
    assert "from_club_level" in BUYER_AGNOSTIC_STATS


def test_get_feature_cols_buyer_agnostic_excludes_buyer_side_columns():
    df = pd.DataFrame({c: [1.0] for c in ALL_STATS + ["log_market_value", "player", "transfer_date"]})
    num, cat = get_feature_cols(df, "log_market_value", FeatureSet.BUYER_AGNOSTIC)
    cols = set(num) | set(cat)
    assert "to_club_level" not in cols
    assert "club_level_delta" not in cols
    assert "from_club_level" in cols


def test_get_feature_cols_with_modern_keeps_buyer_side_columns():
    df = pd.DataFrame({c: [1.0] for c in ALL_STATS + ["log_market_value", "player", "transfer_date"]})
    num, cat = get_feature_cols(df, "log_market_value", FeatureSet.WITH_MODERN)
    cols = set(num) | set(cat)
    assert "to_club_level" in cols
    assert "club_level_delta" in cols
