"""Position flags and point-in-time club levels."""
import numpy as np
import pandas as pd

from src.features import add_position_flags, club_level_before, engineer_features, load_valuations_with_club


def test_position_flags_cover_every_transfermarkt_label():
    df = pd.DataFrame({"position": ["Goalkeeper", "Defender", "Midfield", "Attack", None]})
    df = add_position_flags(df)
    flags = df[["is_goalkeeper", "is_defender", "is_midfielder", "is_forward"]]
    assert flags.iloc[:4].to_numpy().tolist() == np.eye(4, dtype=int).tolist()
    assert flags.iloc[4].sum() == 0


def test_position_interactions_are_non_zero():
    n = 8
    df = pd.DataFrame({
        "position": ["Goalkeeper", "Defender", "Midfield", "Attack"] * 2,
        "season_end_year": [2020] * n, "age_at_transfer": [24.0] * n,
        "nationality_group": ["France"] * n,
        **{c: [3.0] * n for c in [
            "goals", "assists", "expected_goals", "expected_assisted_goals",
            "non_penalty_xg_plus_xag", "key_passes", "progressive_passes", "tackles_won",
            "interceptions", "clearances", "tackles_plus_interceptions", "saves", "clean_sheets",
            "minutes", "nineties", "goals_p_90"]},
    })
    out = engineer_features(df)
    for col in ["fw_goals", "mf_key_passes", "df_clearances", "gk_saves"]:
        assert (out[col] != 0).any(), col


def valuations(rows):
    """rows: (player_id, date, value, current_club_id)."""
    return pd.DataFrame(rows, columns=["player_id", "date", "market_value_eur", "current_club_id"]).assign(
        date=lambda d: pd.to_datetime(d["date"]))


def test_valuation_club_comes_from_history_not_snapshot(tmp_path):
    # player 1 is at club 10 until a move to club 20 on 2020-07-01;
    # the scrape-time snapshot says club 20 for every valuation.
    vals = valuations([(1, "2020-01-01", 5e6, 20), (1, "2021-01-01", 8e6, 20), (2, "2020-01-01", 1e6, 30)])
    path = tmp_path / "vals.parquet"
    vals.to_parquet(path)
    hist = pd.DataFrame({"player_id": [1], "transfer_date": [pd.Timestamp("2020-07-01")],
                         "from_club_id": [10], "to_club_id": [20]})
    tagged = load_valuations_with_club(path, hist).sort_values("date")
    assert tagged["club_id"].tolist() == [10, 20]    # player 2: no history, dropped


def test_club_level_uses_only_earlier_valuations_of_other_players():
    vals = pd.DataFrame({
        "player_id": [1, 2, 3, 4, 9],
        "date": pd.to_datetime(["2019-03-01", "2019-12-01", "2020-06-30", "2020-07-01", "2020-05-01"]),
        "market_value_eur": [1e6, 3e6, 5e6, 100e6, 50e6],
        "club_id": [7, 7, 7, 7, 7],
    }).sort_values("date")
    level = club_level_before(vals, pd.Series([7]), pd.Series([pd.Timestamp("2020-07-01")]), pd.Series([9]))
    # window [2019-07-02, 2020-07-01): players 2 and 3 only.
    # player 1 is too old, player 4 is dated on the transfer day (not before),
    # player 9 is the transferring player.
    assert level[0] == np.log1p(np.median([3e6, 5e6]))


def test_club_level_is_nan_without_history():
    vals = pd.DataFrame({"player_id": [1], "date": pd.to_datetime(["2015-01-01"]),
                         "market_value_eur": [1e6], "club_id": [7]})
    level = club_level_before(vals, pd.Series([7, 8]), pd.Series(pd.to_datetime(["2020-07-01"] * 2)),
                              pd.Series([2, 2]))
    assert np.isnan(level).all()
