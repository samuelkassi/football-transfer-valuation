"""Name normaliser, the FBref join and its match rate, on synthetic rows."""
import numpy as np
import pandas as pd
import pytest

from src.clean import clean_fbref
from src.merge import clean_name, join_fbref


@pytest.mark.parametrize("raw, expected", [
    ("Jules Koundé", "jules kounde"),
    ("Raúl Albiol", "raul albiol"),
    ("  Nicolò   Zaniolo ", "nicolo zaniolo"),
    ("Luka Modrić", "luka modric"),
    ("İlkay Gündoğan", "ilkay gundogan"),
    ("N'Golo Kanté", "ngolo kante"),
    ("Jean-Philippe Mateta", "jeanphilippe mateta"),
    ("", None),
    (None, None),
])
def test_clean_name(raw, expected):
    assert clean_name(raw) == expected


def transfers(rows):
    """rows: (player_id, name, birth_year, season_end_year)."""
    return pd.DataFrame([{
        "player_id": pid,
        "player": name,
        "player_clean": clean_name(name),
        "season_end_year": season,
        "date_of_birth": pd.Timestamp(f"{born}-06-01"),
    } for pid, name, born, season in rows])


def fbref(rows):
    """rows: (name, born, season_end_year)."""
    return pd.DataFrame([{"player": n, "born": b, "season_end_year": s, "comp": "Serie A", "minutes": 2000}
                         for n, b, s in rows])


def test_join_needs_birth_year_and_previous_season():
    tm = transfers([
        (1, "Jules Koundé", 1998, 2022),    # matches FBref 2021, born 1998
        (2, "Fernando", 1987, 2023),        # FBref "Fernando" born 1990: no match
        (3, "Marco Rossi", 1992, 2020),     # no FBref row for 2019
    ])
    fb = fbref([("Jules Kounde", 1998, 2021), ("Fernando", 1990, 2022), ("Marco Rossi", 1992, 2020)])
    merged, stats = join_fbref(tm, fb)
    matched = merged.loc[merged["_merge"] == "both", "player_id"].tolist()
    assert matched == [1]
    assert stats == {"transfers_in_window": 3, "name_season_matches": 2,
                     "birth_year_matches": 1, "verified_matches": 1}


def test_birth_year_tolerance_is_one_year():
    tm = transfers([(1, "Ana Lima", 2000, 2022), (2, "Rui Sá", 2000, 2022)])
    fb = fbref([("Ana Lima", 2001, 2021), ("Rui Sa", 2002, 2021)])
    merged, _ = join_fbref(tm, fb)
    assert merged.loc[merged["_merge"] == "both", "player_id"].tolist() == [1]


def test_ambiguous_namesakes_are_dropped():
    # two different Transfermarkt players, same name and birth year
    tm = transfers([(1, "Pedro", 1997, 2021), (2, "Pedro", 1997, 2021)])
    fb = fbref([("Pedro", 1997, 2020)])
    merged, stats = join_fbref(tm, fb)
    assert stats["verified_matches"] == 0
    assert (merged["_merge"] == "left_only").all()


def test_match_rate_denominator_is_every_transfer():
    tm = transfers([(i, f"Player {chr(65 + i)}", 1995, 2022) for i in range(10)])
    fb = fbref([(f"Player {chr(65 + i)}", 1995, 2021) for i in range(4)])
    merged, stats = join_fbref(tm, fb)
    assert len(merged) == 10                     # one row per transfer, never duplicated
    assert stats["verified_matches"] / stats["transfers_in_window"] == pytest.approx(0.4)


def raw_fbref_row(**overrides):
    row = {
        "season": 2223, "player": "Someone", "league": "ESP-La Liga", "team": "Club",
        "Playing Time_MP": 30, "Playing Time_Starts": 30, "Playing Time_Min": 2700,
        "Playing Time_90s": 30.0, "born_fbref": 1995, "age_fbref": 27,
        "Per 90 Minutes_Gls": 0.2, "long_name": "x",
    }
    row.update(overrides)
    return row


def test_clean_fbref_drops_impossible_and_duplicate_rows():
    raw = pd.DataFrame([
        raw_fbref_row(player="Normal"),
        raw_fbref_row(player="Normal", long_name="y"),           # FIFA duplicate
        raw_fbref_row(player="Fernando", **{"Playing Time_MP": 89, "Playing Time_Min": 7444,
                                            "Playing Time_90s": 82.7}),
        raw_fbref_row(player="Bundes", league="GER-Bundesliga",  # 37 x 90 > 36 x 90
                      **{"Playing Time_MP": 36, "Playing Time_Min": 3300}),
        raw_fbref_row(player="Inf", **{"Per 90 Minutes_Gls": np.inf}),
    ])
    out = clean_fbref(raw)
    assert sorted(out["player"]) == ["Inf", "Normal"]
    assert not np.isinf(out.select_dtypes("number")).any().any()
    assert out.loc[out["player"] == "Normal", "goals"].item() == 6
