"""Clean the FBref half of the source CSV into one row per player-season."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import FBREF_STATS, INTER_DIR
from .load import read_fbref_raw

# FIFA/EAFC columns glued onto the source, plus a junk helper column: dropped.
FIFA_COLS = [
    "fifa_version", "short_name", "long_name", "player_positions", "overall",
    "potential", "value_eur", "wage_eur", "age_fifa", "height_cm", "weight_kg",
    "club_team_id", "club_name", "league_id", "league_name", "club_jersey_number",
    "nationality_name", "preferred_foot", "weak_foot", "skill_moves",
    "international_reputation", "body_type", "pace", "shooting", "passing",
    "dribbling", "defending", "physic", "goalkeeping_diving", "goalkeeping_handling",
    "goalkeeping_kicking", "goalkeeping_positioning", "goalkeeping_reflexes",
    "goalkeeping_speed", "born_fifa", "name_parts",
]

# FBref league code -> league name used everywhere downstream.
LEAGUE_CODE_TO_COMP = {
    "ENG-Premier League":     "Premier League",
    "ESP-La Liga":            "La Liga",
    "GER-Bundesliga":         "Bundesliga",
    "ITA-Serie A":            "Serie A",
    "FRA-Ligue 1":            "Ligue 1",
    "USA-MLS":                "MLS",
    "NED-Eredivisie":         "Eredivisie",
    "POR-Primeira Liga":      "Primeira Liga",
    "BEL-Jupiler Pro League": "Jupiler Pro League",
    "TUR-Super Lig":          "Super Lig",
}
LEAGUES_KEEP = set(LEAGUE_CODE_TO_COMP)

# Most league matches one team can play in a season, over 2017-18 to 2024-25,
# including end-of-season play-offs that FBref counts. Upper bounds on purpose:
# a row above them cannot be a single player.
#   Belgium: 30 to 34 regular + up to 10 play-off games (+1 for the odd decider)
#   Turkey:  21 clubs in 2020-21 -> 40 games
#   MLS:     34 regular season + up to 6 play-off games
#   Others:  regular season + relegation / European play-off or tie-break games
LEAGUE_MAX_MATCHES = {
    "Premier League":     38,
    "La Liga":            38,
    "Serie A":            39,
    "Ligue 1":            40,
    "Bundesliga":         36,
    "Eredivisie":         38,
    "Primeira Liga":      36,
    "Jupiler Pro League": 41,
    "Super Lig":          40,
    "MLS":                40,
}

# Per-90 columns -> season totals (per-90 x 90s played, rounded).
PER90_TO_TOTAL = {
    "Per 90 Minutes_Gls":               "goals",
    "Per 90 Minutes_Ast":               "assists",
    "Per 90 Minutes_G+A":               "goals_and_assists",
    "Per 90 Minutes_xG":                "expected_goals",
    "Per 90 Minutes_xAG":               "expected_assisted_goals",
    "Per 90 Minutes_npxG":              "non_penalty_xg",
    "Per 90 Minutes_npxG+xAG":          "non_penalty_xg_plus_xag",
    "Per 90 Minutes_Tkl+Int":           "tackles_plus_interceptions",
    "Per 90 Minutes_PrgP":              "progressive_passes",
    "Per 90 Minutes_KP":                "key_passes",
    "Per 90 Minutes_Tackles_TklW":      "tackles_won",
    "Per 90 Minutes_Int":               "interceptions",
    "Per 90 Minutes_Clr":               "clearances",
    "Per 90 Minutes_Performance_CS":    "clean_sheets",
    "Per 90 Minutes_Performance_Saves": "saves",
}


def to_snake(s: str) -> str:
    s = s.strip().replace("%", "pct").replace("&", "and").replace("/", "_per_")
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s.lower()


def parse_season_years(season_val: str) -> tuple[int | None, int | None]:
    """'2023-2024' -> (2023, 2024)."""
    if pd.isna(season_val):
        return (None, None)
    m = re.match(r"^\s*(\d{4})\s*-\s*(\d{4})\s*$", str(season_val))
    if not m:
        return (None, None)
    return (int(m.group(1)), int(m.group(2)))


def season_int_to_str(s) -> str | None:
    """Compact source encoding '2324' -> '2023-2024'."""
    if pd.isna(s):
        return None
    s = int(s)
    return f"{2000 + s // 100}-{2000 + s % 100}"


def clean_fbref(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in FIFA_COLS if c in df.columns])

    before = len(df)
    df = df[df["league"].isin(LEAGUES_KEEP)].copy()
    print(f"After league filter: {df.shape} (dropped {before - len(df)})")

    df = df.rename(columns={
        "Playing Time_MP":     "Matches Played",
        "Playing Time_Starts": "Starts",
        "Playing Time_Min":    "minutes",
        "Playing Time_90s":    "nineties",
        "team":                "squad",
        "age_fbref":           "age",
        "born_fbref":          "born",
    })
    df["season"] = df["season"].apply(season_int_to_str)
    df["comp"] = df["league"].map(LEAGUE_CODE_TO_COMP)

    for per90_col, total_col in PER90_TO_TOTAL.items():
        df[total_col] = (df[per90_col] * df["nineties"]).round() if per90_col in df.columns else pd.NA

    df = df.rename(columns={"Per 90 Minutes_Gls": "goals_p_90", "Per 90 Minutes_Ast": "assists_p_90"})

    season_years = df["season"].apply(parse_season_years)
    df["season_start_year"] = season_years.apply(lambda x: x[0])
    df["season_end_year"] = season_years.apply(lambda x: x[1])

    df = df.rename(columns={c: to_snake(c) for c in df.columns})

    # One row per (player, squad, league, season), keeping the most minutes.
    # The source repeats an FBref row once per FIFA identity it was matched
    # to, so this also removes the FIFA-induced duplicates.
    df = df.sort_values("minutes", ascending=False, na_position="last")
    df = df.drop_duplicates(subset=["player", "squad", "comp", "season"], keep="first")

    # per-90 values divided by zero 90s in the source
    num_cols = df.select_dtypes("number").columns
    n_inf = int(np.isinf(df[num_cols]).sum().sum())
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    print(f"Replaced {n_inf} inf values with NaN")

    # The source keeps one row per (name, season) across all leagues, so
    # different players who share a short name (e.g. "Fernando") are summed
    # into one row. Those rows show more matches or minutes than one team
    # plays in a league season.
    cap = df["comp"].map(LEAGUE_MAX_MATCHES)
    impossible = (df["matches_played"] > cap) | (df["minutes"] > cap * 90)
    print(f"Dropped {int(impossible.sum())} rows above the league's max matches/minutes")
    return df[~impossible].copy()


def build_fbref() -> pd.DataFrame:
    INTER_DIR.mkdir(parents=True, exist_ok=True)
    df = clean_fbref(read_fbref_raw())
    df.to_parquet(FBREF_STATS, index=False)
    print(f"Saved: {FBREF_STATS.name} shape={df.shape}")
    return df
