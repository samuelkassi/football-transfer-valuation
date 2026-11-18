"""Point-in-time club levels, engineered features, and the final dataset.

Run the data stages on their own with `python -m src.features`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BUILD_STATS, DATASET, FBREF_STATS, FINAL_DIR, TM_PLAYERS, TM_TRANSFERS, TM_VALUATIONS
from .merge import join_fbref, prepare_transfers

CLUB_LEVEL_WINDOW_DAYS = 365
MIN_MINUTES = 450

# Columns written to the final dataset (identifiers, target, candidate
# features). The model feature sets are chosen in train.py.
DATASET_COLUMNS = [
    # identifiers
    "player", "transfer_date", "season_end_year", "season_index",
    # target (raw; log1p added below)
    "market_value_eur_at_transfer",
    # player info
    "age_at_transfer", "age_squared", "height_in_cm", "position", "sub_position", "foot",
    # league + nationality
    "league", "nationality_group",
    # club context
    "from_club_level", "to_club_level", "club_level_delta",
    # playing time
    "minutes", "nineties",
    # attacking
    "goals", "assists", "expected_goals", "goals_and_assists", "goals_p_90", "assists_p_90",
    "expected_assisted_goals", "non_penalty_xg", "non_penalty_xg_plus_xag",
    # passing
    "progressive_passes", "key_passes",
    # defending
    "tackles_won", "interceptions", "clearances", "tackles_plus_interceptions",
    # goalkeeping
    "clean_sheets", "saves",
    # engineered
    "goal_overperformance", "assist_efficiency", "offensive_contribution", "xg_differential",
    "attacking_efficiency", "age_to_peak", "is_peak_age", "is_young", "age_potential_score",
    "is_starter", "playing_time_per_game", "is_high_volume", "defensive_actions",
    "defensive_per_90", "clean_sheet_ratio", "is_top_nation",
    "is_goalkeeper", "is_defender", "is_midfielder", "is_forward",
    # position x stat interactions
    "fw_goals", "mf_goals", "fw_assists", "mf_assists", "fw_xg", "mf_xg", "fw_xag", "mf_xag",
    "fw_npxg_xag", "mf_npxg_xag", "mf_key_passes", "mf_progressive_passes",
    "df_tackles_won", "df_interceptions", "mf_interceptions", "df_clearances",
    "df_tkl_int", "mf_tkl_int", "gk_saves", "gk_clean_sheets",
    # passing & creativity, compound
    "passing_quality", "progressive_passing_per_90",
    "overall_impact", "star_player_score", "breakout_potential",
]


# ---------------------------------------------------------------------------
# Club levels
# ---------------------------------------------------------------------------

def load_valuations_with_club(valuations_path: Path, club_history: pd.DataFrame) -> pd.DataFrame:
    """Valuations tagged with the club the player was at on the valuation date.

    `current_club_id` in player_valuations is the player's club at scrape time
    (one value per player), not the club on the valuation date. The club on a
    date is taken from the transfer history instead:
      * after a recorded transfer: that transfer's `to_club_id`
      * before the player's first recorded transfer: its `from_club_id`
    Players with no recorded transfer are dropped: their valuation histories
    span years (median 8.6) with no dated club, so the club is unknown.
    """
    vals = pd.read_parquet(valuations_path)
    vals["date"] = pd.to_datetime(vals["date"], errors="coerce")
    vals = vals.dropna(subset=["date", "market_value_eur"]).sort_values("date")

    hist = club_history.dropna(subset=["transfer_date"]).sort_values("transfer_date")
    tagged = pd.merge_asof(
        vals,
        hist[["player_id", "transfer_date", "to_club_id"]],
        left_on="date", right_on="transfer_date", by="player_id", direction="backward",
    )
    first_from = hist.groupby("player_id")["from_club_id"].first()
    tagged["club_id"] = tagged["to_club_id"].fillna(tagged["player_id"].map(first_from))
    tagged = tagged.dropna(subset=["club_id"])
    print(f"valuations with a dated club: {len(tagged)} / {len(vals)} ({len(tagged) / len(vals):.1%})")
    return tagged[["player_id", "date", "market_value_eur", "club_id"]]


def club_level_before(
    vals: pd.DataFrame,
    club_ids: pd.Series,
    dates: pd.Series,
    player_ids: pd.Series,
    window_days: int = CLUB_LEVEL_WINDOW_DAYS,
) -> np.ndarray:
    """log1p(median valuation of the club's players in [date - window, date)),
    excluding the player in question. NaN when no such valuation exists."""
    groups = {
        cid: (g["date"].to_numpy(), g["market_value_eur"].to_numpy(dtype=float), g["player_id"].to_numpy())
        for cid, g in vals.groupby("club_id")
    }
    window = np.timedelta64(window_days, "D")
    out = np.full(len(club_ids), np.nan)
    for i, (cid, d, pid) in enumerate(zip(club_ids.to_numpy(), dates.to_numpy(), player_ids.to_numpy())):
        g = groups.get(cid)
        if g is None or pd.isna(d):
            continue
        dts, mv, pids = g
        lo = np.searchsorted(dts, d - window, side="left")
        hi = np.searchsorted(dts, d, side="left")  # strictly before the transfer date
        sel = mv[lo:hi][pids[lo:hi] != pid]
        if len(sel):
            out[i] = np.log1p(np.median(sel))
    return out


def add_club_levels(tm: pd.DataFrame, club_history: pd.DataFrame, valuations_path: Path = TM_VALUATIONS) -> pd.DataFrame:
    """from_club_level / to_club_level: club strength just before the transfer.

    Captures what league one-hots miss (Man City and Brighton share a league).
    """
    vals = load_valuations_with_club(valuations_path, club_history)
    tm = tm.copy()
    tm["from_club_level"] = club_level_before(vals, tm["from_club_id"], tm["transfer_date"], tm["player_id"])
    tm["to_club_level"] = club_level_before(vals, tm["to_club_id"], tm["transfer_date"], tm["player_id"])
    tm["club_level_delta"] = tm["to_club_level"] - tm["from_club_level"]
    n = len(tm)
    n_from, n_to = tm["from_club_level"].notna().sum(), tm["to_club_level"].notna().sum()
    print(f"club_level coverage: from_club={n_from}/{n} ({n_from / n:.1%}), to_club={n_to}/{n} ({n_to / n:.1%})")
    return tm


def build_nationality_group(series: pd.Series, top_k: int = 25) -> pd.Series:
    """Keep the top_k most common nationalities, group the rest as Other."""
    s = series.fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"})
    top = s.value_counts().head(top_k).index
    return s.where(s.isin(top), other="Other")


# ---------------------------------------------------------------------------
# Engineered features
# ---------------------------------------------------------------------------

def add_position_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flags from Transfermarkt `position` ("Goalkeeper", "Defender", "Midfield",
    "Attack"). The original code compared it with FBref codes (GK, DF, MF, FW),
    which never matched, so every flag was zero."""
    df["is_goalkeeper"] = df["position"].eq("Goalkeeper").astype(int)
    df["is_defender"] = df["position"].eq("Defender").astype(int)
    df["is_midfielder"] = df["position"].eq("Midfield").astype(int)
    df["is_forward"] = df["position"].eq("Attack").astype(int)
    flags = ["is_goalkeeper", "is_defender", "is_midfielder", "is_forward"]
    print("position flags: " + ", ".join(f"{c}={int(df[c].sum())}" for c in flags) + f" of {len(df)} rows")
    return df


POSITION_INTERACTIONS = {
    # name: (stat, position flag). Lets Ridge use per-position coefficients.
    "fw_goals": ("goals", "is_forward"),
    "mf_goals": ("goals", "is_midfielder"),
    "fw_assists": ("assists", "is_forward"),
    "mf_assists": ("assists", "is_midfielder"),
    "fw_xg": ("expected_goals", "is_forward"),
    "mf_xg": ("expected_goals", "is_midfielder"),
    "fw_xag": ("expected_assisted_goals", "is_forward"),
    "mf_xag": ("expected_assisted_goals", "is_midfielder"),
    "fw_npxg_xag": ("non_penalty_xg_plus_xag", "is_forward"),
    "mf_npxg_xag": ("non_penalty_xg_plus_xag", "is_midfielder"),
    "mf_key_passes": ("key_passes", "is_midfielder"),
    "mf_progressive_passes": ("progressive_passes", "is_midfielder"),
    "df_tackles_won": ("tackles_won", "is_defender"),
    "df_interceptions": ("interceptions", "is_defender"),
    "mf_interceptions": ("interceptions", "is_midfielder"),
    "df_clearances": ("clearances", "is_defender"),
    "df_tkl_int": ("tackles_plus_interceptions", "is_defender"),
    "mf_tkl_int": ("tackles_plus_interceptions", "is_midfielder"),
    "gk_saves": ("saves", "is_goalkeeper"),
    "gk_clean_sheets": ("clean_sheets", "is_goalkeeper"),
}

TOP_NATIONS = ["France", "Germany", "Spain", "England", "Brazil", "Argentina", "Portugal"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise engineered features. Only `star_player_score` and
    `breakout_potential` use column-wide std, and neither is a model input."""
    df = df.copy()
    df["season_index"] = df["season_end_year"] - df["season_end_year"].min()

    # performance
    df["goal_overperformance"] = (df["goals"] - df["expected_goals"]).fillna(0)
    df["assist_efficiency"] = df["assists"] / (df["minutes"] / 90 + 1)
    df["offensive_contribution"] = df["goals"] + df["assists"] + (df["expected_goals"].fillna(0) / 2)
    df["xg_differential"] = (df["goals"] - df["expected_goals"]).fillna(0)
    df["attacking_efficiency"] = (df["goals"] + df["assists"]) / (df["minutes"] + 1)

    # age
    df["age_to_peak"] = np.abs(df["age_at_transfer"] - 28.5)
    df["is_peak_age"] = ((df["age_at_transfer"] >= 26) & (df["age_at_transfer"] <= 31)).astype(int)
    df["is_young"] = (df["age_at_transfer"] < 23).astype(int)
    df["age_potential_score"] = np.where(
        df["age_at_transfer"] < 25,
        df["goals_p_90"] * (1 + (25 - df["age_at_transfer"]) / 20),
        df["goals_p_90"],
    )

    # playing time
    df["is_starter"] = (df["minutes"] >= 1500).astype(int)
    df["playing_time_per_game"] = df["minutes"] / (df["nineties"] + 1)
    df["is_high_volume"] = (df["minutes"] >= 2500).astype(int)

    # defending
    df["defensive_actions"] = (
        df["tackles_won"].fillna(0) + df["interceptions"].fillna(0) + df["clearances"].fillna(0)
    )
    df["defensive_per_90"] = df["defensive_actions"] / (df["nineties"] + 1)
    df["clean_sheet_ratio"] = df["clean_sheets"].fillna(0) / (df["nineties"] + 1)

    df["is_top_nation"] = df["nationality_group"].isin(TOP_NATIONS).astype(int)

    # position
    df = add_position_flags(df)
    for name, (stat, flag) in POSITION_INTERACTIONS.items():
        df[name] = df[stat].fillna(0) * df[flag]

    # passing
    df["passing_quality"] = df["key_passes"].fillna(0) / (df["nineties"] + 1)
    df["progressive_passing_per_90"] = df["progressive_passes"].fillna(0) / (df["nineties"] + 1)

    # compound
    df["overall_impact"] = (df["offensive_contribution"] + df["defensive_actions"] * 0.3) / (df["nineties"] + 1)
    df["star_player_score"] = (
        df["is_peak_age"] * 2 + df["is_high_volume"] * 2
        + (df["goals_p_90"] / df["goals_p_90"].std()) + df["is_starter"]
    ).fillna(0)
    df["breakout_potential"] = (
        df["is_young"] * 2
        + (df["xg_differential"] / df["xg_differential"].std())
        + df["age_potential_score"] / (df["age_potential_score"].std() + 1)
    ).fillna(0)
    return df


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    """Intermediate parquets -> data/final/player_market_value_dataset.parquet."""
    transfers = pd.read_parquet(TM_TRANSFERS)
    players = pd.read_parquet(TM_PLAYERS)
    fbref = pd.read_parquet(FBREF_STATS)

    tm, club_history = prepare_transfers(transfers, players)
    print("Computing point-in-time club levels from player valuations...")
    tm = add_club_levels(tm, club_history)
    tm["nationality_group"] = build_nationality_group(tm["country_of_citizenship"], top_k=25)

    merged, stats = join_fbref(tm, fbref)

    # model-ready rows: target, age and at least MIN_MINUTES of FBref minutes
    merged = merged[
        merged["market_value_eur_at_transfer"].notna()
        & merged["minutes"].notna()
        & (merged["minutes"] >= MIN_MINUTES)
        & merged["age_at_transfer"].notna()
    ].copy()

    final = engineer_features(merged)[DATASET_COLUMNS].copy()
    final["log_market_value"] = np.log1p(final["market_value_eur_at_transfer"])

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    final.to_parquet(DATASET, index=False)
    stats["model_rows"] = int(len(final))
    BUILD_STATS.write_text(json.dumps(stats, indent=2))

    print(f"Saved: {DATASET.name}  rows={len(final)}, columns={final.shape[1]}")
    print(f"Model rows as share of transfers in window: {len(final) / stats['transfers_in_window']:.2%}")
    print("Rows by season:", final.groupby("season_end_year").size().to_dict())
    return final


def build_all_data() -> pd.DataFrame:
    """Raw CSVs -> intermediate parquets -> final dataset."""
    from .clean import build_fbref
    from .load import load_transfermarkt

    load_transfermarkt()
    build_fbref()
    return build_dataset()


if __name__ == "__main__":
    build_all_data()
