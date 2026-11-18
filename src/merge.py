"""Join Transfermarkt transfers to the FBref season before each transfer."""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from .config import SEASON_END_YEARS


def clean_name(s: str) -> str | None:
    """Accent-free, lower-case, letters-and-spaces name key.

    NFKD splits accented characters into base + combining mark, and the ASCII
    encode drops the marks, so "Jules Koundé" and "Jules Kounde" match.
    """
    if pd.isna(s):
        return None
    s = unicodedata.normalize("NFKD", str(s).strip()).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z\s]", "", s.lower())
    s = re.sub(r"\s+", " ", s)
    return s or None


def normalise_league(comp: str) -> str:
    if pd.isna(comp):
        return "Unknown"
    comp = str(comp).lower()
    if "premier" in comp:
        return "Premier League"
    if "la liga" in comp or "laliga" in comp:
        return "La Liga"
    if "bundesliga" in comp:
        return "Bundesliga"
    if "serie a" in comp:
        return "Serie A"
    if "ligue 1" in comp:
        return "Ligue 1"
    if "mls" in comp or "major league soccer" in comp:
        return "MLS"
    if "eredivisie" in comp:
        return "Eredivisie"
    if "primeira" in comp or "liga portugal" in comp:
        return "Primeira Liga"
    if "jupiler" in comp or "pro league" in comp:
        return "Jupiler Pro League"
    if "super lig" in comp or "süper lig" in comp:
        return "Super Lig"
    return "Other"


def prepare_transfers(transfers: pd.DataFrame, players: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transfers in the study window joined to player attributes.

    Returns (transfers in window with player info and age, full move history).
    The full history (all seasons) is what dates each player's club spells.
    """
    transfers = transfers.copy()
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
    players = players.copy()
    players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")

    transfers = transfers.sort_values(["player_id", "transfer_date"]).copy()
    club_history = transfers[["player_id", "transfer_date", "from_club_id", "to_club_id"]].copy()

    tr = transfers[transfers["season_end_year"].isin(SEASON_END_YEARS)].copy()
    tr["player_clean"] = tr["player"].apply(clean_name)

    tm = tr.merge(players, on="player_id", how="left", suffixes=("", "_player"))
    tm["age_at_transfer"] = (tm["transfer_date"] - tm["date_of_birth"]).dt.days / 365.25
    tm["age_squared"] = tm["age_at_transfer"] ** 2
    return tm, club_history


def join_fbref(tm: pd.DataFrame, fbref: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach the FBref row for the season before each transfer.

    A name alone is not an identity: several Transfermarkt players share short
    names ("Fernando", "Marko Ilic"), so a name + season join can attach one
    player's stats to another player's transfer. A match therefore also needs
    FBref `born` within 1 year of the Transfermarkt birth year, and is dropped
    as ambiguous when one FBref row would serve more than one Transfermarkt
    player (or one transfer would get more than one FBref row).

    Returns (all transfers, matched or not, with `_merge` in {both, left_only};
    join counts). Match rate denominator: every transfer in the window.
    """
    fbref = fbref.copy()
    fbref["player_clean"] = fbref["player"].apply(clean_name)
    fbref["league"] = fbref["comp"].apply(normalise_league)

    tm = tm.copy()
    tm["fbref_season"] = tm["season_end_year"] - 1  # previous season's stats
    tm = tm.reset_index(drop=True)
    tm["transfer_row_id"] = np.arange(len(tm))
    tm["birth_year"] = tm["date_of_birth"].dt.year

    candidates = tm.merge(
        fbref.assign(fbref_row_id=np.arange(len(fbref))),
        left_on=["player_clean", "fbref_season"],
        right_on=["player_clean", "season_end_year"],
        how="inner",
        suffixes=("", "_fb"),
    )
    n_name = candidates["transfer_row_id"].nunique()
    candidates = candidates[(candidates["born"] - candidates["birth_year"]).abs() <= 1].copy()
    n_born = candidates["transfer_row_id"].nunique()

    ids_per_fbref_row = candidates.groupby("fbref_row_id")["player_id"].transform("nunique")
    rows_per_transfer = candidates.groupby("transfer_row_id")["fbref_row_id"].transform("nunique")
    ambiguous = (ids_per_fbref_row > 1) | (rows_per_transfer > 1)
    candidates = candidates[~ambiguous].copy()

    unmatched = tm[~tm["transfer_row_id"].isin(candidates["transfer_row_id"])]
    merged = pd.concat(
        [candidates.assign(_merge="both"), unmatched.assign(_merge="left_only")],
        ignore_index=True,
    ).sort_values("transfer_row_id")

    n = len(tm)
    n_matched = int((merged["_merge"] == "both").sum())
    print(f"Transfers in window: {n}")
    print(f"  name + season match:          {n_name} ({n_name / n:.2%})")
    print(f"  + birth year within 1 year:   {n_born} ({n_born / n:.2%})")
    print(f"  - ambiguous, dropped:         {int(ambiguous.sum())} candidate rows")
    print(f"FBref join match rate: {n_matched / n:.2%} ({n_matched} / {n})")

    stats = {
        "transfers_in_window": int(n),
        "name_season_matches": int(n_name),
        "birth_year_matches": int(n_born),
        "verified_matches": n_matched,
    }
    return merged, stats
