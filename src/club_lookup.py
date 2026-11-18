"""Club-name → club_level lookup for inference-time scoring.

At training time, `build_dataset.py` computes per-transfer `from_club_level` /
`to_club_level` features directly from `from_club_id` / `to_club_id`, using
log1p(per-year median player MV) with an all-time median fallback.

At inference time — e.g. in the experiment notebook — the user won't know a
club's numeric ID, so this module maps **club names** (with a bit of fuzzy
matching) to the same log1p club-level value the model was trained on.

Typical use:
    from club_lookup import get_club_level, list_clubs
    get_club_level("Manchester City", year=2024)        # -> float (log1p)
    list_clubs(search="madrid", limit=10)               # -> DataFrame
"""

from __future__ import annotations

import difflib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTER_DIR = PROJECT_ROOT / "data" / "intermediate"

VALUATIONS_PATH = INTER_DIR / "tm_player_valuations.parquet"
PLAYERS_PATH = INTER_DIR / "tm_players.parquet"

# Sentinel year used by the internal index to mean "all-time median" — the
# fallback row used when no per-year data exists for the requested year.
_ALLTIME_YEAR = 0


# ============================================================================
# Index construction (cached)
# ============================================================================

@lru_cache(maxsize=1)
def _load_club_index() -> pd.DataFrame:
    """Build a (club_id, club_name, year, median_eur, level_log) DataFrame.

    Combines the two TM parquets:
      * `tm_player_valuations.parquet` — time-stamped valuations
          → per-(club_id, year) median + per-club all-time median
      * `tm_players.parquet` — maps `current_club_id` → `current_club_name`

    For each (club_id, year) pair plus a sentinel year=0 row containing the
    all-time median, we store `level_log = log1p(median_eur)` so callers can
    grab either directly.
    """
    if not VALUATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing valuations parquet at {VALUATIONS_PATH}. "
            "Run src/transfermarkt.py to rebuild it."
        )
    if not PLAYERS_PATH.exists():
        raise FileNotFoundError(
            f"Missing players parquet at {PLAYERS_PATH}. "
            "Run src/transfermarkt.py to rebuild it."
        )

    vals = pd.read_parquet(VALUATIONS_PATH)
    vals["year"] = pd.to_datetime(vals["date"], errors="coerce").dt.year
    vals = vals.dropna(subset=["current_club_id", "market_value_eur", "year"])
    vals["current_club_id"] = vals["current_club_id"].astype(int)
    vals["year"] = vals["year"].astype(int)

    # per-(club, year) median
    by_year = (
        vals.groupby(["current_club_id", "year"])["market_value_eur"]
        .median()
        .reset_index()
        .rename(columns={"current_club_id": "club_id", "market_value_eur": "median_eur"})
    )

    # per-club all-time median, encoded as year=_ALLTIME_YEAR sentinel row
    all_time = (
        vals.groupby("current_club_id")["market_value_eur"]
        .median()
        .reset_index()
        .rename(columns={"current_club_id": "club_id", "market_value_eur": "median_eur"})
    )
    all_time["year"] = _ALLTIME_YEAR

    combined = pd.concat([by_year, all_time], ignore_index=True, sort=False)

    # attach names from tm_players.parquet. A club can appear under multiple
    # names there if players switch between subsidiaries; take the most
    # common name per club_id.
    players = pd.read_parquet(PLAYERS_PATH)
    players = players.dropna(subset=["current_club_id", "current_club_name"]).copy()
    players["current_club_id"] = players["current_club_id"].astype(int)
    name_by_id = (
        players.groupby("current_club_id")["current_club_name"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iat[0])
        .rename("club_name")
        .reset_index()
        .rename(columns={"current_club_id": "club_id"})
    )

    combined = combined.merge(name_by_id, on="club_id", how="left")
    combined["club_name"] = combined["club_name"].fillna(
        combined["club_id"].astype(str).radd("club_")
    )

    combined["level_log"] = np.log1p(combined["median_eur"].astype(float))

    # sort for nicer browsing
    combined = combined.sort_values(["club_name", "year"]).reset_index(drop=True)
    return combined


# ============================================================================
# Fuzzy name matching
# ============================================================================

def _fuzzy_match(
    name: str,
    candidates: pd.Series,
    cutoff: float = 0.7,
) -> str | None:
    """Find the best club name match in *candidates* for *name*.

    Strategy, in order of preference:
      1. Exact case-insensitive match.
      2. Case-insensitive substring containment (query inside candidate).
      3. ``difflib.get_close_matches`` above *cutoff*.

    Returns the matched candidate name, or None if nothing clears the cutoff.
    """
    if not isinstance(name, str) or not name.strip():
        return None

    needle = name.strip().lower()

    # Keep candidates as strings only
    cand_list: list[str] = [str(c) for c in candidates if isinstance(c, str) and c.strip()]
    cand_lower = [c.lower() for c in cand_list]

    # (1) exact case-insensitive
    for c_lower, c in zip(cand_lower, cand_list):
        if c_lower == needle:
            return c

    # (2) substring containment
    contains = [c for c_lower, c in zip(cand_lower, cand_list) if needle in c_lower]
    if contains:
        # prefer the shortest — "Madrid" -> "Real Madrid" rather than
        # "Atlético Madrid Feminino"
        contains.sort(key=len)
        return contains[0]

    # (3) difflib
    close = difflib.get_close_matches(name, cand_list, n=1, cutoff=cutoff)
    if close:
        return close[0]

    return None


# ============================================================================
# Public API
# ============================================================================

def list_clubs(search: str | None = None, limit: int = 50) -> pd.DataFrame:
    """List distinct clubs in the index, optionally filtered by name substring.

    Returns columns: ``club_id``, ``club_name``, ``n_years``, ``alltime_level_log``,
    ``alltime_median_eur``. Useful for browsing in the notebook:

        >>> list_clubs("madrid")
        >>> list_clubs()          # first 50 clubs alphabetically
    """
    idx = _load_club_index()
    alltime = idx[idx["year"] == _ALLTIME_YEAR].copy()
    alltime = alltime.rename(
        columns={"level_log": "alltime_level_log", "median_eur": "alltime_median_eur"}
    )

    years_per_club = (
        idx[idx["year"] != _ALLTIME_YEAR]
        .groupby("club_id")["year"]
        .nunique()
        .rename("n_years")
        .reset_index()
    )
    out = alltime.merge(years_per_club, on="club_id", how="left")
    out["n_years"] = out["n_years"].fillna(0).astype(int)
    out = out[["club_id", "club_name", "n_years", "alltime_level_log", "alltime_median_eur"]]

    if search:
        needle = search.strip().lower()
        mask = out["club_name"].str.lower().str.contains(needle, na=False)
        out = out[mask]

    out = out.sort_values("alltime_median_eur", ascending=False).reset_index(drop=True)
    if limit is not None and limit > 0:
        out = out.head(limit)
    return out


def get_club_level(
    club_name: str,
    year: int | None = None,
    return_eur: bool = False,
    fuzzy_cutoff: float = 0.7,
) -> float | None:
    """Return the log1p(median MV) club-level score the training pipeline used.

    Parameters
    ----------
    club_name : str
        Display name of the club (e.g. "Manchester City"). Fuzzy matched.
    year : int, optional
        Season/calendar year of the (hypothetical) transfer. If no data for
        that year exists, falls back to the all-time median.
    return_eur : bool
        If True, also return the raw median euro value alongside the log1p
        feature as a tuple ``(level_log, median_eur)``.
    fuzzy_cutoff : float
        Similarity threshold passed to difflib for fallback matching.

    Returns
    -------
    float or None
        The log1p level (what models_v2 expects), or None if no match.
        If ``return_eur`` is True, returns a tuple; if no match, ``(None, None)``.
    """
    idx = _load_club_index()

    # Build the unique list of known club names once
    unique_names = idx[["club_name"]].drop_duplicates()["club_name"]

    matched = _fuzzy_match(club_name, unique_names, cutoff=fuzzy_cutoff)
    if matched is None:
        return (None, None) if return_eur else None

    # Only the rows for this club
    club_rows = idx[idx["club_name"] == matched]

    # Try the requested year first, then fall back to the all-time sentinel
    chosen = None
    if year is not None:
        hit = club_rows[club_rows["year"] == int(year)]
        if not hit.empty:
            chosen = hit.iloc[0]

    if chosen is None:
        hit = club_rows[club_rows["year"] == _ALLTIME_YEAR]
        if hit.empty:
            # No all-time row (shouldn't happen, but be defensive) — use the
            # median of whatever per-year rows we do have.
            if club_rows.empty:
                return (None, None) if return_eur else None
            level_log = float(club_rows["level_log"].median())
            median_eur = float(club_rows["median_eur"].median())
            return (level_log, median_eur) if return_eur else level_log
        chosen = hit.iloc[0]

    level_log = float(chosen["level_log"])
    median_eur = float(chosen["median_eur"])
    return (level_log, median_eur) if return_eur else level_log


def resolve_club_name(club_name: str, fuzzy_cutoff: float = 0.7) -> str | None:
    """Return the canonical club name the index holds for *club_name*, or None.

    Useful for sanity-checking what name the fuzzy matcher actually picked up:

        >>> resolve_club_name("man city")
        'Manchester City'
    """
    idx = _load_club_index()
    unique_names = idx[["club_name"]].drop_duplicates()["club_name"]
    return _fuzzy_match(club_name, unique_names, cutoff=fuzzy_cutoff)


if __name__ == "__main__":
    # quick smoke test
    print("Top 10 clubs by all-time median MV:")
    print(list_clubs(limit=10).to_string(index=False))
    print()
    for q in ["Manchester City", "Real Madrid", "Brighton", "man utd", "nonexistent fc"]:
        lvl = get_club_level(q, year=2024, return_eur=True)
        print(f"  {q!r:30s} -> resolved={resolve_club_name(q)!r}, level={lvl}")
