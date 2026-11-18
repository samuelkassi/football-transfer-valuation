from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTER_DIR = PROJECT_ROOT / "data" / "intermediate"
FINAL_DIR = PROJECT_ROOT / "data" / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = FINAL_DIR / "player_market_value_dataset.parquet"

# transfers span from 17-18 to 24-25 -> end years 18 to 25

SEASON_END_YEARS = list(range(2018, 2026))

# helper functions

# gets rid of inconsistencies in player names.
# Unicode-normalises accents/diacritics first (é → e, ñ → n, ć → c, ü → u, ...)
# so that names like "Jules Koundé"/"Jules Kounde" and "Raúl Albiol"/"Raul Albiol"
# hash to the same key. Diagnostic showed this picks up ~175 extra players
# (+5.8%) that the old regex alone would silently drop during the FBref join.
def clean_name(s: str) -> str | None:
    if pd.isna(s):
        return None
    s = str(s).strip()
    # NFKD decomposes accented characters into base + combining mark;
    # encoding to ASCII with errors="ignore" then drops the combining marks.
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s or None

# removes any incosistencies in league names
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

# finds a column with a name from a list of possible names in a data frame
def first_existing(df: pd.DataFrame, candidates: list[str], *, required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"None of these columns exist: {candidates}. Available: {df.columns.tolist()}")
    return None

# makes sure a column is in a datetime type, modifying the DF in place when its not
def ensure_datetime(df: pd.DataFrame, col: str) -> None:

    if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors="coerce")


# groups lesser known nationalites as other, and keep others as they are
def build_nationality_group(series: pd.Series, top_k: int = 25) -> pd.Series:
    s = series.fillna("Unknown").astype(str).str.strip()
    s = s.replace({"": "Unknown"})
    top = s.value_counts().head(top_k).index
    return s.where(s.isin(top), other="Other")


def main():
    # loading intermediate data
    fbref = pd.read_parquet(INTER_DIR / "fbref_player_season_stats.parquet")
    transfers = pd.read_parquet(INTER_DIR / "tm_transfers.parquet")
    players = pd.read_parquet(INTER_DIR / "tm_players.parquet")

    # standardising key column names across possible versions
    
    # transfers
    transfer_player_col = first_existing(transfers, ["player", "player_name", "name"])
    transfer_fee_col = first_existing(transfers, ["transfer_fee_eur", "transfer_fee", "fee", "transfer_fee_in_eur"], required=False)
    mv_at_transfer_col = first_existing(
        transfers,
        ["market_value_eur_at_transfer", "market_value_in_eur", "market_value", "market_value_eur"],
    )
    transfer_date_col = first_existing(transfers, ["transfer_date", "date"])
    season_end_col = first_existing(transfers, ["season_end_year", "transfer_season_end_year", "season_end"])

    # players
    dob_col = first_existing(players, ["date_of_birth", "dob", "birth_date"])
    citizenship_col = first_existing(players, ["country_of_citizenship", "nationality", "citizenship"], required=False)
    contract_exp_col = first_existing(players, ["contract_expiration_date", "contract_expiry", "contract_expires"], required=False)

    # Ensure datetimes
    ensure_datetime(transfers, transfer_date_col)
    ensure_datetime(players, dob_col)
    if contract_exp_col:
        ensure_datetime(players, contract_exp_col)

    # Compute next_transfer_date from the FULL (unfiltered) transfer history
    # so transfers near the end of our SEASON_END_YEARS window still see a
    # following transfer if the player actually moved again later. Used
    # downstream to fix the broken contract_years_remaining calc.
    pid_col_raw = first_existing(transfers, ["player_id", "id", "tm_player_id"])
    transfers = transfers.sort_values([pid_col_raw, transfer_date_col]).copy()
    transfers["next_transfer_date"] = (
        transfers.groupby(pid_col_raw)[transfer_date_col].shift(-1)
    )

    # filtering out transfers outside fbref data range
    transfers = transfers[transfers[season_end_col].isin(SEASON_END_YEARS)].copy()

    # minimal transfers df with consistent names
    tr = transfers.rename(
        columns={
            transfer_player_col: "player",
            transfer_date_col: "transfer_date",
            season_end_col: "season_end_year",
            mv_at_transfer_col: "market_value_eur_at_transfer",
        }
    ).copy()

    # Ensure a consistent transfer fee column exists; set to NaN when missing
    if transfer_fee_col and transfer_fee_col in transfers.columns:
        tr["transfer_fee_eur"] = transfers[transfer_fee_col]
    else:
        tr["transfer_fee_eur"] = np.nan

    # make the names in both datasets consistent for merging
    fbref["player_clean"] = fbref["player"].apply(clean_name)
    tr["player_clean"] = tr["player"].apply(clean_name)

    # normalising league
    if "comp" in fbref.columns:
        fbref["league"] = fbref["comp"].apply(normalise_league)
    else:
        fbref["league"] = "Unknown"

    # Join transfers -> players by player_id 
    if "player_id" not in tr.columns:
        pid_col = first_existing(transfers, ["player_id", "id", "tm_player_id"])
        tr["player_id"] = transfers[pid_col]

    tm = tr.merge(players, on="player_id", how="left", suffixes=("", "_player"))

    # calculating variables
    tm["age_at_transfer"] = (tm["transfer_date"] - tm[dob_col]).dt.days / 365.25
    tm["age_squared"] = tm["age_at_transfer"] ** 2

    # ------------------------------------------------------------------
    # contract_years_remaining  (FIXED -- see bug note below)
    #
    # Previously this was computed as (snapshot_contract_expiry - transfer_date).
    # That's a semantic bug: players.csv contains a SINGLE snapshot of each
    # player's current contract (scraped ~early 2025 -- contract_expiration_date
    # values only live in 2023..2034). A 2018 transfer was being compared to
    # the player's 2026 contract at a totally different club, producing garbage
    # like "17 years remaining" at Dante's 2009 transfer.
    #
    # Fixed logic:
    #   1) Preferred -- years until this player's NEXT transfer. That's the
    #      actual stay length, which is a lower bound / reasonable proxy for
    #      the contract they signed at the current transfer.
    #   2) Fallback -- for the player's latest transfer (no subsequent move
    #      recorded), use the snapshot calc, but only if the transfer_date is
    #      recent enough (post-2023-01-01) for the snapshot contract to
    #      plausibly be the one they signed here.
    #   3) Else NaN -- signal explicitly missing.
    #
    # Cap at 6 years: rare to sign longer than that originally; long gaps
    # between transfers reflect extensions, not the original deal.
    # ------------------------------------------------------------------
    SNAPSHOT_VALID_FROM = pd.Timestamp("2023-01-01")

    if contract_exp_col and contract_exp_col in tm.columns:
        # (1) primary signal: years until next transfer event
        years_until_next = (tm["next_transfer_date"] - tm["transfer_date"]).dt.days / 365.25

        # (2) fallback: snapshot-based, only for recent transfers
        years_to_snapshot = (tm[contract_exp_col] - tm["transfer_date"]).dt.days / 365.25
        snapshot_valid = tm["transfer_date"] >= SNAPSHOT_VALID_FROM

        tm["contract_years_remaining"] = years_until_next.where(
            years_until_next.notna(),
            years_to_snapshot.where(snapshot_valid),
        )
        tm["contract_years_remaining"] = tm["contract_years_remaining"].clip(lower=0, upper=6)

        # Quick diagnostic so we can see coverage after the fix.
        n_total = len(tm)
        n_from_next = years_until_next.notna().sum()
        n_from_snap = (years_until_next.isna() & years_to_snapshot.notna() & snapshot_valid).sum()
        n_missing = tm["contract_years_remaining"].isna().sum()
        print(
            f"contract_years_remaining: next-transfer={n_from_next}, "
            f"snapshot-fallback={n_from_snap}, missing={n_missing} / {n_total} "
            f"(populated {100*(n_total-n_missing)/n_total:.1f}%)"
        )
    else:
        tm["contract_years_remaining"] = np.nan

    # ------------------------------------------------------------------
    # Club-level features
    #
    # Score each club by the median market value of its players in a given
    # year (from tm_player_valuations.parquet -- 496k time-stamped records).
    # A transfer then gets `from_club_level` / `to_club_level` = log1p(median
    # MV of that club that year), with all-time median as fallback for sparse
    # years. Captures "where is this player moving through" in a way that
    # neither league one-hots nor raw stats do (Man City and Brighton both
    # get `league=Premier League` but very different club levels).
    #
    # Uses only club identity at inference time (user provides the club),
    # so it's compatible with predicting theoretical-player values.
    # ------------------------------------------------------------------
    print("Computing club-level tiers from player valuations...")
    vals = pd.read_parquet(INTER_DIR / "tm_player_valuations.parquet")
    vals["year"] = pd.to_datetime(vals["date"], errors="coerce").dt.year
    vals = vals.dropna(subset=["current_club_id", "market_value_eur", "year"])

    club_year_tier = (
        vals.groupby(["current_club_id", "year"])["market_value_eur"]
        .median()
        .reset_index()
        .rename(columns={"market_value_eur": "club_tier_yr"})
    )
    club_alltime_tier = (
        vals.groupby("current_club_id")["market_value_eur"]
        .median()
        .reset_index()
        .rename(columns={"market_value_eur": "club_tier_all"})
    )

    def _join_club_level(df: pd.DataFrame, club_col: str) -> pd.Series:
        """Return log1p(per-year median, fall back all-time), same index as df."""
        tmp = pd.DataFrame({
            club_col: df[club_col].values,
            "year": df["transfer_date"].dt.year.values,
        })
        tmp = tmp.merge(
            club_year_tier.rename(columns={"current_club_id": club_col}),
            on=[club_col, "year"], how="left",
        )
        tmp = tmp.merge(
            club_alltime_tier.rename(columns={"current_club_id": club_col}),
            on=club_col, how="left",
        )
        coalesced = tmp["club_tier_yr"].fillna(tmp["club_tier_all"])
        return np.log1p(coalesced).values

    tm["from_club_level"] = _join_club_level(tm, "from_club_id")
    tm["to_club_level"]   = _join_club_level(tm, "to_club_id")
    tm["club_level_delta"] = tm["to_club_level"] - tm["from_club_level"]

    # diagnostic
    n_from_ok = tm["from_club_level"].notna().sum()
    n_to_ok   = tm["to_club_level"].notna().sum()
    print(
        f"club_level coverage: from_club={n_from_ok}/{len(tm)} "
        f"({100*n_from_ok/len(tm):.1f}%), to_club={n_to_ok}/{len(tm)} "
        f"({100*n_to_ok/len(tm):.1f}%)"
    )

    # building nationality groups
    if citizenship_col and citizenship_col in tm.columns:
        tm["country_of_citizenship"] = tm[citizenship_col]
    else:
        tm["country_of_citizenship"] = np.nan

    tm["nationality_group"] = build_nationality_group(tm["country_of_citizenship"], top_k=25)

    # Map transfer season to the FBref season (previous season stats expected)



    # join FBref season before transfer
    tm["fbref_season"] = tm["season_end_year"] - 1  # previous season stats

    # Join on cleaned name + season + birth year. A name alone is not an
    # identity: several Transfermarkt players share short names ("Fernando",
    # "Marko Ilic"), so a name+season join attaches one player's stats to
    # another player's transfer. FBref `born` must be within 1 year of the
    # Transfermarkt birth year, and a match is dropped as ambiguous when one
    # FBref row would serve more than one Transfermarkt player.
    tm = tm.reset_index(drop=True)
    tm["transfer_row_id"] = np.arange(len(tm))
    tm["birth_year"] = tm[dob_col].dt.year

    candidates = tm.merge(
        fbref.assign(fbref_row_id=np.arange(len(fbref))),
        left_on=["player_clean", "fbref_season"],
        right_on=["player_clean", "season_end_year"],
        how="inner",
        suffixes=("", "_fb"),
    )
    n_name_matches = candidates["transfer_row_id"].nunique()
    born_ok = (candidates["born"] - candidates["birth_year"]).abs() <= 1
    candidates = candidates[born_ok].copy()
    n_born_matches = candidates["transfer_row_id"].nunique()

    ids_per_fbref_row = candidates.groupby("fbref_row_id")["player_id"].transform("nunique")
    rows_per_transfer = candidates.groupby("transfer_row_id")["fbref_row_id"].transform("nunique")
    ambiguous = (ids_per_fbref_row > 1) | (rows_per_transfer > 1)
    candidates = candidates[~ambiguous].copy()

    unmatched = tm[~tm["transfer_row_id"].isin(candidates["transfer_row_id"])]
    merged = pd.concat(
        [candidates.assign(_merge="both"), unmatched.assign(_merge="left_only")],
        ignore_index=True,
    ).sort_values("transfer_row_id")

    # join match rate. Denominator: every Transfermarkt transfer in the
    # 2018-2025 seasons (one row per transfer). Numerator: transfers with a
    # verified (name + season + birth year, unambiguous) FBref row.
    n_transfers = len(tm)
    n_matched = int((merged["_merge"] == "both").sum())
    print(f"Transfers in window: {n_transfers}")
    print(f"  name+season match:            {n_name_matches} ({n_name_matches / n_transfers:.2%})")
    print(f"  + birth year within 1 year:   {n_born_matches} ({n_born_matches / n_transfers:.2%})")
    print(f"  - ambiguous, dropped:         {int(ambiguous.sum())} candidate rows")
    print(f"FBref join match rate: {n_matched / n_transfers:.2%} ({n_matched} / {n_transfers})")

    # join match rate by season
    by_season = (
        merged.assign(matched=(merged["_merge"] == "both"))
        .groupby("season_end_year", dropna=False)["matched"]
        .mean()
        .reset_index()
        .sort_values("season_end_year")
    )
    print("\nJoin match rate by transfer season_end_year:")
    print(by_season.to_string(index=False))

    # Filter to model-ready rows
    # Target present + reasonable minutes + age present
    # Sanity check: FBref minutes column must exist for playing-time filters
    if "minutes" not in merged.columns:
        raise ValueError(
            "Expected 'minutes' from FBref in merged dataset, but it isn't present. "
            "Check your FBref parquet columns."
        )

    merged = merged[
        merged["market_value_eur_at_transfer"].notna()
        & merged["minutes"].notna()
        & (merged["minutes"] >= 450)
        & merged["age_at_transfer"].notna()
    ].copy()

    # Feature engineering: season index
    merged["season_index"] = merged["season_end_year"] - merged["season_end_year"].min()

    # ============================================================================
    # PHASE 2: ADVANCED FEATURE ENGINEERING
    # ============================================================================
    print("\n" + "="*80)
    print("PHASE 2: ADVANCED FEATURE ENGINEERING")
    print("="*80)
    
    # 1. PERFORMANCE METRICS & EFFICIENCY
    print("\n✓ Creating performance efficiency features...")
    
    # Goal overperformance: beats expected goals?
    merged["goal_overperformance"] = (merged["goals"] - merged["expected_goals"]).fillna(0)
    
    # Assist efficiency per 90 minutes
    merged["assist_efficiency"] = merged["assists"] / (merged["minutes"] / 90 + 1)
    
    # Combined offensive contribution
    merged["offensive_contribution"] = merged["goals"] + merged["assists"] + (merged["expected_goals"].fillna(0) / 2)
    
    # Goals against expected - how much they differ from xG
    merged["xg_differential"] = merged["goals"] - merged["expected_goals"]
    merged["xg_differential"] = merged["xg_differential"].fillna(0)
    
    # Attacking efficiency (goals + assists per minute played)
    merged["attacking_efficiency"] = (merged["goals"] + merged["assists"]) / (merged["minutes"] + 1)
    
    # 2. AGE & DEVELOPMENT FEATURES
    print("✓ Creating age and development features...")
    
    # Distance from peak age (peak is 27-31)
    merged["age_to_peak"] = np.abs(merged["age_at_transfer"] - 28.5)
    
    # Is player in peak years?
    merged["is_peak_age"] = ((merged["age_at_transfer"] >= 26) & 
                              (merged["age_at_transfer"] <= 31)).astype(int)
    
    # Young prospect flag
    merged["is_young"] = (merged["age_at_transfer"] < 23).astype(int)
    
    # Age-adjusted potential (younger = higher potential ceiling)
    merged["age_potential_score"] = np.where(
        merged["age_at_transfer"] < 25,
        merged["goals_p_90"] * (1 + (25 - merged["age_at_transfer"]) / 20),  # boost young players
        merged["goals_p_90"]
    )
    
    # 3. PLAYING TIME & CONSISTENCY
    print("✓ Creating playing time features...")
    
    # Regular starter status (>1500 minutes = consistent starter)
    merged["is_starter"] = (merged["minutes"] >= 1500).astype(int)
    
    # Playing time per game (consistency indicator)
    merged["playing_time_per_game"] = merged["minutes"] / (merged["nineties"] + 1)
    
    # High volume indicator (lots of minutes)
    merged["is_high_volume"] = (merged["minutes"] >= 2500).astype(int)
    
    # 4. DEFENSIVE PROFILE (for position weighting)
    print("✓ Creating defensive metrics...")
    
    # Defensive actions (combines tackles, interceptions, clearances)
    merged["defensive_actions"] = (
        merged["tackles_won"].fillna(0) + 
        merged["interceptions"].fillna(0) + 
        merged["clearances"].fillna(0)
    )
    
    # Defensive per 90
    merged["defensive_per_90"] = merged["defensive_actions"] / (merged["nineties"] + 1)
    
    # GK specific: clean sheet ratio
    merged["clean_sheet_ratio"] = merged["clean_sheets"].fillna(0) / (merged["nineties"] + 1)
    
    # 5. LEAGUE & COMPETITION ADJUSTMENTS
    print("✓ Creating league-adjusted features...")
    
    # League quality multiplier (rough estimates based on competitive strength)
    # Rough competitive-strength weights. Big 5 values unchanged; the
    # five newly-added leagues are placeholder estimates -- revisit once
    # the new data is in and you can benchmark against observed valuations.
    league_multipliers = {
        'Premier League':     1.00,
        'La Liga':            0.95,
        'Bundesliga':         0.92,
        'Serie A':            0.90,
        'Ligue 1':            0.88,
        'Primeira Liga':      0.78,
        'Eredivisie':         0.76,
        'Super Lig':          0.72,
        'Jupiler Pro League': 0.70,
        'MLS':                0.65,
        'Other':              0.60,
    }
    merged["league_multiplier"] = merged["league"].map(league_multipliers).fillna(0.60)
    
    # League-adjusted goal scoring (account for league difficulty)
    merged["league_adjusted_goals"] = merged["goals"] * merged["league_multiplier"]
    
    # League-adjusted xG
    merged["league_adjusted_xg"] = merged["expected_goals"].fillna(0) * merged["league_multiplier"]
    
    # 6. INTERNATIONAL PRESTIGE
    print("✓ Creating international prestige features...")
    
    # Top 5 nation flag (these countries have highest commercial value)
    top_nations = ['France', 'Germany', 'Spain', 'England', 'Brazil', 'Argentina', 'Portugal']
    merged["is_top_nation"] = merged["nationality_group"].isin(top_nations).astype(int)
    
    # 7. POSITION-BASED FEATURES
    print("✓ Creating position-based features...")
    
    # Mark position categories
    # `position` comes from Transfermarkt players.csv, whose labels are
    # "Goalkeeper", "Defender", "Midfield" and "Attack". The earlier checks
    # for FBref-style codes ("GK", "DF", "MF", "FW") never matched, so every
    # flag and every position x stat feature below was zero in every row.
    merged["is_goalkeeper"] = merged["position"].eq("Goalkeeper").astype(int)
    merged["is_defender"] = merged["position"].eq("Defender").astype(int)
    merged["is_midfielder"] = merged["position"].eq("Midfield").astype(int)
    merged["is_forward"] = merged["position"].eq("Attack").astype(int)
    print(
        "position flags: "
        + ", ".join(f"{c}={int(merged[c].sum())}" for c in
                    ["is_goalkeeper", "is_defender", "is_midfielder", "is_forward"])
        + f" of {len(merged)} rows"
    )
    
    # 7b. POSITION × STAT INTERACTIONS
    # Without these, xgboost has to spend splits learning that 50 clearances
    # is huge for a DF but noise for a FW; Ridge can't model this at all (one
    # shared coefficient has to work across positions). Masking stat-by-position
    # gives the model per-position coefficients "for free".
    print("✓ Creating position × stat interaction features...")

    # helper -- stat * position flag, safe to nan
    def _mask(stat_col: str, flag_col: str) -> pd.Series:
        return merged[stat_col].fillna(0) * merged[flag_col]

    # --- attacking output: mostly forwards, secondarily midfielders ---
    merged["fw_goals"]      = _mask("goals",                  "is_forward")
    merged["mf_goals"]      = _mask("goals",                  "is_midfielder")
    merged["fw_assists"]    = _mask("assists",                "is_forward")
    merged["mf_assists"]    = _mask("assists",                "is_midfielder")
    merged["fw_xg"]         = _mask("expected_goals",         "is_forward")
    merged["mf_xg"]         = _mask("expected_goals",         "is_midfielder")
    merged["fw_xag"]        = _mask("expected_assisted_goals","is_forward")
    merged["mf_xag"]        = _mask("expected_assisted_goals","is_midfielder")
    merged["fw_npxg_xag"]   = _mask("non_penalty_xg_plus_xag","is_forward")
    merged["mf_npxg_xag"]   = _mask("non_penalty_xg_plus_xag","is_midfielder")

    # --- creative output: midfielders ---
    merged["mf_key_passes"]         = _mask("key_passes",         "is_midfielder")
    merged["mf_progressive_passes"] = _mask("progressive_passes", "is_midfielder")

    # --- defensive output: defenders + defensive mids ---
    merged["df_tackles_won"]  = _mask("tackles_won",                 "is_defender")
    merged["df_interceptions"] = _mask("interceptions",              "is_defender")
    merged["mf_interceptions"] = _mask("interceptions",              "is_midfielder")
    merged["df_clearances"]   = _mask("clearances",                  "is_defender")
    merged["df_tkl_int"]      = _mask("tackles_plus_interceptions",  "is_defender")
    merged["mf_tkl_int"]      = _mask("tackles_plus_interceptions",  "is_midfielder")

    # --- goalkeeping: only meaningful for GKs (zero for everyone else) ---
    merged["gk_saves"]        = _mask("saves",        "is_goalkeeper")
    merged["gk_clean_sheets"] = _mask("clean_sheets", "is_goalkeeper")

    # 8. PASSING & CREATIVITY
    print("✓ Creating passing and creativity features...")
    
    # Passing quality (key passes per 90)
    merged["passing_quality"] = merged["key_passes"].fillna(0) / (merged["nineties"] + 1)
    
    # Progressive passing per 90
    merged["progressive_passing_per_90"] = merged["progressive_passes"].fillna(0) / (merged["nineties"] + 1)
    
    # 9. CONTRACT & STABILITY
    print("✓ Creating contract features...")
    
    # Players with long contracts are more stable/valuable
    merged["has_long_contract"] = (merged["contract_years_remaining"] >= 3).astype(int)
    
    # Contract expiration urgency (players expiring soon might be cheaper)
    merged["contract_expiring_soon"] = (merged["contract_years_remaining"] <= 1).astype(int)
    merged["contract_expiring_soon"] = merged["contract_expiring_soon"].fillna(0)
    
    # 10. COMPOUND SCORING FEATURES (multivariate)
    print("✓ Creating compound scoring features...")
    
    # Overall impact score: combines goals, assists, and defensive play
    merged["overall_impact"] = (
        merged["offensive_contribution"] + 
        (merged["defensive_actions"] * 0.3)
    ) / (merged["nineties"] + 1)
    
    # "Star player" composite: peak age, high volume, strong stats
    merged["star_player_score"] = (
        merged["is_peak_age"] * 2 +
        merged["is_high_volume"] * 2 +
        (merged["goals_p_90"] / merged["goals_p_90"].std()) * 1 +  # z-score
        merged["is_starter"]
    )
    merged["star_player_score"] = merged["star_player_score"].fillna(0)
    
    # Breakout player score: young + high performance
    merged["breakout_potential"] = (
        merged["is_young"] * 2 +
        (merged["xg_differential"] / merged["xg_differential"].std()) +  # beating xG
        merged["age_potential_score"] / (merged["age_potential_score"].std() + 1)
    )
    merged["breakout_potential"] = merged["breakout_potential"].fillna(0)
    
    print("\n✅ Feature engineering complete!")
    print(f"   Total features created in this phase: 40+")

    # Feature selection
    
    feature_cols = [
        # identifiers
        "player",
        "transfer_date",
        "season_end_year",
        "season_index",
        # target
        "market_value_eur_at_transfer",
        # player info
        "age_at_transfer",
        "age_squared",
        "height_in_cm",
        "contract_years_remaining",
        "position",
        "sub_position",
        "foot",
        # league + nationality
        "league",
        "nationality_group",
        # club context (log1p of per-club, per-year median player MV)
        "from_club_level",
        "to_club_level",
        "club_level_delta",
        # playing time
        "minutes",
        "nineties",

        # attacking
        "goals",
        "assists",
        "expected_goals",
        "goals_and_assists",
        "goals_p_90",
        "assists_p_90",
        # modern attacking / creation metrics (added from expanded fbref source)
        "expected_assisted_goals",     # xAG: goal-leading passes
        "non_penalty_xg",              # npxG: xG excluding penalties
        "non_penalty_xg_plus_xag",     # npxG + xAG: overall non-penalty creation

        # passing
        "progressive_passes",
        "key_passes",

        # defending
        "tackles_won",
        "interceptions",
        "clearances",
        "tackles_plus_interceptions",  # Tkl+Int: combined defensive actions

        # goalkeeping
        "clean_sheets",
        "saves",
        
        # ========== PHASE 2 ENGINEERED FEATURES ==========
        
        # Performance & Efficiency
        "goal_overperformance",
        "assist_efficiency",
        "offensive_contribution",
        "xg_differential",
        "attacking_efficiency",
        
        # Age & Development
        "age_to_peak",
        "is_peak_age",
        "is_young",
        "age_potential_score",
        
        # Playing Time & Consistency
        "is_starter",
        "playing_time_per_game",
        "is_high_volume",
        
        # Defensive Profile
        "defensive_actions",
        "defensive_per_90",
        "clean_sheet_ratio",
        
        # League & Competition
        # (league_multiplier / league_adjusted_* removed -- redundant with
        #  the one-hot `league` feature, which dominated importance instead.
        #  Keeping the computation upstream for now; just not feeding it to
        #  the model.)

        # International Prestige
        "is_top_nation",
        
        # Position-based
        "is_goalkeeper",
        "is_defender",
        "is_midfielder",
        "is_forward",

        # Position x stat interactions (see section 7b above)
        # attacking (FW / MF)
        "fw_goals", "mf_goals",
        "fw_assists", "mf_assists",
        "fw_xg", "mf_xg",
        "fw_xag", "mf_xag",
        "fw_npxg_xag", "mf_npxg_xag",
        # creative (MF)
        "mf_key_passes",
        "mf_progressive_passes",
        # defensive (DF / MF)
        "df_tackles_won",
        "df_interceptions", "mf_interceptions",
        "df_clearances",
        "df_tkl_int", "mf_tkl_int",
        # goalkeeping (GK)
        "gk_saves",
        "gk_clean_sheets",
        
        # Passing & Creativity
        "passing_quality",
        "progressive_passing_per_90",
        
        # Contract & Stability
        "has_long_contract",
        "contract_expiring_soon",
        
        # Compound Scores
        "overall_impact",
        "star_player_score",
        "breakout_potential",
    ]
    # Keep only features that actually exist in the merged dataframe
    feature_cols = [c for c in feature_cols if c in merged.columns]
    final_df = merged[feature_cols].copy()

    # Log target for modelling
    final_df["log_market_value"] = np.log1p(final_df["market_value_eur_at_transfer"])

    # Save the prepared dataset to parquet for downstream modelling
    final_df.to_parquet(OUT_PATH, index=False)

    
    print(f"\nSaved: {OUT_PATH}")
    print(f"Rows: {final_df.shape[0]}, Columns: {final_df.shape[1]}")
    print(f"Model rows as share of transfers in window: {final_df.shape[0] / n_transfers:.2%}")
    print("\nRows by transfer season_end_year:")
    print(final_df.groupby("season_end_year").size().to_string())

    if "league" in final_df.columns:
        print("\nLeague distribution (top 10):")
        print(final_df["league"].value_counts(dropna=False).head(10).to_string())

    if "nationality_group" in final_df.columns:
        print("\nNationality group distribution (top 15):")
        print(final_df["nationality_group"].value_counts(dropna=False).head(15).to_string())



if __name__ == "__main__":
    main()