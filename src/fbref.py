from pathlib import Path
import re
import pandas as pd

# directories which store raw and intermediate data

RAW_DIR = Path("data/raw/fbref")
OUT_DIR = Path("data/intermediate")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "fbref_player_season_stats.parquet"

# Single-source input file. It ships with an EAFC/FIFA merge glued on, which
# we drop below -- we only want the FBref half. Covers 10 leagues across
# seasons 2017-18 -> 2024-25 (2024-25 is partial for some leagues).
# The legacy cleaned_YYYY-YY.csv files in this folder are no longer read.
SOURCE_FILE = RAW_DIR / "fifa_fbref_merged.csv"

# FIFA/EAFC columns (0-34) + the name_parts junk column -- all dropped.
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

# The 10 leagues we keep. Süper Lig replaces the unavailable Championship.
LEAGUES_KEEP = {
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
    "USA-MLS",
    "NED-Eredivisie",
    "POR-Primeira Liga",
    "BEL-Jupiler Pro League",
    "TUR-Super Lig",
}

# fbref league code -> human-readable comp name (matches the old `comp` column
# values the downstream code already knows how to normalise).
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

# Per-90 columns in the source -> pipeline "total" column names we synthesise
# by multiplying the per-90 by Playing Time_90s. Counts are rounded.
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


# helper functions

# used to normalise column names
def to_snake(s: str) -> str:
    s = s.strip()
    s = s.replace("%", "pct")
    s = s.replace("&", "and")
    s = s.replace("/", "_per_")
    s = re.sub(r"[^\w\s]", "", s)   # drop punctuation
    s = re.sub(r"\s+", "_", s)      # spaces -> underscore
    return s.lower()


# returns start and end year from a season string like "2023-2024"
def parse_season_years(season_val: str) -> tuple[int | None, int | None]:
    if pd.isna(season_val):
        return (None, None)
    m = re.match(r"^\s*(\d{4})\s*-\s*(\d{4})\s*$", str(season_val))
    if not m:
        return (None, None)
    return (int(m.group(1)), int(m.group(2)))


# converts the compact season encoding in the new source ("2324") to the
# "2023-2024" string the rest of the pipeline expects.
def season_int_to_str(s) -> str:
    if pd.isna(s):
        return None
    s = int(s)
    start = 2000 + s // 100
    end = 2000 + s % 100
    return f"{start}-{end}"


def main():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Source file not found: {SOURCE_FILE.resolve()}")

    df = pd.read_csv(SOURCE_FILE, low_memory=False)
    print(f"Loaded {SOURCE_FILE.name}: shape={df.shape}")

    # 1) drop the FIFA/EAFC block + junk helper col
    df = df.drop(columns=[c for c in FIFA_COLS if c in df.columns])

    # 2) filter to the 10 leagues we want (defensive; source already only has these 10)
    before = len(df)
    df = df[df["league"].isin(LEAGUES_KEEP)].copy()
    print(f"After league filter: {df.shape} (dropped {before - len(df)})")

    # 3) rename identity / playing-time columns to match the legacy schema
    df = df.rename(columns={
        "Playing Time_MP":     "Matches Played",
        "Playing Time_Starts": "Starts",
        "Playing Time_Min":    "minutes",
        "Playing Time_90s":    "nineties",
        "team":                "squad",
        "age_fbref":           "age",
        "born_fbref":          "born",
    })

    # 4) season "2324" -> "2023-2024" so parse_season_years still works
    df["season"] = df["season"].apply(season_int_to_str)

    # 5) comp column (human-readable league name) derived from the code column
    df["comp"] = df["league"].map(LEAGUE_CODE_TO_COMP)

    # 6) synthesise totals from per-90 * 90s, rounded (counts must be discrete)
    for per90_col, total_col in PER90_TO_TOTAL.items():
        if per90_col in df.columns:
            df[total_col] = (df[per90_col] * df["nineties"]).round()
        else:
            df[total_col] = pd.NA

    # 7) expose goals_p_90 / assists_p_90 under the names the pipeline expects
    df = df.rename(columns={
        "Per 90 Minutes_Gls": "goals_p_90",
        "Per 90 Minutes_Ast": "assists_p_90",
    })

    # 8) season start / end years for downstream joins
    season_years = df["season"].apply(parse_season_years)
    df["season_start_year"] = season_years.apply(lambda x: x[0])
    df["season_end_year"] = season_years.apply(lambda x: x[1])

    # 9) snake_case everything else so downstream code sees consistent names
    rename_map = {c: to_snake(c) for c in df.columns}
    df = df.rename(columns=rename_map)

    # 10) dedupe on (player, squad, comp, season) keeping the row with the
    #     most minutes -- same key as before but stronger "keep" rule.
    key_cols = [c for c in ["player", "squad", "comp", "season"] if c in df.columns]
    if key_cols:
        df = df.sort_values("minutes", ascending=False, na_position="last")
        df = df.drop_duplicates(subset=key_cols, keep="first")

    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}  shape={df.shape}")
    print("Sample columns:", list(df.columns)[:25])


if __name__ == "__main__":
    main()
