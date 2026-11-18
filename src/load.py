"""Read the raw Transfermarkt and FBref CSVs (see data/README.md)."""
from __future__ import annotations

import re

import pandas as pd

from .config import INTER_DIR, RAW_FBREF_DIR, RAW_TM_DIR, TM_PLAYERS, TM_TRANSFERS, TM_VALUATIONS

FBREF_SOURCE = RAW_FBREF_DIR / "fifa_fbref_merged.csv"


def parse_season_end_year(season: str) -> int | None:
    """'2017/2018', '2017-2018', '17/18' or '2018' -> 2018."""
    if pd.isna(season):
        return None
    s = str(season).strip()
    m = re.match(r"^(\d{4})\s*[/\-]\s*(\d{4})$", s)
    if m:
        return int(m.group(2))
    m = re.match(r"^(\d{2})\s*[/\-]\s*(\d{2})$", s)
    if m:
        end_yy = int(m.group(2))
        return 1900 + end_yy if end_yy >= 50 else 2000 + end_yy
    m = re.match(r"^(\d{4})$", s)
    if m:
        return int(m.group(1))
    return None


def load_transfermarkt() -> None:
    """Raw Transfermarkt CSVs -> tm_transfers / tm_player_valuations / tm_players parquets."""
    INTER_DIR.mkdir(parents=True, exist_ok=True)

    transfers = pd.read_csv(RAW_TM_DIR / "transfers.csv")
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
    transfers["season_end_year"] = transfers["transfer_season"].apply(parse_season_end_year)
    transfers["transfer_fee_eur"] = transfers["transfer_fee"]
    transfers["market_value_eur_at_transfer"] = transfers["market_value_in_eur"]
    transfers = transfers.rename(columns={"player_name": "player"})
    transfers = transfers[[
        "player_id", "player", "transfer_date", "transfer_season", "season_end_year",
        "from_club_id", "to_club_id", "from_club_name", "to_club_name",
        "transfer_fee_eur", "market_value_eur_at_transfer",
    ]].copy()
    transfers.to_parquet(TM_TRANSFERS, index=False)
    print(f"Saved: {TM_TRANSFERS.name} shape={transfers.shape}")

    vals = pd.read_csv(RAW_TM_DIR / "player_valuations.csv")
    vals["date"] = pd.to_datetime(vals["date"], errors="coerce")
    vals["market_value_eur"] = vals["market_value_in_eur"]
    vals = vals[[
        "player_id", "date", "market_value_eur", "current_club_id",
        "player_club_domestic_competition_id",
    ]].copy()
    vals.to_parquet(TM_VALUATIONS, index=False)
    print(f"Saved: {TM_VALUATIONS.name} shape={vals.shape}")

    players = pd.read_csv(RAW_TM_DIR / "players.csv")
    players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
    players["contract_expiration_date"] = pd.to_datetime(players["contract_expiration_date"], errors="coerce")
    players = players[[
        "player_id", "name", "first_name", "last_name", "date_of_birth",
        "country_of_citizenship", "position", "sub_position", "foot", "height_in_cm",
        "current_club_id", "current_club_name", "current_club_domestic_competition_id",
        "contract_expiration_date", "agent_name", "market_value_in_eur",
        "highest_market_value_in_eur",
    ]].copy()
    players.to_parquet(TM_PLAYERS, index=False)
    print(f"Saved: {TM_PLAYERS.name} shape={players.shape}")


def read_fbref_raw() -> pd.DataFrame:
    if not FBREF_SOURCE.exists():
        raise FileNotFoundError(f"Source file not found: {FBREF_SOURCE}. See data/README.md.")
    df = pd.read_csv(FBREF_SOURCE, low_memory=False)
    print(f"Loaded {FBREF_SOURCE.name}: shape={df.shape}")
    return df
