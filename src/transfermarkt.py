from pathlib import Path
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "transfermarkt"
OUT_DIR = PROJECT_ROOT / "data" / "intermediate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRANSFERS_IN = RAW_DIR / "transfers.csv"
VALS_IN = RAW_DIR / "player_valuations.csv"
PLAYERS_IN = RAW_DIR / "players.csv"

TRANSFERS_OUT = OUT_DIR / "tm_transfers.parquet"
VALS_OUT = OUT_DIR / "tm_player_valuations.parquet"
PLAYERS_OUT = OUT_DIR / "tm_players.parquet"


def parse_season_end_year(season: str) -> int | None:
    if pd.isna(season):
        return None
    s = str(season).strip()

    # 2017/2018 or 2017-2018
    m = re.match(r"^(\d{4})\s*[/\-]\s*(\d{4})$", s)
    if m:
        return int(m.group(2))

    # 17/18, 94/95, etc.
    m = re.match(r"^(\d{2})\s*[/\-]\s*(\d{2})$", s)
    if m:
        end_yy = int(m.group(2))
        # choose century
        if end_yy >= 50:
            return 1900 + end_yy
        return 2000 + end_yy

    # single year
    m = re.match(r"^(\d{4})$", s)
    if m:
        return int(m.group(1))

    return None


def main():
    # reading the transfers csv
    transfers = pd.read_csv(TRANSFERS_IN)
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
    transfers["season_end_year"] = transfers["transfer_season"].apply(parse_season_end_year)

    transfers["transfer_fee_eur"] = transfers["transfer_fee"]
    transfers["market_value_eur_at_transfer"] = transfers["market_value_in_eur"]

    transfers = transfers.rename(columns={"player_name": "player"})

    transfers_out = transfers[
        [
            "player_id",
            "player",
            "transfer_date",
            "transfer_season",
            "season_end_year",
            "from_club_id",
            "to_club_id",
            "from_club_name",
            "to_club_name",
            "transfer_fee_eur",
            "market_value_eur_at_transfer",
        ]
    ].copy()

    transfers_out.to_parquet(TRANSFERS_OUT, index=False)
    print(f"Saved: {TRANSFERS_OUT} shape={transfers_out.shape}")

    # reading the valuations csv

    vals = pd.read_csv(VALS_IN)
    vals["date"] = pd.to_datetime(vals["date"], errors="coerce")
    vals["market_value_eur"] = vals["market_value_in_eur"]

    vals_out = vals[
        [
            "player_id",
            "date",
            "market_value_eur",
            "current_club_id",
            "player_club_domestic_competition_id",
        ]
    ].copy()

    vals_out.to_parquet(VALS_OUT, index=False)
    print(f"Saved: {VALS_OUT} shape={vals_out.shape}")

    # read the players csv
    players = pd.read_csv(PLAYERS_IN)

    # parse dates which are used for calculations later
    players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
    players["contract_expiration_date"] = pd.to_datetime(players["contract_expiration_date"], errors="coerce")

    players_out = players[
        [
            "player_id",
            "name",
            "first_name",
            "last_name",
            "date_of_birth",
            "country_of_citizenship",
            "position",
            "sub_position",
            "foot",
            "height_in_cm",
            "current_club_id",
            "current_club_name",
            "current_club_domestic_competition_id",
            "contract_expiration_date",
            "agent_name",
            "market_value_in_eur",
            "highest_market_value_in_eur",
        ]
    ].copy()

    players_out.to_parquet(PLAYERS_OUT, index=False)
    print(f"Saved: {PLAYERS_OUT} shape={players_out.shape}")


if __name__ == "__main__":
    main()
