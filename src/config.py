"""Paths and run-wide constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_TM_DIR = DATA_DIR / "raw" / "transfermarkt"
RAW_FBREF_DIR = DATA_DIR / "raw" / "fbref"
INTER_DIR = DATA_DIR / "intermediate"
FINAL_DIR = DATA_DIR / "final"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

TM_TRANSFERS = INTER_DIR / "tm_transfers.parquet"
TM_VALUATIONS = INTER_DIR / "tm_player_valuations.parquet"
TM_PLAYERS = INTER_DIR / "tm_players.parquet"
FBREF_STATS = INTER_DIR / "fbref_player_season_stats.parquet"
DATASET = FINAL_DIR / "player_market_value_dataset.parquet"
BUILD_STATS = FINAL_DIR / "build_stats.json"

SEED = 42
TEST_SEASON = 2025                       # last season in the data: held out
SEASON_END_YEARS = list(range(2018, 2026))  # transfer seasons 2017-18 to 2024-25
TARGET_COL = "log_market_value"          # log1p(market value at transfer)
SEASON_COL = "season_end_year"
