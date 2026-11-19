"""Build analysis/transfers.duckdb from the backtest predictions (Phase 3).

Star schema: fact_transfer (one row per backtested transfer) plus dim_club,
dim_league, dim_season and dim_position. A view `fact_paid` restricts to paid
deals (fee > 0) and adds the premium:

    premium_log = ln(fee) - pred_log_buyer_agnostic

so a positive premium means the buyer paid more than the buyer-agnostic
valuation says the player is worth. `above_fair_value` is fee above the
buyer-agnostic prediction in euros.

Club league proxy: there is no clubs table in the sources, so a club's league
is its most common selling league across all backtest rows. For buying clubs
this is a proxy (a club usually buys from the league it plays in), and it uses
all seasons, not point-in-time. `league` on a transfer row is the league the
player played in the previous season, used as the selling-league proxy.

Rebuild with: python -m analysis.build_db
"""
from __future__ import annotations

import duckdb
import pandas as pd

from pathlib import Path

from src.config import PROJECT_ROOT

DB_PATH = Path(__file__).resolve().parent / "transfers.duckdb"
PREDICTIONS = PROJECT_ROOT / "data" / "final" / "backtest_predictions.parquet"

DDL = """
CREATE OR REPLACE TABLE dim_season AS
SELECT DISTINCT season_end_year FROM fact_transfer ORDER BY season_end_year;

CREATE OR REPLACE TABLE dim_position AS
SELECT DISTINCT position, sub_position FROM fact_transfer
ORDER BY position, sub_position;

CREATE OR REPLACE TABLE dim_league AS
SELECT DISTINCT league FROM fact_transfer WHERE league IS NOT NULL ORDER BY league;

CREATE OR REPLACE TABLE dim_club AS
WITH clubs AS (
  SELECT club_id, MAX(club_name) AS club_name FROM (
    SELECT from_club_id AS club_id, from_club_name AS club_name FROM fact_transfer
    UNION ALL
    SELECT to_club_id AS club_id, to_club_name AS club_name FROM fact_transfer
  ) GROUP BY club_id
),
sell_league AS (
  SELECT from_club_id AS club_id, league, COUNT(*) AS n
  FROM fact_transfer
  WHERE league IS NOT NULL
  GROUP BY 1, 2
  QUALIFY ROW_NUMBER() OVER (PARTITION BY club_id ORDER BY n DESC) = 1
)
SELECT c.club_id, c.club_name, s.league
FROM clubs c
LEFT JOIN sell_league s ON c.club_id = s.club_id;

CREATE OR REPLACE VIEW fact_paid AS
SELECT *,
       ln(transfer_fee_eur) - pred_log_buyer_agnostic AS premium_log,
       (transfer_fee_eur > pred_eur_buyer_agnostic) AS above_fair_value
FROM fact_transfer
WHERE transfer_fee_eur > 0;
"""


def build_db(predictions: Path = PREDICTIONS, db_path: Path = DB_PATH) -> None:
    fact = pd.read_parquet(predictions).reset_index().rename(columns={"index": "transfer_id"})

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("DROP TABLE IF EXISTS fact_transfer")
    con.execute("DROP VIEW IF EXISTS fact_paid")
    con.execute("CREATE TABLE fact_transfer AS SELECT * FROM fact")
    con.execute(DDL)

    n = con.execute("SELECT COUNT(*) FROM fact_transfer").fetchone()[0]
    paid = con.execute("SELECT COUNT(*) FROM fact_paid").fetchone()[0]
    print(f"fact_transfer rows: {n:,}")
    print(f"paid deals (fee > 0): {paid:,}")
    print(f"clubs: {con.execute('SELECT COUNT(*) FROM dim_club').fetchone()[0]:,}")
    print(f"Saved {db_path}")
    con.close()


if __name__ == "__main__":
    build_db()
