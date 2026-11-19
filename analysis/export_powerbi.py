"""Export the star schema from transfers.duckdb as CSVs for Power BI.

Writes to analysis/powerbi/data/:

  fact_transfer.csv   one row per backtested transfer (2020 to 2025);
                      premium_log and above_fair_value are NULL unless the
                      deal was paid (fee > 0)
  dim_club.csv        club id, name and league proxy
  dim_league.csv      league names
  dim_season.csv      season end years
  dim_position.csv    position and sub position

Run with: python -m analysis.export_powerbi
"""
from __future__ import annotations

import duckdb

from pathlib import Path

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "transfers.duckdb"
OUT_DIR = HERE / "powerbi" / "data"

FACT_SQL = """
SELECT f.*, p.premium_log, p.above_fair_value
FROM fact_transfer f
LEFT JOIN fact_paid p USING (transfer_id)
ORDER BY f.transfer_id
"""

EXPORTS = [
    ("fact_transfer", FACT_SQL),
    ("dim_club", "SELECT club_id, club_name, league FROM dim_club ORDER BY club_id"),
    ("dim_league", "SELECT league FROM dim_league ORDER BY league"),
    ("dim_season", "SELECT season_end_year FROM dim_season ORDER BY season_end_year"),
    ("dim_position", "SELECT position, sub_position FROM dim_position ORDER BY position, sub_position"),
]


def export_powerbi(db_path: Path = DB_PATH) -> None:
    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path.name} not found; build it first with python -m analysis.build_db"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=True)
    for name, query in EXPORTS:
        out = OUT_DIR / f"{name}.csv"
        # POSIX separators keep the path out of SQL string escaping rules
        con.execute(f"COPY ({query}) TO '{out.as_posix()}' (HEADER, DELIMITER ',')")
        rows = len(out.read_text(encoding="utf-8").splitlines()) - 1
        print(f"{name}.csv ({rows} rows) -> {out}")
    con.close()


if __name__ == "__main__":
    export_powerbi()
