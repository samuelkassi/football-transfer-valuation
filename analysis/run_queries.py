"""Run every SQL file in analysis/sql/ against transfers.duckdb.

Each file is one question and writes one CSV with the same stem to
analysis/output/. Rebuild the database first with python -m analysis.build_db.

Run with: python -m analysis.run_queries
"""
from __future__ import annotations

import duckdb

from pathlib import Path

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "transfers.duckdb"
SQL_DIR = HERE / "sql"
OUT_DIR = HERE / "output"


def run_queries(db_path: Path = DB_PATH) -> list[Path]:
    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path.name} not found; build it first with python -m analysis.build_db"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=True)
    written = []
    for sql_file in sorted(SQL_DIR.glob("*.sql")):
        query = sql_file.read_text(encoding="utf-8")
        out = OUT_DIR / f"{sql_file.stem}.csv"
        # POSIX separators keep the path out of SQL string escaping rules
        con.execute(f"COPY ({query}) TO '{out.as_posix()}' (HEADER, DELIMITER ',')")
        rows = len(out.read_text(encoding="utf-8").splitlines()) - 1
        print(f"{sql_file.name} -> {out.name} ({rows} rows)")
        written.append(out)
    con.close()
    return written


if __name__ == "__main__":
    run_queries()
