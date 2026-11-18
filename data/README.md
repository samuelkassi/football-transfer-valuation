# Data

No data is committed. Both sources are free Kaggle datasets, but their
contents are scraped from Transfermarkt and FBref (whose advanced stats are
Opta data), so this repo ships download instructions only.

## Sources

| Source | Kaggle dataset | Files used |
|---|---|---|
| Transfermarkt | [davidcariboo/player-scores](https://www.kaggle.com/datasets/davidcariboo/player-scores) (built by [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets)) | `transfers.csv`, `player_valuations.csv`, `players.csv` |
| FBref + FIFA | [jacksonjohannessen/fifa-and-irl-soccer-player-data](https://www.kaggle.com/datasets/jacksonjohannessen/fifa-and-irl-soccer-player-data) | `fifa_fbref_merged.csv` |

Downloaded April 2026. Check each dataset's licence on its Kaggle page before use.

## Layout

```
data/
  raw/
    transfermarkt/
      transfers.csv
      player_valuations.csv
      players.csv
    fbref/
      fifa_fbref_merged.csv
  intermediate/   written by the pipeline
  final/          written by the pipeline
```

## Snapshot used for the results

`player-scores` is refreshed weekly, so a fresh download will contain more
recent transfers and valuations and will not reproduce the published numbers
exactly. Use the Kaggle dataset version from around 25 April 2026 if it is
still available. The files behind `results/` have these SHA-256 hashes:

| File | SHA-256 |
|---|---|
| `fbref/fifa_fbref_merged.csv` | `096e13b13da401d054a69c4f51563b52d7f54de717cb7e16d864b33e34d52027` |
| `transfermarkt/player_valuations.csv` | `ec91ac4a58e96d6aec0b2a0d2deb67cd08a7ce340ff9f88c24043d643fd350f7` |
| `transfermarkt/players.csv` | `6973df9ce9e6362dbce55fb6b3ac70638d1f41144fcfe9bdd182ffe18e88406b` |
| `transfermarkt/transfers.csv` | `19a263cfa72c143410b62487b14e280d5bd09f538653ece52dd2ab502511d87a` |

Check yours with `sha256sum data/raw/*/*.csv` (or `Get-FileHash` on Windows).

## Known quirks of the sources

- The FBref file keeps one row per (name, season) across all ten leagues, so
  different players who share a short name are summed into one row. The
  pipeline drops rows with more matches or minutes than a team can play and
  joins on name, season and birth year (see `src/clean.py`, `src/merge.py`).
- `current_club_id` in `player_valuations.csv` is the player's club when the
  data was scraped, not on the valuation date. Club levels are dated from the
  transfer history instead (see `src/features.py`).
- `players.csv` holds one contract snapshot per player, so no point-in-time
  contract length is available and none is used.
