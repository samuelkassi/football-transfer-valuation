# What each fix did

Test set: 2025 transfers. Train: all earlier seasons. Target: log1p(market value).
Every row is a full rerun of the pipeline from raw data after that fix.

## Test R squared

| Step | Rows (train / test) | Ridge trad | Ridge modern | RF trad | RF modern | XGB trad | XGB modern | Global mean | Position mean | League mean |
|---|---|---|---|---|---|---|---|---|---|---|
| Paper (submitted) | 5,575 / 1,462 | 0.581 | 0.601 | 0.651 | 0.659 | 0.688 | 0.719 | -0.002 | 0.003 | 0.285 |
| 0. Original code, rebuilt from raw | 5,575 / 1,462 | 0.573 | 0.587 | 0.651 | 0.660 | 0.687 | 0.718 | | | |
| 1. Add trivial baselines | 5,575 / 1,462 | 0.573 | 0.587 | 0.651 | 0.660 | 0.687 | 0.718 | -0.002 | 0.003 | 0.285 |
