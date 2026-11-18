# What each fix did

Test set: 2025 transfers. Train: all earlier seasons. Target: log1p(market value).
Every row is a full rerun of the pipeline from raw data after that fix.

## Test R squared

| Step | Rows (train / test) | Ridge trad | Ridge modern | RF trad | RF modern | XGB trad | XGB modern | Global mean | Position mean | League mean |
|---|---|---|---|---|---|---|---|---|---|---|
| Paper (submitted) | 5,575 / 1,462 | 0.581 | 0.601 | 0.651 | 0.659 | 0.688 | 0.719 | -0.002 | 0.003 | 0.285 |
| 0. Original code, rebuilt from raw | 5,575 / 1,462 | 0.573 | 0.587 | 0.651 | 0.660 | 0.687 | 0.718 | | | |
| 1. Add trivial baselines | 5,575 / 1,462 | 0.573 | 0.587 | 0.651 | 0.660 | 0.687 | 0.718 | -0.002 | 0.003 | 0.285 |
| 2. Name + birth year join, drop impossible rows | 5,293 / 1,413 | 0.641 | 0.671 | 0.665 | 0.672 | 0.712 | 0.737 | -0.004 | 0.005 | 0.290 |
| 3. Position flags mapped to Transfermarkt labels | 5,293 / 1,413 | 0.646 | 0.678 | 0.663 | 0.672 | 0.710 | 0.736 | -0.004 | 0.005 | 0.290 |
| 4. Contract feature removed | 5,293 / 1,413 | 0.650 | 0.681 | 0.662 | 0.674 | 0.715 | 0.742 | -0.004 | 0.005 | 0.290 |

Contract ablation: step 3 is the run with the leaky contract feature
(years until the next transfer, snapshot fallback for 2023+ transfers);
step 4 is the same pipeline without it. Removing it did not lower test
R squared (XGBoost modern 0.736 to 0.742). The leak did not inflate the
2025 score because most 2025 transfers have no later transfer yet, so their
values came from the snapshot fallback while training rows used the future
transfer date: the feature meant different things in train and test.

## Match rate

Denominator: all 54,698 Transfermarkt transfers in the 2018 to 2025 seasons.

| Step | Name + season match | Verified match (birth year, unambiguous) | Model rows |
|---|---|---|---|
| 0. Original | 8,188 (14.97%) | not checked | 7,037 (12.87%) |
| 2. After fix | 8,039 (14.70%) | 7,741 (14.15%) | 6,706 (12.26%) |

The paper's 14.97% is the name + season match rate before the 450 minute filter;
7,037 is what remains after that filter and the target and age checks.
