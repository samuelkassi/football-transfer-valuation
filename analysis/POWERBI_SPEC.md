# Power BI report spec: who overpays in the transfer market

Data: `analysis/powerbi/data/` (exported from `transfers.duckdb` by
`python -m analysis.export_powerbi`). Load the five CSVs into Power BI Desktop
and relate them: `fact_transfer.to_club_id` to `dim_club.club_id`, and the
`league`, `season_end_year`, `position` columns to their dim tables. The fact
table has 6,106 transfers from 2020 to 2025; `premium_log` and
`above_fair_value` are blank for unpaid deals (free transfers and loans).

## What the numbers mean

- `premium_log` = ln(fee) minus the buyer-agnostic predicted log value. A fee
  above the prediction means the buyer paid more than the model says the
  player is worth, regardless of who buys. A premium of +0.69 is roughly a
  fee twice the predicted value.
- `above_fair_value` = fee above the buyer-agnostic prediction in euros.
- The buyer-agnostic predictions come from the expanding-window backtest
  (out of sample), see `src/backtest.py` and `results/backtest_r2.md`.

## Measures (DAX)

```
Paid Deals          = COUNTROWS(FILTER(fact_transfer, NOT ISBLANK(fact_transfer[premium_log])))
Median Premium Log  = MEDIAN(fact_transfer[premium_log])
Premium %           = EXP([Median Premium Log]) - 1
Share Above Fair    = DIVIDE(
                        CALCULATE(COUNT(fact_transfer[transfer_id]), fact_transfer[above_fair_value] = TRUE()),
                        [Paid Deals])
Total Fees EUR m    = DIVIDE(SUM(fact_transfer[transfer_fee_eur]), 1000000)
Median Fee EUR m    = DIVIDE(MEDIAN(fact_transfer[transfer_fee_eur]), 1000000)
Mean Residual Log   = AVERAGEX(fact_transfer, fact_transfer[actual_log_value] - fact_transfer[pred_log_main])
Median Residual Log = MEDIANX(fact_transfer, fact_transfer[actual_log_value] - fact_transfer[pred_log_main])
```

Format `Premium %` as percentage with one decimal, `Median Premium Log` and the
residuals as three decimals.

## Pages

**1. Overview.** KPI cards: Paid Deals (1,813), Median Premium % (about +5%),
Share Above Fair (about 52%), Total Fees EUR m. Line chart: median premium by
`season_end_year` (use 01_premium_by_season.csv if the quartile bands matter).
Histogram: `premium_log` on paid deals, bins of 0.25 from -2 to 2.

**2. Buyers.** Bar chart: top 20 buying clubs by Median Premium Log, with the
visual filter Paid Deals >= 10 (this is the club-level reliability threshold,
82 clubs qualify). Table: club, league, Paid Deals, Median Premium Log,
Premium %, Share Above Fair, Median Fee EUR m. Second bar chart: median
premium by buying league (`dim_club.league`). Caption note: Man Utd are above
the valuation on 18 of 18 paid deals.

**3. Sellers.** Bar chart: median premium by selling league (`league` on the
fact table), sorted ascending. Table with Paid Deals, Share Above Fair and
Median Fee EUR m per league. Caption: MLS and Super Lig sell cheapest
relative to value; Premier League sells dearest.

**4. Players.** Bar charts: median premium by position and by age group
(copy the banding from 06_premium_by_age_group.csv into a calculated column:
<21, 21-23, 24-26, 27-29, 30-32, 33+). Scatter: `age_at_transfer` against
`premium_log` on paid deals. Caption: the youth premium falls monotonically
with age, from about +104% under 21 to about -38% at 33+.

**5. Model check.** Bar charts of Mean Residual Log by league and by market
value band (use 10_residual_by_value_band.csv for the bands). Table of
residuals by position and age group. Caption (important): the model shrinks
predictions toward the mean, underpredicting players over EUR 30m by about
0.51 log and overpredicting players under EUR 1m by about 0.45 log, so
premiums at the extremes of the value range are inflated in that direction.
Read club and league rankings within value bands, not only in the aggregate.

## Filters on every page

Slicers: `season_end_year` (2020 to 2025), `league` (the selling-league
proxy), `position`. A "paid only" filter lives in the measures, not in the
page, so the Model check page still sees all 6,106 rows.

## Caveats to carry in the report footer

- Club league is a proxy: a club's most common selling league across all
  backtest rows. The sources have no clubs table.
- The selling league is the league the player played in the season before the
  transfer.
- Fees are only recorded for 6,187 of the 6,706 model rows; add-ons and loan
  fees are not separated in the source.
- Club rankings need 10+ paid deals to be meaningful.
- Only XGBoost valuations are used, and predictions are out of sample.

## What to produce

Build the .pbix from the CSVs (the .pbix itself stays local, it is
gitignored), then export a PDF and one screenshot per page into
`analysis/powerbi/`. Suggested PDF name: `overpayment_report.pdf`.
