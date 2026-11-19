-- Trend of the overpayment premium by season (paid deals only).
-- premium_log = ln(fee) - buyer-agnostic predicted log value.
-- The last column compares fee with the Transfermarkt market value itself.
SELECT
  season_end_year AS season,
  COUNT(*) AS paid_deals,
  ROUND(MEDIAN(premium_log), 3) AS median_premium_log,
  ROUND(QUANTILE_CONT(premium_log, 0.25), 3) AS q25_premium_log,
  ROUND(QUANTILE_CONT(premium_log, 0.75), 3) AS q75_premium_log,
  ROUND(AVG(above_fair_value::INT), 3) AS share_above_fair_value,
  ROUND(MEDIAN(ln(transfer_fee_eur) - ln(market_value_eur)), 3) AS median_log_fee_minus_log_value
FROM fact_paid
GROUP BY 1
ORDER BY 1
