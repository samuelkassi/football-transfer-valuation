-- Selling leagues that sell cheapest relative to value. The selling league
-- is the league the player played in the season before the transfer.
SELECT
  f.league AS selling_league,
  COUNT(*) AS paid_deals,
  ROUND(MEDIAN(f.premium_log), 3) AS median_premium_log,
  ROUND(AVG(f.above_fair_value::INT), 3) AS share_above_fair_value,
  ROUND(MEDIAN(f.transfer_fee_eur) / 1e6, 2) AS median_fee_eur_m
FROM fact_paid f
WHERE f.league IS NOT NULL
GROUP BY 1
ORDER BY median_premium_log ASC
