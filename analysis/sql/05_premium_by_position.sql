-- Which positions are cheapest relative to the buyer-agnostic valuation.
SELECT
  position,
  COUNT(*) AS paid_deals,
  ROUND(MEDIAN(premium_log), 3) AS median_premium_log,
  ROUND(AVG(above_fair_value::INT), 3) AS share_above_fair_value
FROM fact_paid
WHERE position IS NOT NULL
GROUP BY 1
ORDER BY median_premium_log ASC
