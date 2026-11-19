-- Which age groups are cheapest relative to the buyer-agnostic valuation.
SELECT
  CASE
    WHEN age_at_transfer < 21 THEN '<21'
    WHEN age_at_transfer < 24 THEN '21-23'
    WHEN age_at_transfer < 27 THEN '24-26'
    WHEN age_at_transfer < 30 THEN '27-29'
    WHEN age_at_transfer < 33 THEN '30-32'
    ELSE '33+'
  END AS age_group,
  COUNT(*) AS paid_deals,
  ROUND(MEDIAN(premium_log), 3) AS median_premium_log,
  ROUND(AVG(above_fair_value::INT), 3) AS share_above_fair_value
FROM fact_paid
GROUP BY 1
ORDER BY MIN(age_at_transfer) ASC
