-- Buying leagues that pay most over the odds. Minimum 10 paid deals.
-- The buying club's league is proxied by its most common selling league.
SELECT
  c.league AS buying_league,
  COUNT(*) AS paid_deals,
  COUNT(DISTINCT f.to_club_id) AS buying_clubs,
  ROUND(MEDIAN(f.premium_log), 3) AS median_premium_log,
  ROUND(AVG(f.above_fair_value::INT), 3) AS share_above_fair_value
FROM fact_paid f
JOIN dim_club c ON c.club_id = f.to_club_id
WHERE c.league IS NOT NULL
GROUP BY 1
HAVING COUNT(*) >= 10
ORDER BY median_premium_log DESC
