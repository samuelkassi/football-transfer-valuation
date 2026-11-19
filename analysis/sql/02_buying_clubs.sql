-- Buying clubs that pay most over the buyer-agnostic valuation.
-- Minimum 10 paid deals. The club league is a proxy: the club's most common
-- selling league across all backtest rows (see analysis/build_db.py).
SELECT
  c.club_name AS buying_club,
  c.league AS club_league_proxy,
  COUNT(*) AS paid_deals,
  ROUND(MEDIAN(f.premium_log), 3) AS median_premium_log,
  ROUND(AVG(f.above_fair_value::INT), 3) AS share_above_fair_value,
  ROUND(MEDIAN(f.transfer_fee_eur) / 1e6, 2) AS median_fee_eur_m
FROM fact_paid f
JOIN dim_club c ON c.club_id = f.to_club_id
GROUP BY 1, 2
HAVING COUNT(*) >= 10
ORDER BY median_premium_log DESC
