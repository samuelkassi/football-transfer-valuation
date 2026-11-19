-- Where the main model is most wrong: residuals by market value band.
SELECT
  CASE
    WHEN market_value_eur < 1e6 THEN '< EUR 1m'
    WHEN market_value_eur < 5e6 THEN 'EUR 1-5m'
    WHEN market_value_eur < 15e6 THEN 'EUR 5-15m'
    WHEN market_value_eur < 30e6 THEN 'EUR 15-30m'
    ELSE 'EUR 30m+'
  END AS value_band,
  COUNT(*) AS transfers,
  ROUND(AVG(actual_log_value - pred_log_main), 3) AS mean_residual_log,
  ROUND(MEDIAN(actual_log_value - pred_log_main), 3) AS median_residual_log,
  ROUND(AVG(ABS(actual_log_value - pred_log_main)), 3) AS mae_log
FROM fact_transfer
GROUP BY 1
ORDER BY MIN(market_value_eur) ASC
