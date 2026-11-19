-- Where the main model is most wrong: residuals by position.
SELECT
  position,
  COUNT(*) AS transfers,
  ROUND(AVG(actual_log_value - pred_log_main), 3) AS mean_residual_log,
  ROUND(MEDIAN(actual_log_value - pred_log_main), 3) AS median_residual_log,
  ROUND(AVG(ABS(actual_log_value - pred_log_main)), 3) AS mae_log
FROM fact_transfer
WHERE position IS NOT NULL
GROUP BY 1
ORDER BY mean_residual_log ASC
