-- Where the main model is most wrong: residuals by age group.
SELECT
  CASE
    WHEN age_at_transfer < 21 THEN '<21'
    WHEN age_at_transfer < 24 THEN '21-23'
    WHEN age_at_transfer < 27 THEN '24-26'
    WHEN age_at_transfer < 30 THEN '27-29'
    WHEN age_at_transfer < 33 THEN '30-32'
    ELSE '33+'
  END AS age_group,
  COUNT(*) AS transfers,
  ROUND(AVG(actual_log_value - pred_log_main), 3) AS mean_residual_log,
  ROUND(MEDIAN(actual_log_value - pred_log_main), 3) AS median_residual_log,
  ROUND(AVG(ABS(actual_log_value - pred_log_main)), 3) AS mae_log
FROM fact_transfer
GROUP BY 1
ORDER BY MIN(age_at_transfer) ASC
