# Backtest check: expanding-window R squared

XGBoost with the tuned parameters of the main run, refit per season on earlier seasons only, scored on that season.

| Season | n | R squared, main | R squared, buyer-agnostic |
|---|---|---|---|
| 2020 | 657 | 0.760 | 0.623 |
| 2021 | 663 | 0.765 | 0.693 |
| 2022 | 847 | 0.739 | 0.682 |
| 2023 | 1202 | 0.751 | 0.708 |
| 2024 | 1324 | 0.791 | 0.718 |
| 2025 | 1413 | 0.816 | 0.763 |
