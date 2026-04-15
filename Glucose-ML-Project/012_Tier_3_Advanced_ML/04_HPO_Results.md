# Tier 3 — Optuna HPO 결과 요약

> 튜닝 완료: 2026-04-15 16:17  | 총 소요: 4.6h

## 성능 비교표 (Validation / Test RMSE, mg/dL)

| Dataset         | Scale   | Model    |   n_trials_done |   Val_RMSE |   Test_RMSE |   Test_MAE |   Test_MAPE% |   actual_n_est |   elapsed_s | Best?   |
|:----------------|:--------|:---------|----------------:|-----------:|------------:|-----------:|-------------:|---------------:|------------:|:--------|
| AIDET1D         | small   | LightGBM |              12 |     22.07  |      22.42  |     15.733 |        11.76 |            264 |          30 | —       |
| AIDET1D         | small   | CatBoost |               9 |     22.036 |      22.354 |     15.68  |        11.72 |            804 |         102 | ⭐       |
| BIGIDEAs        | small   | LightGBM |              21 |     12.584 |      14.355 |      9.406 |         7.87 |             65 |          11 | ⭐       |
| BIGIDEAs        | small   | CatBoost |              17 |     12.522 |      14.456 |      9.445 |         7.9  |             78 |          41 | —       |
| Bris-T1D_Open   | medium  | LightGBM |               9 |     28.307 |      27.115 |     19.035 |        13.27 |            325 |          71 | —       |
| Bris-T1D_Open   | medium  | CatBoost |               7 |     28.273 |      27.08  |     19.041 |        13.28 |            576 |         195 | ⭐       |
| CGMacros_Dexcom | small   | LightGBM |              13 |      3.026 |       2.887 |      1.859 |         1.33 |            798 |         304 | ⭐       |
| CGMacros_Dexcom | small   | CatBoost |               4 |      3.547 |       3.415 |      2.013 |         1.4  |            674 |         326 | —       |
| CGMacros_Libre  | small   | LightGBM |              16 |      1.315 |       1.41  |      0.753 |         0.68 |            607 |         303 | ⭐       |
| CGMacros_Libre  | small   | CatBoost |               6 |      1.434 |       1.657 |      1.067 |         0.98 |            617 |         305 | —       |
| CGMND           | small   | LightGBM |              23 |     13.646 |      14.203 |      9.686 |         9.34 |             23 |          11 | —       |
| CGMND           | small   | CatBoost |              11 |     13.648 |      14.142 |      9.683 |         9.34 |            371 |          41 | ⭐       |
| GLAM            | large   | LightGBM |               5 |     13.534 |      13.524 |      9.742 |        10.04 |            997 |        1418 | ⭐       |
| GLAM            | large   | CatBoost |               4 |     13.552 |      13.547 |      9.76  |        10.06 |            998 |        2927 | —       |
| HUPA-UCM        | small   | LightGBM |               6 |     14.601 |      14.593 |      9.94  |         8.16 |            581 |         303 | —       |
| HUPA-UCM        | small   | CatBoost |               7 |     14.597 |      14.492 |      9.825 |         8.04 |            893 |         310 | ⭐       |
| IOBP2           | large   | LightGBM |               5 |     24.594 |      24.742 |     17.423 |        11.68 |           1000 |         858 | ⭐       |
| IOBP2           | large   | CatBoost |               2 |     24.617 |      24.764 |     17.449 |        11.69 |            999 |        2455 | —       |
| Park_2025       | small   | LightGBM |              24 |     22.272 |      19.906 |     15.194 |        16.48 |             29 |          30 | —       |
| Park_2025       | small   | CatBoost |              11 |     22.221 |      19.852 |     15.22  |        16.49 |            157 |          40 | ⭐       |
| PEDAP           | large   | LightGBM |               5 |     27.764 |      28.434 |     20.342 |        14.82 |           1000 |         438 | ⭐       |
| PEDAP           | large   | CatBoost |               5 |     27.823 |      28.49  |     20.393 |        14.85 |            999 |        1781 | —       |
| UCHTT1DM        | small   | LightGBM |              27 |     19.85  |      17.882 |     11.761 |        11.58 |             46 |           8 | —       |
| UCHTT1DM        | small   | CatBoost |              26 |     20.033 |      17.752 |     11.715 |        11.45 |            148 |          25 | ⭐       |
