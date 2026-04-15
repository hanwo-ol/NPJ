# Tier 3 Baseline Boosting Results

XGBoost / LightGBM / CatBoost 디폴트 하이퍼파라미터 기저 성능 비교

## 1. 성능 비교표 (RMSE mg/dL)

| Dataset         |   Windows |   Dim |   XGB_RMSE |   LGBM_RMSE |   Cat_RMSE | Best     |   Best_RMSE |   Best_MAE |   Best_MAPE% |
|:----------------|----------:|------:|-----------:|------------:|-----------:|:---------|------------:|-----------:|-------------:|
| AIDET1D         |    471528 |    20 |      22.54 |       22.46 |      22.36 | CatBoost |       22.36 |      15.7  |         11.8 |
| BIGIDEAs        |     36409 |    92 |      14.4  |       14.28 |      14.03 | CatBoost |       14.03 |       9.07 |          7.6 |
| Bris-T1D_Open   |    818600 |   236 |      27.49 |       27.42 |      27.34 | CatBoost |       27.34 |      19.28 |         13.4 |
| CGMacros_Dexcom |    415299 |   398 |       3.22 |        3.08 |       3.22 | LightGBM |        3.08 |       1.97 |          1.4 |
| CGMacros_Libre  |    455705 |   398 |       1.63 |        1.4  |       1.34 | CatBoost |        1.34 |       0.81 |          0.8 |
| CGMND           |    114409 |    20 |      14.29 |       14.13 |      13.92 | CatBoost |       13.92 |       9.49 |          9.2 |
| GLAM            |  26165917 |    38 |      13.51 |       13.5  |      13.51 | LightGBM |       13.5  |       9.72 |         10   |
| HUPA-UCM        |    309092 |   200 |      14.72 |       14.63 |      14.42 | CatBoost |       14.42 |       9.77 |          7.9 |
| IOBP2           |  14027905 |    74 |      24.79 |       24.78 |      24.81 | LightGBM |       24.78 |      17.45 |         11.7 |
| Park_2025       |     23064 |    38 |      21.13 |       21.02 |      20.18 | CatBoost |       20.18 |      15.43 |         16.5 |
| PEDAP           |   7055811 |    56 |      28.34 |       28.33 |      28.38 | LightGBM |       28.33 |      20.28 |         14.8 |
| UCHTT1DM        |     27220 |    38 |      19.11 |       19    |      17.78 | CatBoost |       17.78 |      11.98 |         11.8 |

## 2. Best Model Feature Importance & Training Time

| Dataset         | Best_Top5                                                                                                                            | XGB_Time   | LGBM_Time   | Cat_Time   |
|:----------------|:-------------------------------------------------------------------------------------------------------------------------------------|:-----------|:------------|:-----------|
| AIDET1D         | glucose_value_mg_dl_t-0 (60.7%), Velocity (18.8%), glucose_value_mg_dl_t-2 (3.8%), glucose_value_mg_dl_t-1 (3.2%), Window_Std (2.2%) | 2s         | 2s          | 5s         |
| BIGIDEAs        | glucose_value_mg_dl_t-0 (37.0%), Velocity (11.2%), Acceleration (5.8%), tod_cos (4.4%), tod_sin (3.8%)                               | 2s         | 1s          | 3s         |
| Bris-T1D_Open   | glucose_value_mg_dl_t-0 (55.5%), Velocity (22.2%), glucose_value_mg_dl_t-1 (4.1%), SD1 (3.3%), tod_sin (2.3%)                        | 14s        | 4s          | 52s        |
| CGMacros_Dexcom | Velocity (10.7%), glucose_value_mg_dl_t-0 (4.6%), Libre GL_t-0 (3.9%), SD1 (2.9%), Window_Std (2.7%)                                 | 131s       | 27s         | 24s        |
| CGMacros_Libre  | glucose_value_mg_dl_t-0 (20.0%), TIR (14.8%), glucose_value_mg_dl_t-1 (12.5%), glucose_value_mg_dl_t-2 (9.3%), Window_AUC (7.6%)     | 32s        | 11s         | 24s        |
| CGMND           | glucose_value_mg_dl_t-0 (28.7%), tod_cos (11.8%), Velocity (8.6%), glucose_value_mg_dl_t-5 (8.5%), tod_sin (6.4%)                    | 1s         | 1s          | 2s         |
| GLAM            | Velocity (10.0%), glucose_value_mg_dl_t-0 (7.5%), Jerk (7.5%), SD1 (7.4%), glucose_value_mg_dl_t-5 (7.2%)                            | 413s       | 179s        | 298s       |
| HUPA-UCM        | glucose_value_mg_dl_t-0 (46.4%), Velocity (20.9%), glucose_value_mg_dl_t-1 (7.6%), TIR (2.6%), glucose_value_mg_dl_t-2 (1.8%)        | 9s         | 5s          | 13s        |
| IOBP2           | Velocity (12.2%), Jerk (10.0%), Acceleration (9.9%), SD1 (9.1%), Window_Std (7.3%)                                                   | 137s       | 60s         | 230s       |
| Park_2025       | Window_AUC (24.5%), Window_Mean (20.2%), HBGI (9.9%), TIR (7.0%), TAR (5.7%)                                                         | 3s         | 3s          | 7s         |
| PEDAP           | Velocity (9.7%), tod_cos (6.5%), tod_sin (6.4%), SD1 (6.1%), glucose_value_mg_dl_t-0 (5.9%)                                          | 109s       | 47s         | 82s        |
| UCHTT1DM        | glucose_value_mg_dl_t-0 (41.1%), Velocity (17.1%), glucose_value_mg_dl_t-1 (6.6%), Window_Mean (4.5%), Window_Std (2.9%)             | 1s         | 0s          | 2s         |
