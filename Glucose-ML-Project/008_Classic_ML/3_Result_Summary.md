# 실험 결과 요약 (Tier 2 ML Baseline)

Decision Tree 및 Random Forest 벤치마크 결과표 (SVM 데이터 무결성 보존을 위해 제거됨)

| Dataset         |   Windows |   Covariates |   DT_RMSE |   DT_MAPE% |   RF_RMSE |   RF_MAE |   RF_MAPE% | Top_3_Important_Features                                                   |
|:----------------|----------:|-------------:|----------:|-----------:|----------:|---------:|-----------:|:---------------------------------------------------------------------------|
| AIDET1D         |    471528 |            0 |     25.24 |       13.8 |     22.98 |    16.16 |       12.1 | glucose_value_mg_dl (100.0%)                                               |
| BIGIDEAs        |     36409 |            2 |     16.08 |        8.9 |     14.72 |     9.64 |        8   | glucose_value_mg_dl (92.6%), calorie (4.8%), protein (2.6%)                |
| Bris-T1D_Open   |    818600 |            6 |     30.44 |       15.3 |     28    |    19.71 |       13.7 | glucose_value_mg_dl (100.0%), insulin (0.0%), steps (0.0%)                 |
| CGMacros_Dexcom |    415299 |           15 |      5.8  |        2.8 |      3.41 |     2.19 |        1.5 | glucose_value_mg_dl (99.7%), Libre GL (0.1%), Calories (Activity) (0.1%)   |
| CGMacros_Libre  |    455705 |           15 |      4    |        2.6 |      1.62 |     0.96 |        1   | glucose_value_mg_dl (100.0%), Calories (Activity) (0.0%), Dexcom GL (0.0%) |
| CGMND           |    114409 |            0 |     14.61 |        9.7 |     14.43 |     9.7  |        9.4 | glucose_value_mg_dl (100.0%)                                               |
| GLAM            |  26165917 |            1 |     14.13 |       10.5 |     13.69 |     9.86 |       10.2 | glucose_value_mg_dl (99.9%), EventMarker (0.1%)                            |
| HUPA-UCM        |    309092 |            6 |     19.12 |       11.2 |     15.48 |    10.57 |        8.6 | glucose_value_mg_dl (97.2%), heart_rate (1.0%), calories (0.8%)            |
| IOBP2           |  14027905 |            1 |     28.59 |       14.1 |     25.15 |    17.75 |       11.9 | glucose_value_mg_dl (100.0%), MealDoseAnnounceAmt (0.0%)                   |
| Park_2025       |     23064 |            1 |     21.6  |       18   |     20.47 |    15.52 |       16.5 | glucose_value_mg_dl (95.3%), rep (4.7%)                                    |
| PEDAP           |   7055811 |            2 |     32.36 |       17.4 |     28.98 |    20.78 |       15.2 | glucose_value_mg_dl (99.3%), BasalRate (0.7%), BolusType (0.0%)            |
| UCHTT1DM        |     27220 |            1 |     18.83 |       13   |     18.18 |    12.17 |       12   | glucose_value_mg_dl (99.0%), Value (g) (1.0%)                              |