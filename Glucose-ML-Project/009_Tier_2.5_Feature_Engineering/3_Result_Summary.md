# 실험 결과 요약 (Tier 2.5 Feature Engineering)

고난도 임상 특징 파생변수 투입에 따른 Random Forest(max_depth=20) 벤치마크 결과

| Dataset         |   Windows |   Feature_Dim |   DT_RMSE |   RF_RMSE |   RF_MAE |   RF_MAPE% | Top_5_Important_Features                                                                                                 |
|:----------------|----------:|--------------:|----------:|----------:|---------:|-----------:|:-------------------------------------------------------------------------------------------------------------------------|
| AIDET1D         |    471528 |            18 |     24.4  |     22.98 |    16.15 |       12.1 | glucose_value_mg_dl_t-0 (88.2%), Velocity (3.5%), Window_Std (1.1%), glucose_value_mg_dl_t-5 (1.0%), Acceleration (0.9%) |
| BIGIDEAs        |     36409 |            54 |     15.45 |     14.21 |     9.44 |        7.9 | glucose_value_mg_dl_t-0 (58.5%), Velocity (5.7%), Acceleration (4.5%), Window_Std (2.7%), cos_hour (2.7%)                |
| Bris-T1D_Open   |    818600 |           126 |     29.23 |     27.9  |    19.63 |       13.7 | glucose_value_mg_dl_t-0 (79.2%), Velocity (6.5%), Window_Std (2.0%), glucose_value_mg_dl_t-5 (1.5%), Acceleration (1.5%) |
| CGMacros_Dexcom |    415299 |           288 |      5.63 |      3.35 |     2.16 |        1.5 | glucose_value_mg_dl_t-0 (98.3%), Velocity (1.1%), Libre GL_t-0 (0.0%), sin_hour (0.0%), cos_hour (0.0%)                  |
| CGMacros_Libre  |    455705 |           288 |      4.06 |      1.71 |     1.03 |        1.1 | glucose_value_mg_dl_t-0 (99.2%), Velocity (0.7%), Window_Std (0.0%), glucose_value_mg_dl_t-1 (0.0%), sin_hour (0.0%)     |
| CGMND           |    114409 |            18 |     14.64 |     14.38 |     9.86 |        9.6 | glucose_value_mg_dl_t-0 (44.5%), Window_Mean (11.0%), glucose_value_mg_dl_t-5 (5.9%), Window_Std (5.2%), cos_hour (4.2%) |
| GLAM            |  26165917 |            36 |     14.08 |     13.57 |     9.77 |       10.1 | glucose_value_mg_dl_t-0 (84.0%), Acceleration (2.6%), glucose_value_mg_dl_t-5 (2.0%), Window_Std (1.7%), Velocity (1.5%) |
| HUPA-UCM        |    309092 |           126 |     17.63 |     15.43 |    10.62 |        8.7 | glucose_value_mg_dl_t-0 (88.1%), Velocity (5.1%), Window_Std (0.2%), glucose_value_mg_dl_t-5 (0.2%), cos_hour (0.2%)     |
| IOBP2           |  14027905 |            36 |     27.32 |     25    |    17.6  |       11.8 | glucose_value_mg_dl_t-0 (91.4%), Velocity (4.9%), glucose_value_mg_dl_t-5 (0.6%), Window_Std (0.5%), Acceleration (0.4%) |
| Park_2025       |     23064 |            36 |     20.24 |     20.71 |    15.76 |       16.9 | Window_Mean (56.2%), Window_AUC (8.6%), HBGI (4.4%), glucose_value_mg_dl_t-3 (2.5%), glucose_value_mg_dl_t-4 (2.5%)      |
| PEDAP           |   7055811 |            54 |     30.83 |     28.7  |    20.53 |       15   | glucose_value_mg_dl_t-0 (85.5%), Velocity (7.1%), Acceleration (0.7%), Window_Std (0.7%), glucose_value_mg_dl_t-5 (0.5%) |
| UCHTT1DM        |     27220 |            36 |     18.82 |     18.34 |    12.46 |       12.4 | glucose_value_mg_dl_t-0 (83.8%), Velocity (2.4%), Window_Std (1.5%), Acceleration (1.1%), cos_hour (1.0%)                |