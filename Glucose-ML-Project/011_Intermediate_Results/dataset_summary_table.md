# Table 1. Characteristics of the 12 CGM Datasets

> **Abbreviations:** N = number of subjects; Readings = total CGM time-points; Windows = 30-min prediction windows (Lookback 6 + Forecast 6 steps); TIR = Time In Range (70–180 mg/dL); TAR = Time Above Range (>180); TBR = Time Below Range (<70).

| Dataset         | Cohort   | Country   |   N | Readings   | Windows   |      Duration (days, max)| Sensor          | Interval   | Mean G   (mg/dL)  |   TIR (%) |   TAR (%) |   TBR (%) |
|:----------------|:---------|:----------|----:|:-----------|:----------|--------------:|:----------------|:-----------|:-----------|----------:|----------:|----------:|
| IOBP2           | T1DM     | USA       | 440 | 14.33M     | 14.0M     |        1951.9 | Dexcom          | 5 min      | 170.7±68.5 |      60.8 |      37.1 |       2.1 |
| PEDAP           | T1DM     | USA       | 103 | 7.23M      | 7.1M      |         481   | Dexcom          | 5 min      | 158.2±65.5 |      67.4 |      29.8 |       2.8 |
| AIDET1D         | T1DM     | Turkey    |  29 | 484K       | 472K      |         378.3 | Dexcom G5       | 5 min      | 154.9±64.2 |      68.7 |      27.9 |       3.4 |
| Bris-T1D_Open   | T1DM     | Australia |  20 | 849K       | 819K      |         225.7 | Libre/Dexcom    | 5 min      | 156.2±58.0 |      70.7 |      27.5 |       1.8 |
| UCHTT1DM        | T1DM     | USA       |  20 | 29K        | 27K       |           6.8 | Dexcom          | 5 min      | 110.8±45.4 |      85.5 |       8.6 |       5.9 |
| HUPA-UCM        | T1DM/ND  | Spain     |  25 | 309K       | 309K      |         574   | Dexcom          | 5 min      | 141.4±57.1 |      71.7 |      21.7 |       6.6 |
| GLAM            | GDM      | Spain     | 886 | 26.59M     | 26.2M     |         237.2 | Libre           | 15 min     | 100.2±21.4 |      94.8 |       0.3 |       4.9 |
| BIGIDEAs        | ND/PreD  | USA       |  16 | 37K        | 36K       |           9.9 | Dexcom G6       | 5 min      | 114.5±23.1 |      97.7 |       1.8 |       0.6 |
| CGMND           | ND       | Multiple  |  45 | 117K       | 114K      |         238.3 | Various         | 5 min      | 106.4±22.0 |      98   |       0.8 |       1.2 |
| Park_2025       | ND       | Korea     |  38 | 24K        | 23K       |           0   | Dexcom G7       | 5 min      | 112.6±32.7 |      92.2 |       4.2 |       3.6 |
| CGMacros_Dexcom | ND       | USA       |  30 | 418K       | 415K      |          19.1 | Dexcom G6       | 5 min      | 141.5±43.0 |      84.7 |      14.9 |       0.4 |
| CGMacros_Libre  | ND       | USA       |  30 | 458K       | 456K      |          19.2 | FreeStyle Libre | 15 min     | 110.7±42.4 |      82.5 |       6.7 |      10.8 |

---

## Dataset Scale Breakdown

### T1DM
- Datasets: IOBP2, PEDAP, AIDET1D, Bris-T1D_Open, UCHTT1DM
- Total subjects: **612**
- Total readings: **22.93M**
- Mean glucose: 150.2 ± 60.3 mg/dL
- Pooled TIR: 70.6% | TAR: 26.2% | TBR: 3.2%

### T1DM/ND
- Datasets: HUPA-UCM
- Total subjects: **25**
- Total readings: **309K**
- Mean glucose: 141.4 ± 57.1 mg/dL
- Pooled TIR: 71.7% | TAR: 21.7% | TBR: 6.6%

### GDM
- Datasets: GLAM
- Total subjects: **886**
- Total readings: **26.59M**
- Mean glucose: 100.2 ± 21.4 mg/dL
- Pooled TIR: 94.8% | TAR: 0.3% | TBR: 4.9%

### ND/PreD
- Datasets: BIGIDEAs
- Total subjects: **16**
- Total readings: **37K**
- Mean glucose: 114.5 ± 23.1 mg/dL
- Pooled TIR: 97.7% | TAR: 1.8% | TBR: 0.6%

### ND
- Datasets: CGMND, Park_2025, CGMacros_Dexcom, CGMacros_Libre
- Total subjects: **143**
- Total readings: **1.02M**
- Mean glucose: 117.8 ± 35.0 mg/dL
- Pooled TIR: 89.3% | TAR: 6.6% | TBR: 4.0%

