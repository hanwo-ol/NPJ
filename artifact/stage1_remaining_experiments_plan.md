# Stage 1 Remaining Experiments Plan: Generalizability and Reproducibility of CGM AI

This document establishes the detailed plan, mathematical definitions, and implementation protocol for the remaining Stage 1 experiments of the Glucose CGM research project. These experiments are designed to address reviewer feedback and align with the publication standards of *npj Digital Medicine*.

---

## 1. S1-1: Within Variation (Computational Reproducibility)

### 1.1 Objective
Measure how much model performance and clinical conclusions fluctuate when random seed and hyperparameter configurations vary under identical data setups.

### 1.2 Protocol
* **Fixed Data**: Source pool (T1D pool) and Target cohort (ShanghaiT2DM).
* **Base Model**: LightGBM.
* **Seed Sweep**: 10 random seeds (42, 123, 456, 789, 1024, 2048, 3333, 5555, 7777, 9999).
* **Hyperparameter Sweeps (Optional)**:
  * `learning_rate` ∈ {0.01, 0.05, 0.1}
  * `num_leaves` ∈ {31, 63, 127}
  * `max_depth` ∈ {-1, 6, 10}
* **Evaluation Schemes**: 5-Way models (source_only, target_only, mixed, coral, tradaboost).

### 1.3 Target Metrics
* **Coefficient of Variation (CV)**:
  $$\text{CV} = \frac{\sigma_{\text{RMSE}}}{\mu_{\text{RMSE}}}$$
* **Interquartile Range (IQR)**: Of the RMSE distributions across seeds.
* **Flip Rate (FR)**: The frequency with which the clinical conclusion "Transfer learning outperforms Target-Only" is reversed:
  $$\text{FR} = \frac{\sum_{s} \mathbb{I}(\text{RMSE}_{\text{TL}, s} > \text{RMSE}_{\text{Target\_Only}, s})}{\text{Total Seeds}}$$
* **Rank Stability (Kendall's $\tau$)**: Pairwise rank correlation of the 5 models' ordering across seeds.

---

## 2. S1-2: Leave-One-Dataset-Out (LODO) Generalizability

### 2.1 Objective
Evaluate the cross-domain generalization penalty of CGM AI when deployed at a new clinic/site that was completely unseen during training.

### 2.2 Protocol
Following the project's sampling frequency separation rule, the 12 active datasets are split into two groups and evaluated independently:

#### 5-Minute Frequency Group (10 Datasets)
* **T1D**: AIDET1D, AZT1D, CGMND, D1NAMO, HUPA-UCM, IOBP2, PEDAP, PhysioCGM, RT-CGM, SENCE, WISDM, FLAIR, SHD, ReplaceBG.
* **ND / Mixed**: BIGIDEAs, Colas_2019, CITY, Hall_2018.

#### 15-Minute Frequency Group (2 Datasets)
* **T1D / T2D**: Bris-T1D_Open, ShanghaiT1DM, ShanghaiT2DM.

For each target dataset $D_i$ in its respective group:
1. **Train Set**: Combine all other datasets in the same frequency group (excluding $D_i$).
2. **Test Set**: Test partition of $D_i$.
3. **Models**: Source-Only, Target-Only (trained on $D_i$ train split), CORAL, and TrAdaBoost.

### 2.3 Target Metrics
* LODO RMSE, MAE, and MARD for each target.
* Heatmap representing the $N \times M$ matrix (Target datasets vs. Transfer methods).

---

## 3. S1-3: Domain Distance vs. Performance Degradation

### 3.1 Objective
Verify if mathematical distances between source and target feature distributions can serve as a proxy to predict transfer performance degradation prior to deployment.

### 3.2 Protocol
For each (Source Pool, Target Dataset) pair in the LODO splits, compute the following metrics in the 22-dimensional feature space:

1. **Maximum Mean Discrepancy (MMD)**:
   $$\text{MMD}^2(X_S, X_T) = \frac{1}{n_s^2} \sum_{i,j} k(x_i^s, x_j^s) - \frac{2}{n_s n_t} \sum_{i,j} k(x_i^s, x_j^t) + \frac{1}{n_t^2} \sum_{i,j} k(x_i^t, x_j^t)$$
   using an RBF kernel $k(x, y) = \exp(-\gamma ||x-y||^2)$.
2. **Proxy-A-Distance (PAD)**:
   Train a linear classifier to distinguish between source and target samples. If the classifier error rate is $\epsilon$, then:
   $$\text{PAD} = 2(1 - 2\epsilon)$$
3. **Covariance Frobenius Norm**:
   $$\text{Dist}_{\text{Cov}} = ||\Sigma_S - \Sigma_T||_F$$
4. **Wasserstein Distance**: Computes optimal transport cost between the target and source distributions.

### 3.3 Correlation Analysis
* Perform linear regression: $\Delta \text{RMSE} = \alpha + \beta \cdot \text{Distance}$ (where $\Delta \text{RMSE} = \text{LODO\_RMSE} - \text{Self\_RMSE}$).
* Report Spearman's $\rho$, Pearson's $r$, and the coefficient of determination $R^2$.

---

## 4. S1-4: Temporal Limitations of Static Models

### 4.1 S1-4a: Residual Autocorrelation (ACF) Analysis
* **Core Logic**: If a model successfully captures time-series dynamics, the residual errors should resemble white noise (i.e., no sequential correlation). Significant autocorrelation in residuals indicates unmodeled temporal structures.
* **Formula**:
  For each model $M$ and target $T$, extract the ordered residual sequence $e_t = y_t - \hat{y}_t$ for each patient. Calculate ACF for lags $k = 1, 2, \dots, 12$:
  $$\text{ACF}(k) = \frac{\sum_{t=k+1}^N (e_t - \bar{e})(e_{t-k} - \bar{e})}{\sum_{t=1}^N (e_t - \bar{e})^2}$$
* **Statistical Test**:
  Run the Ljung-Box Q-test at significance level $\alpha = 0.01$:
  $$Q = N(N+2) \sum_{k=1}^m \frac{\hat{\rho}_k^2}{N-k}$$
  If $p < 0.01$, reject the null hypothesis $H_0$ (meaning residuals are autocorrelated).
* **Summary Metrics**: ACF(lag=1) value and Durbin-Watson statistic.

### 4.2 S1-4b: Segment-wise Error Decomposition
* **Core Logic**: Analyze if static transfer models perform poorly specifically during rapid glucose transitions.
* **Formula**:
  Compute glucose rate of change (velocity):
  $$v_t = \frac{g_t - g_{t-1}}{\Delta t}$$
* **Segment Definitions**:
  * **Stable**: $|v_t| \le 1.0 \text{ mg/dL/min}$
  * **Rapid Rise**: $v_t > 2.0 \text{ mg/dL/min}$
  * **Rapid Fall**: $v_t < -2.0 \text{ mg/dL/min}$
  * **Transient**: $1.0 < |v_t| \le 2.0 \text{ mg/dL/min}$
* **Metrics**: Compare the RMSE of Target-Only vs. CORAL/TrAdaBoost inside each segment.

---

## 5. S1-5: Regression to 3-Class Classification Transition

### 5.1 Objective
Convert continuous glucose predictions into discrete clinical classifications to analyze reproducibility on categorical outcomes (as requested in professor meetings).

### 5.2 Category Boundary (Clinical Standard)
* **Hypoglycemia**: $\text{CGM} < 70 \text{ mg/dL}$
* **In-Range (Normal)**: $70 \le \text{CGM} \le 180 \text{ mg/dL}$
* **Hyperglycemia**: $\text{CGM} > 180 \text{ mg/dL}$

### 5.3 Classification Metrics
* **Cohen's Kappa ($\kappa$)**: Standard measure for inter-rater agreement.
* **Macro-averaged F1-score**.
* **Hypo/Hyper-specific Sensitivity and Specificity**.

---

## 6. Execution Pipeline and Script Design

The remaining experiments will be implemented as modular scripts within `Glucose-ML-Project`:

1. **`018_Tier_8_Reproducibility/run_within_variation.py`**:
   * Implements S1-1 and S1-5.
   * Runs the 5-way setup across 10 random seeds and saves a summary CSV of mean, std, CV, and flip rates.
2. **`018_Tier_8_Reproducibility/run_lodo.py`**:
   * Implements S1-2.
   * Automates the loop of leaving one dataset out within the sampling rate groups, calling LightGBM/CORAL/TrAdaBoost.
3. **`018_Tier_8_Reproducibility/analyze_domain_distance.py`**:
   * Implements S1-3.
   * Computes MMD and PAD on feature representations and runs the correlation models.
4. **`018_Tier_8_Reproducibility/analyze_temporal_limitations.py`**:
   * Implements S1-4 (ACF analysis and velocity-based RMSE decomposition).

---

## 7. Timeline

* **Week 1**: Implement `run_within_variation.py` (S1-1) and integrate the 3-class classification thresholding (S1-5).
* **Week 2**: Implement `run_lodo.py` (S1-2) and `analyze_temporal_limitations.py` (S1-4).
* **Week 3**: Implement `analyze_domain_distance.py` (S1-3) and generate correlation curves.
* **Week 4**: Aggregate all metrics, plot the violin charts, heatmaps, and regression lines for the final *npj Digital Medicine* manuscript.
