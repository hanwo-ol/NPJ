# [Algorithm Reference] Robust Angle-based Transfer Learning (AngleTL)

**논문 원제:** "Robust angle-based transfer learning in high dimensions" (Gu et al., 2023)  
본 문서는 논문에서 제안된 핵심 알고리즘 3가지를 원형(Original Paper's Context)에 가깝게 자연어 버전과 수식(Pseudo-Code) 버전으로 분리하여 정리한 레퍼런스 가이드입니다.

---

## 1. 기본 AngleTL 최적화 (Base AngleTL Algorithm)

단일 소스 모델이 주어졌을 때, 타겟 데이터의 예측 성능을 향상시키기 위한 기본 각도 기반 전이 학습 과정입니다.

### 1.1 자연어 버전
1. **소스 파라미터 수신:** 외부 병원이나 연구에서 이미 학습이 완료된 회귀 모델의 파라미터(가중치 벡터) $\hat{w}$를 전달받습니다. (환자 데이터는 필요 없습니다.)
2. **하이퍼파라미터 그리드 준비:** 모델이 얼마나 과적합을 피할지 결정하는 정규화 강도($\lambda$)와, 소스 모델의 방향을 얼마나 강하게 따를지 결정하는 각도 규제 강도($\eta$)의 조합들을 준비합니다.
3. **교차 검증(Cross-Validation):** 타겟 데이터에 대해 교차 검증을 수행합니다. 각 폴드(Fold)마다 타겟 오차를 줄이는 동시에, 소스 모델 파라미터와 내적(각도 일치)이 커지도록 최적의 $\beta$를 찾습니다.
4. **최적 모델 선택:** 교차 검증 오차가 가장 낮은 최적의 $(\lambda, \eta)$ 조합을 선택한 후, 전체 타겟 데이터로 최종 타겟 모델 파라미터 $\hat{\beta}$를 산출합니다.

### 1.2 수식 / 코드 버전 (Pseudo-Code)
**입력:** 타겟 데이터 $X \in \mathbb{R}^{n \times p}, Y \in \mathbb{R}^n$, 소스 모델 파라미터 $\hat{w} \in \mathbb{R}^p$  
**목적 함수:** $J(\beta) = \frac{1}{n}\|Y - X\beta\|_2^2 + \lambda\|\beta\|_2^2 - 2\eta \hat{w}^\top \beta$

```text
def AngleTL_Base(X, Y, w_hat, lambda_grid, eta_grid):
    best_error = INFINITY
    best_params = (None, None)

    FOR lambda IN lambda_grid:
        FOR eta IN eta_grid:
            cv_error = 0
            FOR fold_train_X, fold_train_Y, fold_val_X, fold_val_Y IN CV_Folds(X, Y):
                
                # 목적 함수에 대한 Closed-form 해 (해석적 해)
                # beta = (X^T X + n * \lambda * I)^(-1) * (X^T Y + n * \eta * w_hat)
                inv_matrix = Inverse(Transpose(fold_train_X) * fold_train_X + n * lambda * IdentityMatrix)
                beta_temp = inv_matrix * (Transpose(fold_train_X) * fold_train_Y + n * eta * w_hat)
                
                # 검증
                error = MeanSquaredError(fold_val_Y, fold_val_X * beta_temp)
                cv_error += error
            
            IF cv_error < best_error:
                best_error = cv_error
                best_params = (lambda, eta)
                
    lambda_opt, eta_opt = best_params
    
    # 전체 타겟 데이터로 최종 모델 학습
    inv_matrix_final = Inverse(Transpose(X) * X + n * lambda_opt * IdentityMatrix)
    beta_final = inv_matrix_final * (Transpose(X) * Y + n * eta_opt * w_hat)
    
    RETURN beta_final
```

---

## 2. 논문의 Algorithm 1: Q-Aggregation (Supervised Multi-Source)

다수의 소스 모델이 존재하고, 타겟 데이터의 일부를 검증용(Validation)으로 사용할 수 있을 때 최적의 소스 모델 가중치를 찾는 방식입니다.

### 2.1 자연어 버전
1. **타겟 데이터 분할:** 타겟 데이터의 일부(예: 30%)를 검증 세트(Validation Set)로 따로 빼둡니다.
2. **각 소스별 예측:** 타겟 검증 세트의 피처 데이터를 $K$개의 각 소스 모델에 통과시켜 $K$개의 예측값 세트를 얻습니다.
3. **가중치 학습 (Q-Aggregation):** 검증 세트의 실제 정답(Label)과 $K$개의 예측값 세트를 비교하여, 오차를 가장 줄여주는 최적의 소스 모델 조합 비율(가중치)을 찾습니다. (이때 합계가 1이 되는 심플렉스 제약을 걸 수 있습니다.)
4. **글로벌 소스 생성:** 찾아낸 가중치를 이용해 $K$개의 소스 모델 파라미터를 가중합(Weighted Sum)하여 단일 '통합 소스 파라미터'로 만듭니다.
5. **AngleTL 적용:** 이렇게 만들어진 통합 소스 파라미터를 앞선 `AngleTL_Base` 알고리즘에 투입합니다.

### 2.2 수식 / 코드 버전 (Pseudo-Code)
**입력:** 타겟 훈련 데이터 $D_{train}$, 타겟 검증 데이터 $D_{val} = (X_{val}, Y_{val})$, $K$개의 소스 파라미터 $\hat{w}_1 \dots \hat{w}_K$

```text
def Algorithm_1_Q_Aggregation(D_train, D_val, [w_1, ..., w_K]):
    X_val, Y_val = D_val
    n_val = length(Y_val)
    
    # 각 소스 모델별 예측값 생성 행렬 Z (크기: n_val x K)
    Z = Zeros(n_val, K)
    FOR k = 1 TO K:
        Z[:, k] = X_val * w_k
        
    # Q-Aggregation: 실제 정답 Y_val과의 예측 오차를 최소화하는 가중치 벡터 s 도출
    # s는 보통 양수이고 합이 1이 되도록(Simplex) 제한을 둘 수 있음
    s_hat = argmin_{s >= 0, sum(s)=1} ||Y_val - Z * s||_2^2
    
    # 통합 글로벌 소스 파라미터 생성
    w_global = Zeros(p)
    FOR k = 1 TO K:
        w_global += s_hat[k] * w_k
        
    # AngleTL_Base 알고리즘에 전달하여 최종 타겟 모델 학습
    beta_final = AngleTL_Base(D_train.X, D_train.Y, w_global)
    
    RETURN beta_final
```

---

## 3. 논문의 Algorithm 2: Spectral Weighting (Unsupervised Multi-Source)

다수의 소스 모델이 존재하지만, 타겟 검증 데이터를 단 하나도 희생하고 싶지 않을 때 사용하는 혁신적인 비지도 앙상블 방식입니다.

### 3.1 자연어 버전
1. **소스 파라미터 정규화:** $K$개 병원의 소스 파라미터 벡터들의 크기를 모두 1로 맞춰줍니다. (각도, 즉 방향성만 비교하기 위함)
2. **주성분 분석(PCA):** 정규화된 $K$개의 벡터를 쌓아 올린 후 고유값 분해(Eigen-decomposition)를 수행하여, 모든 소스들이 공통적으로 가리키는 지배적인 방향(첫 번째 고유벡터)을 찾습니다.
3. **스펙트럴 가중치 계산:** 이 공통 방향에 기여하는 정도(고유벡터의 절대값)를 그대로 해당 소스 모델의 가중치로 사용합니다. 대다수와 방향이 비슷하면 가중치가 커지고, 튀는 이상한 모델이면 가중치가 작아집니다.
4. **글로벌 소스 생성 & AngleTL 적용:** 가중합을 통해 단일 통합 소스 파라미터를 만들고, `AngleTL_Base` 알고리즘에 투입합니다.

### 3.2 수식 / 코드 버전 (Pseudo-Code)
**입력:** 타겟 전체 데이터 $X, Y$, $K$개의 소스 파라미터 $\hat{w}_1 \dots \hat{w}_K$

```text
def Algorithm_2_Spectral_Weighting(X, Y, [w_1, ..., w_K]):
    # 1. 파라미터 방향 정규화
    w_bar_matrix = Zeros(K, p)
    FOR k = 1 TO K:
        w_bar_matrix[k, :] = w_k / L2_Norm(w_k)
        
    # 2. 공분산 행렬(Covariance-like) 계산 및 Eigen-decomposition
    # C의 크기는 K x K 가 됨
    C = w_bar_matrix * Transpose(w_bar_matrix)
    eigenvalues, eigenvectors = EigenDecomposition(C)
    
    # 3. 첫 번째 주성분(가장 큰 고유값에 해당하는 고유벡터) 추출
    u_1 = eigenvectors[index_of_max(eigenvalues)]
    
    # 4. 고유벡터 요소의 절대값을 가중치로 할당
    s_hat = Zeros(K)
    FOR k = 1 TO K:
        s_hat[k] = Abs(u_1[k])
        
    # 5. 최종 통합 글로벌 소스 벡터 생성
    w_global = Zeros(p)
    FOR k = 1 TO K:
        w_global += s_hat[k] * w_bar_matrix[k, :]
        
    # 6. 타겟 데이터 전체를 사용하여 AngleTL 수행
    beta_final = AngleTL_Base(X, Y, w_global)
    
    RETURN beta_final
```
