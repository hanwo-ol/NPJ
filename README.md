# Glucose-ML Prediction Project

## 소개 (Introduction)
이 저장소는 원본 논문인 [Glucose-ML](https://arxiv.org/abs/2507.14077)의 공식 깃허브 저장소를 복제(Clone)하여, 연속 혈당 측정(CGM) 데이터를 활용한 **혈당 예측 문제**로 전환하려는 연구와 실험을 담고 있습니다.

현재 다음과 같은 단계로 예측 모델링을 시도하고 있습니다:
- **`7_Baseline_Linear_Regression/`**: 기본 선형 회귀(Baseline Linear Regression)를 활용한 초기 예측 모델 구축
- **`8_Classic_ML/`**: 전통적인 머신러닝(Classic Machine Learning) 기법들을 적용한 혈당 예측
- **`9_Tier_2.5_Feature_Engineering/`**: 모델 성능 향상을 위해 심화된 특성 공학(Feature Engineering) 진행

**향후 연구 계획 (Future Work):**
- **Tier 3**: 딥러닝을 제외한, 더욱 진보된 최신 머신러닝 알고리즘들을 탐색하고 시도할 예정입니다.
- **Tier 4**: 이전 단계에서 개발된 머신러닝 모델들을 활용하여, 다른 도메인이나 다른 혈당 데이터셋에 대한 전이 학습(Transfer Learning) 연구를 진행할 계획입니다.

## 인용 (Citation)
이 프로젝트는 원본 Glucose-ML 프로젝트의 데이터와 코드를 기반으로 하고 있습니다. 관련 연구를 사용할 경우 아래 원본 논문을 인용해 주시기 바랍니다:

**Paper 1:**
```text
Prioleau, T., Lu, B. and Cui, Y., 2025. Glucose-ML: A collection of longitudinal diabetes datasets for development of robust AI solutions. arXiv preprint arXiv:2507.14077.
https://doi.org/10.48550/arXiv.2507.14077
```

**Paper 2 (Under review):**
```text
Pontius, R., Pitakanonda, P., Li, Z., Lhabaik, K., Wang, F., Lu, B. and Cui, Y., Prioleau, T., 2026. Glucose-ML: An evolving collection of continuous glucose datasets to accelerate data-centric AI for diabetes.
```

- 원본 GitHub 저장소: [Augmented-Health-Lab/Glucose-ML-Project](https://github.com/Augmented-Health-Lab/Glucose-ML-Project)
