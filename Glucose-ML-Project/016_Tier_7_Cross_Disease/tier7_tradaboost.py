"""
Tier 7: TrAdaBoost — Instance-Level Transfer Learning
=======================================================
Dai et al. (2007) "Boosting for Transfer Learning"

알고리즘:
  - 소스 + 타겟 혼합 학습
  - 타겟 오차가 큰 소스 인스턴스의 가중치를 반복적으로 감소
  - 타겟 분포와 다른 소스 노이즈를 자동 제거
  - 후반 N/2 모델 앙상블로 안정적 예측
"""

import sys
import numpy as np
import lightgbm as lgb
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from tier7_config import Tier7Config


class TrAdaBoostRegressor:
    """
    TrAdaBoost for regression (LightGBM base learner).

    Parameters
    ----------
    n_iterations : int
        전체 부스팅 반복 수
    n_ensemble : int
        앙상블에 사용할 후반 모델 수 (마지막 n_ensemble개 평균)
    lgbm_params : dict
        LightGBM 파라미터
    lgbm_rounds : int
        각 반복에서의 LightGBM 라운드
    """

    def __init__(self,
                 n_iterations: int   = None,
                 n_ensemble:   int   = None,
                 lgbm_params:  dict  = None,
                 lgbm_rounds:  int   = 300):

        self.n_iterations = n_iterations or Tier7Config.TRADABOOST_N_ITER
        self.n_ensemble   = n_ensemble   or Tier7Config.TRADABOOST_ENSEMBLE
        self.lgbm_params  = lgbm_params  or dict(Tier7Config.LGBM_PARAMS)
        self.lgbm_rounds  = lgbm_rounds
        self.models_      = []

    def fit(self, X_src: np.ndarray, y_src: np.ndarray,
                  X_tgt: np.ndarray, y_tgt: np.ndarray,
                  X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Parameters
        ----------
        X_src, y_src : 소스 데이터 (T1D)
        X_tgt, y_tgt : 타겟 훈련 데이터 (T2D train)
        X_val, y_val : 타겟 검증 데이터 (LightGBM early stopping용)
        """
        ns = len(X_src)
        nt = len(X_tgt)
        n  = ns + nt

        # 초기 가중치: 소스와 타겟 균등 (각 집합 합계 = 0.5)
        w_src = np.ones(ns) / (2 * ns)
        w_tgt = np.ones(nt) / (2 * nt)

        beta = 1.0 / (1.0 + np.sqrt(2.0 * np.log(ns) / self.n_iterations))

        self.models_ = []

        for t in tqdm(range(self.n_iterations),
                       desc="    TrAdaBoost", leave=False, ncols=70):
            # 가중치 정규화
            w_all = np.concatenate([w_src, w_tgt])
            w_all = w_all / w_all.sum()

            X_all = np.vstack([X_src, X_tgt])
            y_all = np.concatenate([y_src, y_tgt])

            # LightGBM 학습
            ds_train = lgb.Dataset(X_all, label=y_all, weight=w_all)
            callbacks = [lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(-1)]
            if X_val is not None:
                ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train)
                model = lgb.train(
                    self.lgbm_params,
                    ds_train,
                    num_boost_round=self.lgbm_rounds,
                    valid_sets=[ds_val],
                    callbacks=callbacks,
                )
            else:
                model = lgb.train(
                    self.lgbm_params,
                    ds_train,
                    num_boost_round=self.lgbm_rounds,
                    callbacks=[lgb.log_evaluation(-1)],
                )
            self.models_.append(model)

            # 타겟 훈련 오차 계산 (정규화)
            y_tgt_pred = model.predict(X_tgt)
            errors = np.abs(y_tgt_pred - y_tgt)
            max_err = errors.max()
            if max_err == 0:
                norm_errors = np.zeros_like(errors)
            else:
                norm_errors = errors / max_err  # [0, 1]

            # 소스 가중치 감소: 타겟 오차에 비례하여 소스 가중치 업데이트
            # 소스 예측 오차
            y_src_pred = model.predict(X_src)
            src_errors = np.abs(y_src_pred - y_src)
            max_src = src_errors.max()
            if max_src > 0:
                norm_src = src_errors / max_src
            else:
                norm_src = np.zeros_like(src_errors)

            # 소스 인스턴스: 타겟 분포에 잘 맞는 것은 유지, 아닌 것은 감소
            # norm_src 가 클수록 해당 소스 인스턴스는 타겟에서 멀다 → 가중치 감소
            w_src = w_src * (beta ** norm_src)

            # 타겟 가중치는 변경하지 않음 (항상 중요)
            # (원 논문에서는 타겟 오차 기반 업데이트도 있으나
            #  회귀 안정성을 위해 타겟 가중치 고정)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """후반 n_ensemble 모델의 평균 앙상블 예측."""
        if not self.models_:
            raise RuntimeError("fit() must be called before predict()")
        ensemble_models = self.models_[-self.n_ensemble:]
        preds = np.stack([m.predict(X) for m in ensemble_models], axis=0)
        return preds.mean(axis=0)

    @property
    def best_model(self):
        """SHAP 분석용: 마지막 모델 반환."""
        return self.models_[-1] if self.models_ else None
