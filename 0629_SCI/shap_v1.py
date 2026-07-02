"""
ADAP SHAP vs NMF 비교 실험
===========================
목적: SHAP 기반 feature 추출이 NMF Loading과 어떻게 다른지 비교.
      논문 ablation 근거 확보.

흐름:
  1. User × Feature 행렬 로드 (앞선 실험과 동일)
  2. ratings에서 (user_feature, rating) 샘플 구성
  3. XGBoost 평점 예측 모델 학습
  4. Global SHAP 계산
  5. NMF Loading과 나란히 비교

필요 파일:
  ml-32m/ratings.csv
  adap_results_32m_B_deviation/user_feature_matrix.csv
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, time, warnings
warnings.filterwarnings("ignore")

DATA_DIR  = "ml-32m"
FEAT_CSV  = "adap_results_32m_B_deviation/user_feature_matrix.csv"
OUT_DIR   = "adap_shap_results"
os.makedirs(OUT_DIR, exist_ok=True)

K            = 3
RANDOM_STATE = 42
SAMPLE_SIZE  = 200_000   # 학습 샘플 수 (32M 전체는 과함)
CHUNK_SIZE   = 2_000_000


# ─── 1. Feature Matrix 로드 ──────────────────────────────────────────
print("[1] Feature Matrix 로드...")
feat_df  = pd.read_csv(FEAT_CSV, index_col=0)
nmf_cols = [c for c in feat_df.columns if not c.startswith("NMF_EXCLUDE__")]
feat_nmf = feat_df[nmf_cols]   # NMF 입력과 동일한 변수
print(f"    shape: {feat_df.shape}, NMF 입력 변수: {len(nmf_cols)}개")


# ─── 2. ratings 샘플 구성 ────────────────────────────────────────────
# (userId, movieId, rating) → userId로 user feature join
# → (user_feature_vector, rating) 쌍 SAMPLE_SIZE개 추출
print(f"\n[2] ratings 샘플 구성 ({SAMPLE_SIZE:,}건)...")
t0 = time.time()

chunks = []
for chunk in pd.read_csv(
    f"{DATA_DIR}/ratings.csv",
    usecols=["userId", "movieId", "rating"],
    dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    chunksize=CHUNK_SIZE
):
    # feature matrix에 있는 사용자만 유효
    sub = chunk[chunk["userId"].isin(feat_nmf.index)]
    chunks.append(sub)

ratings_all = pd.concat(chunks, ignore_index=True)
print(f"    전체 valid ratings: {len(ratings_all):,}건 ({time.time()-t0:.1f}s)")

# 샘플링
ratings_sample = ratings_all.sample(
    n=min(SAMPLE_SIZE, len(ratings_all)),
    random_state=RANDOM_STATE
)

# user feature join
X_raw = feat_nmf.loc[ratings_sample["userId"].values].values
y     = ratings_sample["rating"].values
print(f"    학습 데이터: X={X_raw.shape}, y={y.shape}")


# ─── 3. XGBoost 평점 예측 모델 학습 ─────────────────────────────────
print("\n[3] XGBoost 학습...")
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=RANDOM_STATE
)

model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)

y_pred = model.predict(X_test)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"    Test RMSE: {rmse:.4f}")
print(f"    (참고: 사용자 평균 평점 예측 baseline RMSE ≈ "
      f"{np.sqrt(mean_squared_error(y_test, np.full_like(y_test, y_train.mean()))):.4f})")


# ─── 4. Global SHAP 계산 ─────────────────────────────────────────────
print("\n[4] Global SHAP 계산 (TreeExplainer)...")
# 속도를 위해 background sample 사용
background  = X_train[np.random.choice(len(X_train), 1000, replace=False)]
explainer   = shap.TreeExplainer(model, background)

# SHAP 계산은 전체 test set에서 샘플링
shap_sample = X_test[np.random.choice(len(X_test), 5000, replace=False)]
shap_values = explainer.shap_values(shap_sample)   # (n_samples, n_features)

# Global importance: mean |SHAP|
global_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.Series(global_shap, index=nmf_cols).sort_values(ascending=False)

print("\n  Global SHAP 상위 변수:")
for feat, val in shap_importance.head(10).items():
    bar = "█" * int(val * 100)
    print(f"    {bar:<20} {val:.4f}  {feat}")


# ─── 5. NMF Loading 계산 ─────────────────────────────────────────────
print("\n[5] NMF Loading (비교 기준)...")
X_scaled = StandardScaler().fit_transform(feat_nmf.values)
X_nn     = np.clip(X_scaled, 0, None)
nmf      = NMF(n_components=K, random_state=RANDOM_STATE,
               max_iter=500, init="nndsvda")
nmf.fit(X_nn)
H        = nmf.components_   # (K, features)

# NMF 전체 loading magnitude (각 변수의 모든 Factor에 걸친 최대값)
nmf_importance = pd.Series(
    H.max(axis=0), index=nmf_cols
).sort_values(ascending=False)

print("\n  NMF Loading 상위 변수 (전체 Factor 최대값 기준):")
for feat, val in nmf_importance.head(10).items():
    bar = "█" * int(val / 2)
    print(f"    {bar:<20} {val:.4f}  {feat}")


# ─── 6. SHAP vs NMF 비교 시각화 ──────────────────────────────────────
print("\n[6] 비교 시각화...")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# SHAP
top_shap = shap_importance.head(15)
axes[0].barh(top_shap.index[::-1], top_shap.values[::-1], color="#E53935")
axes[0].set_title("Global SHAP Importance\n(XGBoost 평점 예측 모델 기반)",
                  fontsize=11)
axes[0].set_xlabel("Mean |SHAP value|")
axes[0].grid(axis="x", alpha=0.3)

# NMF — Factor별로 색 구분
colors_k = ["#1976D2", "#388E3C", "#F57C00"]
for k in range(K):
    loading_k = pd.Series(H[k], index=nmf_cols).sort_values(ascending=False)
    top15_idx = nmf_importance.head(15).index
    vals      = loading_k[top15_idx]
    axes[1].barh(
        [i + k * 0.25 for i in range(len(top15_idx))],
        vals.values[::-1],
        height=0.25,
        color=colors_k[k],
        label=f"F{k+1}",
        alpha=0.85
    )

axes[1].set_yticks(range(len(nmf_importance.head(15))))
axes[1].set_yticklabels(nmf_importance.head(15).index[::-1])
axes[1].set_title("NMF Factor Loading\n(Factor별 분해, K=3)",
                  fontsize=11)
axes[1].set_xlabel("Loading value")
axes[1].legend()
axes[1].grid(axis="x", alpha=0.3)

plt.suptitle("SHAP vs NMF: Feature Importance 비교\n"
             "SHAP=예측 기여도(단일 스칼라) / NMF=잠재 패턴별 구조(K개 벡터)",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/shap_vs_nmf_comparison.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"    저장: {OUT_DIR}/shap_vs_nmf_comparison.png")


# ─── 7. 핵심 차이 요약 출력 ──────────────────────────────────────────
print(f"\n{'='*60}")
print("  SHAP vs NMF 핵심 차이 요약")
print(f"{'='*60}")

top10_shap = set(shap_importance.head(10).index)
top10_nmf  = set(nmf_importance.head(10).index)
overlap    = top10_shap & top10_nmf
only_shap  = top10_shap - top10_nmf
only_nmf   = top10_nmf  - top10_shap

print(f"\n  상위 10개 변수 비교:")
print(f"  공통: {len(overlap)}개  {sorted(overlap)}")
print(f"  SHAP만: {len(only_shap)}개  {sorted(only_shap)}")
print(f"  NMF만:  {len(only_nmf)}개  {sorted(only_nmf)}")s

print(f"\n[Done]")