"""
ADAP Phase 1 v2 — Factor 프로필 검증 (담백한 버전)
===================================================
NMF K=3 Factor별 대표 사용자의 실제 고평점 영화를 확인한다.

실행:
    python adap_phase1_v2.py

필요 파일:
    ml-32m/ratings.csv
    ml-32m/movies.csv
    adap_results_32m_B_deviation/user_feature_matrix.csv
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import os, time

DATA_DIR = "ml-32m"
FEAT_CSV = "adap_results_32m_B_deviation/user_feature_matrix.csv"
K        = 3
RANDOM_STATE = 42

# ─── 1. Feature Matrix 로드 + NMF 재학습 ───────────────────────────
print("Feature Matrix 로드...")
feat_df = pd.read_csv(FEAT_CSV, index_col=0)

nmf_cols   = [c for c in feat_df.columns if not c.startswith("NMF_EXCLUDE__")]
X_scaled   = StandardScaler().fit_transform(feat_df[nmf_cols].values)
X_nn       = np.clip(X_scaled, 0, None)

print(f"NMF 학습 (K={K})...")
nmf = NMF(n_components=K, random_state=RANDOM_STATE, max_iter=500, init="nndsvda")
W   = nmf.fit_transform(X_nn)   # (n_users, K)
H   = nmf.components_           # (K, n_features)

# W를 DataFrame으로
W_df = pd.DataFrame(W, index=feat_df.index,
                    columns=[f"F{k+1}" for k in range(K)])

# ─── 2. Factor별 대표 사용자 추출 ────────────────────────────────────
# 각 Factor에서 W값이 가장 높은 사용자 상위 N명
# 제약 없음 — 단순히 해당 Factor 값이 가장 높은 사람들
N_USERS = 300

dominant = W_df.idxmax(axis=1)   # 각 사용자의 주요 Factor
rep_users = {}
for k in range(K):
    col   = f"F{k+1}"
    # 이 Factor가 dominant인 사용자 중 W값 상위 N명
    mask  = dominant == col
    top   = W_df.loc[mask, col].nlargest(N_USERS)
    rep_users[col] = top.index.tolist()
    print(f"F{k+1} 대표 사용자: {len(top)}명 (전체 {mask.sum():,}명 중 상위)")

# ─── 3. ratings 로드 ─────────────────────────────────────────────────
all_users = set()
for ids in rep_users.values():
    all_users.update(ids)

print(f"\nratings 로드 (대상 {len(all_users):,}명)...")
chunks = []
for chunk in pd.read_csv(
    f"{DATA_DIR}/ratings.csv",
    usecols=["userId", "movieId", "rating"],
    dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    chunksize=2_000_000
):
    sub = chunk[chunk["userId"].isin(all_users)]
    if len(sub):
        chunks.append(sub)
ratings = pd.concat(chunks, ignore_index=True)
print(f"로드 완료: {len(ratings):,}건")

movies = pd.read_csv(f"{DATA_DIR}/movies.csv", dtype={"movieId": "int32"})

# ─── 4. Factor별 대표 영화 추출 ─────────────────────────────────────
# 대표 사용자들 중 4점 이상 평가한 영화를 집계
# 최소 10명이 평가한 영화만 (개인 취향 제거)
MIN_RATING = 4.0
MIN_VOTES  = 10
TOP_N      = 30

print()
for k in range(K):
    col      = f"F{k+1}"
    user_ids = rep_users[col]

    sub = ratings[
        ratings["userId"].isin(user_ids) &
        (ratings["rating"] >= MIN_RATING)
    ]

    agg = (sub.groupby("movieId")["rating"]
              .agg(mean_rating="mean", vote_count="count")
              .reset_index()
              .query(f"vote_count >= {MIN_VOTES}")
              .sort_values(["mean_rating", "vote_count"], ascending=[False, False])
              .head(TOP_N)
              .merge(movies, on="movieId"))

    # Loading 상위 변수
    loading = H[k]
    top_idx = np.argsort(loading)[::-1][:6]

    print(f"\n{'='*60}")
    print(f"  F{k+1}  (대표 사용자 {len(user_ids)}명)")
    print(f"{'='*60}")
    print(f"  Loading 상위 변수:")
    for i in top_idx:
        bar = "█" * int(loading[i] / 2)
        print(f"    {bar:<12} {loading[i]:>6.1f}  {nmf_cols[i]}")

    print(f"\n  고평점 영화 Top {len(agg)}편 (4점이상, 최소{MIN_VOTES}명 평가):")
    for _, row in agg.iterrows():
        print(f"    {row['mean_rating']:.2f}점 "
              f"({int(row['vote_count'])}명) "
              f"{row['title']}  [{row['genres']}]")

# ─── 5. W 분포 확인 ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  W 행렬 분포 (사용자별 Factor 가중치)")
print(f"{'='*60}")
W_norm = W_df.div(W_df.sum(axis=1) + 1e-9, axis=0)
print(f"\n  dominant Factor 분포:")
dominant_counts = W_df.idxmax(axis=1).value_counts()
for col, cnt in dominant_counts.items():
    print(f"    {col}: {cnt:,}명 ({cnt/len(W_df)*100:.1f}%)")

print(f"\n  정규화된 W 기술통계:")
print(W_norm.describe().round(3).to_string())

W_df.to_csv("adap_phase1_results/user_factor_weights_v2.csv")
os.makedirs("adap_phase1_results", exist_ok=True)
W_df.to_csv("adap_phase1_results/user_factor_weights_v2.csv")
print("\n저장: adap_phase1_results/user_factor_weights_v2.csv")