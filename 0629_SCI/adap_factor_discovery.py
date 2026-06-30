"""
ADAP Factor Discovery Experiment — ml-32m
==========================================
목적: MovieLens 32M 데이터에서 NMF / Factor Analysis / PCA로
      잠재 행동 Factor를 발견하고 해석 가능성을 비교한다.

실험 구성:
  Step 1. Feature Engineering  : Raw 트랜잭션 → User × Feature Matrix
  Step 2. Factor Discovery      : NMF / FA / PCA × K={3,5,10}
  Step 3. Factor 품질 평가      : Loading 해석, Factor 분리도, Silhouette
  Step 4. Data Volume Ablation  : 10→30→60→100% 사용자로 Factor 안정성 측정

실행 방법:
  1. DATA_DIR을 ml-32m 폴더 경로로 수정
  2. python adap_factor_discovery_ml32m.py
  결과는 OUTPUT_DIR(adap_results_32m/)에 저장됨

데이터 요구사항:
  ratings.csv  : userId, movieId, rating, timestamp
  movies.csv   : movieId, title, genres
  (links.csv, tags.csv는 미사용)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF, FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import os, json, time

# ─────────────────────────────────────────────
# 0. 설정  ← 여기만 수정
# ─────────────────────────────────────────────
DATA_DIR   = "ml-32m"          # ratings.csv, movies.csv가 있는 폴더
OUTPUT_DIR = "adap_results_32m"

RANDOM_STATE = 42
K_LIST       = [3, 5, 10]                    # 탐색할 Factor 수
VOLUME_FRACS = [0.10, 0.30, 0.60, 1.00]     # Data Volume Ablation 비율
CHUNK_SIZE   = 2_000_000                     # 32M 행 청크 로딩 단위
MIN_RATINGS  = 20                            # 사용자 최소 평가 수 필터

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────
def load_data():
    print("[Load] movies.csv...")
    movies = pd.read_csv(
        f"{DATA_DIR}/movies.csv",
        dtype={"movieId": "int32"}
    )

    print("[Load] ratings.csv (chunked, 32M rows)...")
    t0 = time.time()
    chunks = []
    for i, chunk in enumerate(pd.read_csv(
        f"{DATA_DIR}/ratings.csv",
        dtype={"userId": "int32", "movieId": "int32",
               "rating": "float32"},
        usecols=["userId", "movieId", "rating", "timestamp"],
        chunksize=CHUNK_SIZE
    )):
        chunks.append(chunk)
        print(f"  chunk {i+1}: {len(chunk):,} rows  "
              f"(누적 {sum(len(c) for c in chunks):,})")
    ratings = pd.concat(chunks, ignore_index=True)
    print(f"  완료: {len(ratings):,} ratings  ({time.time()-t0:.1f}s)")

    # 최소 평가 수 필터 (이미 데이터셋 조건이지만 명시적으로 재확인)
    user_counts = ratings.groupby("userId")["movieId"].count()
    valid_users = user_counts[user_counts >= MIN_RATINGS].index
    ratings = ratings[ratings["userId"].isin(valid_users)]
    print(f"  필터 후: {len(ratings):,} ratings  "
          f"{ratings['userId'].nunique():,} users  "
          f"{ratings['movieId'].nunique():,} movies")

    return ratings, movies


# ─────────────────────────────────────────────
# 2. Feature Engineering
#    Raw 트랜잭션 → User × Feature Matrix
# ─────────────────────────────────────────────
def compute_popularity_tier(ratings_full: pd.DataFrame,
                             movies: pd.DataFrame) -> pd.DataFrame:
    """
    전체 ratings 기준으로 Popularity Tier를 한 번만 계산해 고정.
    Ablation 시 서브셋마다 Tier가 달라지는 문제를 방지.
    반환: movieId 컬럼과 pop_tier 컬럼을 가진 DataFrame
    """
    print("[Feature] Popularity Tier 계산 (전체 데이터 기준, 고정)...")
    movie_pop = ratings_full.groupby("movieId")["rating"].count().rename("pop_count")
    q90 = movie_pop.quantile(0.90)
    q50 = movie_pop.quantile(0.50)
    tier_df = movie_pop.reset_index()
    tier_df["pop_tier"] = pd.cut(
        tier_df["pop_count"],
        bins=[-1, q50, q90, float("inf")],
        labels=["low", "mid", "high"]
    )
    print(f"  q50={q50:.0f}  q90={q90:.0f}  "
          f"(low/mid/high: "
          f"{(tier_df['pop_tier']=='low').sum():,}/"
          f"{(tier_df['pop_tier']=='mid').sum():,}/"
          f"{(tier_df['pop_tier']=='high').sum():,} movies)")
    return tier_df[["movieId", "pop_tier"]]


def build_feature_matrix(ratings: pd.DataFrame,
                          movies:  pd.DataFrame,
                          tier_df: pd.DataFrame) -> pd.DataFrame:
    """
    집계 변수:
      - 장르별 평균 평점          (genre_{g}_mean)
      - 장르별 평가 비율          (genre_{g}_ratio)
      - 흥행 등급별 평균 평점     (pop_high/mid/low_mean)
        ※ Popularity Tier는 전체 데이터 기준으로 고정 (tier_df 인자로 전달)
      - 평점 분산                 (rating_std)
      - 평균 평점                 (rating_mean)
      - 총 평가 수 log            (log_n_ratings)
      - Popularity Bias 지수      (pop_high_mean - pop_low_mean)
    """
    print("[Feature] 장르 파싱...")
    # 장르 목록 추출
    all_genres = set()
    for g in movies["genres"].dropna():
        if g != "(no genres listed)":
            all_genres.update(g.split("|"))
    all_genres = sorted(all_genres)

    genre_cols = []
    movies = movies.copy()
    for g in all_genres:
        col = f"genre_{g}"
        movies[col] = movies["genres"].apply(
            lambda x: 1 if isinstance(x, str) and g in x.split("|") else 0
        )
        genre_cols.append(col)

    print(f"  장르 수: {len(all_genres)}")

    # Popularity Tier: 외부에서 고정된 tier_df 사용 (서브셋마다 재계산 안 함)
    movies = movies.merge(tier_df, on="movieId", how="left")

    print("[Feature] ratings ↔ movies 병합...")
    merged = ratings.merge(
        movies[["movieId", "pop_tier"] + genre_cols],
        on="movieId", how="left"
    )

    print("[Feature] 사용자별 집계 중 (시간 소요)...")
    agg_dict = {}

    # 장르별 집계
    for g, col in zip(all_genres, genre_cols):
        sub = merged[merged[col] == 1]
        grp = sub.groupby("userId")["rating"]
        agg_dict[f"genre_{g}_mean"]  = grp.mean()
        agg_dict[f"genre_{g}_ratio"] = (
            grp.count() / merged.groupby("userId")["movieId"].count()
        )

    # Popularity Tier별 집계
    for tier in ["high", "mid", "low"]:
        sub = merged[merged["pop_tier"] == tier]
        agg_dict[f"pop_{tier}_mean"] = sub.groupby("userId")["rating"].mean()

    # 기본 통계
    base = merged.groupby("userId")["rating"].agg(["mean", "std", "count"])
    agg_dict["rating_mean"]   = base["mean"]
    agg_dict["rating_std"]    = base["std"].fillna(0)
    agg_dict["log_n_ratings"] = np.log1p(base["count"])

    # Popularity Bias 지수
    agg_dict["pop_bias_idx"] = (
        agg_dict.get("pop_high_mean", pd.Series(dtype="float32"))
        .subtract(agg_dict.get("pop_low_mean", pd.Series(dtype="float32")),
                  fill_value=0)
    )

    feat_df = pd.DataFrame(agg_dict)
    feat_df = feat_df.fillna(feat_df.median())

    print(f"[Feature] 완료: shape={feat_df.shape}  (users × features)")
    return feat_df, all_genres


# ─────────────────────────────────────────────
# 3. Factor Discovery
# ─────────────────────────────────────────────
def run_factor_discovery(feat_df: pd.DataFrame, K: int) -> dict:
    results = {}

    # NMF: non-negative 입력 필요
    X_nn = MinMaxScaler().fit_transform(feat_df.values)
    nmf  = NMF(n_components=K, random_state=RANDOM_STATE,
               max_iter=500, init="nndsvda")
    W_nmf = nmf.fit_transform(X_nn)
    results["NMF"] = {
        "W": W_nmf,
        "H": nmf.components_,
        "reconstruction_err": nmf.reconstruction_err_,
        "feature_names": feat_df.columns.tolist()
    }

    # FA / PCA: StandardScaler
    X_std = StandardScaler().fit_transform(feat_df.values)

    fa    = FactorAnalysis(n_components=K, random_state=RANDOM_STATE,
                           max_iter=1000)
    W_fa  = fa.fit_transform(X_std)
    results["FA"] = {
        "W": W_fa,
        "H": fa.components_,
        "feature_names": feat_df.columns.tolist()
    }

    pca   = PCA(n_components=K, random_state=RANDOM_STATE)
    W_pca = pca.fit_transform(X_std)
    results["PCA"] = {
        "W": W_pca,
        "H": pca.components_,
        "explained_var": pca.explained_variance_ratio_.tolist(),
        "feature_names": feat_df.columns.tolist()
    }

    return results


# ─────────────────────────────────────────────
# 4. Factor 품질 평가
#    (Ground Truth 없음 → 비지도 지표만 사용)
# ─────────────────────────────────────────────
def evaluate_factors(results: dict, K: int,
                     sample_size: int = 5000) -> dict:
    """
    E1. Factor 분리도  : Factor Loading 간 cosine similarity (↓ 좋음)
    E2. Silhouette     : KMeans 클러스터 품질 (↑ 좋음)
    E3. Top features   : 각 Factor 상위 변수 (해석 가능성, 정성)
    """
    metrics = {}
    for method, res in results.items():
        W = res["W"]
        H = res["H"]

        # E1. Factor 분리도
        H_norm  = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
        cos_sims = [
            abs(np.dot(H_norm[i], H_norm[j]))
            for i in range(K) for j in range(i+1, K)
        ]
        mean_cos = float(np.mean(cos_sims)) if cos_sims else 0.0

        # E2. Silhouette (대용량 → 샘플링)
        km   = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
        pred = km.fit_predict(W)
        sil  = float(silhouette_score(
            W, pred,
            sample_size=min(sample_size, len(W)),
            random_state=RANDOM_STATE
        ))

        # E3. Top-5 features per Factor
        top_features = {}
        for k in range(K):
            loading = H[k]
            top_idx = np.argsort(np.abs(loading))[::-1][:5]
            top_features[f"F{k+1}"] = [
                (res["feature_names"][i], round(float(loading[i]), 3))
                for i in top_idx
            ]

        metrics[method] = {
            "mean_factor_cosine_sim": round(mean_cos, 4),
            "silhouette_score":       round(sil, 4),
            "top_features_per_factor": top_features
        }

    return metrics


# ─────────────────────────────────────────────
# 5. Data Volume Ablation
# ─────────────────────────────────────────────
def data_volume_ablation(ratings:    pd.DataFrame,
                          movies:    pd.DataFrame,
                          tier_df:  pd.DataFrame,
                          K:         int = 3) -> dict:
    """
    사용자를 frac씩 샘플링 → Feature Matrix → NMF → Silhouette / Factor 분리도
    "데이터↑ → Factor 안정성↑" 가설 검증

    수정: Popularity Tier를 전체 데이터 기준으로 고정(tier_df)하고
    서브셋마다 재계산하지 않음 → Feature 정의 일관성 보장
    """
    all_users = ratings["userId"].unique()
    ablation  = {}

    for frac in VOLUME_FRACS:
        n_sub = max(int(len(all_users) * frac), 100)
        sampled_users = np.random.choice(all_users, n_sub, replace=False)
        rat_sub = ratings[ratings["userId"].isin(sampled_users)]

        # tier_df 고정값 전달 — 서브셋마다 Tier 재계산 없음
        feat_sub, _ = build_feature_matrix(rat_sub, movies, tier_df)

        X_nn = MinMaxScaler().fit_transform(feat_sub.values)
        nmf  = NMF(n_components=K, random_state=RANDOM_STATE,
                   max_iter=500, init="nndsvda")
        W    = nmf.fit_transform(X_nn)
        H    = nmf.components_

        km   = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
        pred = km.fit_predict(W)
        sil  = float(silhouette_score(
            W, pred,
            sample_size=min(5000, len(W)),
            random_state=RANDOM_STATE
        ))

        H_norm   = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
        cos_sims = [abs(np.dot(H_norm[i], H_norm[j]))
                    for i in range(K) for j in range(i+1, K)]

        ablation[frac] = {
            "n_users":              n_sub,
            "n_ratings":            len(rat_sub),
            "silhouette":           round(sil, 4),
            "mean_factor_cosine_sim": round(float(np.mean(cos_sims)), 4),
            "reconstruction_err":   round(float(nmf.reconstruction_err_), 4)
        }
        print(f"  [{frac*100:>5.0f}%]  users={n_sub:,}  "
              f"ratings={len(rat_sub):,}  "
              f"Sil={sil:.4f}  "
              f"FactorSep={np.mean(cos_sims):.4f}")

    return ablation


# ─────────────────────────────────────────────
# 6. 시각화
# ─────────────────────────────────────────────
def plot_loading_heatmap(results: dict, K: int, method: str,
                         top_n: int = 20):
    res        = results[method]
    H          = res["H"]
    feat_names = res["feature_names"]
    importance = np.max(np.abs(H), axis=0)
    top_idx    = np.argsort(importance)[::-1][:top_n]
    H_sub      = H[:, top_idx]
    feat_sub   = [feat_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(max(12, top_n * 0.55), K * 1.3 + 1))
    sns.heatmap(H_sub,
                xticklabels=feat_sub,
                yticklabels=[f"F{k+1}" for k in range(K)],
                cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.6})
    ax.set_title(f"{method}  Factor Loading Matrix  (K={K}, top-{top_n} features)",
                 fontsize=11, pad=10)
    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/loading_heatmap_{method}_K{K}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_method_comparison(all_metrics: dict):
    methods = ["NMF", "FA", "PCA"]
    colors  = {"NMF": "#2196F3", "FA": "#4CAF50", "PCA": "#FF9800"}
    K_vals  = sorted(all_metrics.keys())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, (metric, label) in zip(axes, [
        ("silhouette_score",       "Silhouette Score  (↑)"),
        ("mean_factor_cosine_sim", "Factor Cosine Sim  (↓ = 분리 잘 됨)"),
    ]):
        for method in methods:
            vals = [all_metrics[K][method][metric] for K in K_vals]
            ax.plot(K_vals, vals, marker="o", label=method,
                    color=colors[method], linewidth=2, markersize=7)
        ax.set_xlabel("K (Factor 수)", fontsize=10)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.set_xticks(K_vals)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Factor Discovery — Method × K Comparison  (ml-32m)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/method_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_ablation_curve(ablation: dict):
    fracs = sorted(ablation.keys())
    sils  = [ablation[f]["silhouette"] for f in fracs]
    seps  = [ablation[f]["mean_factor_cosine_sim"] for f in fracs]
    xs    = [int(f * 100) for f in fracs]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(xs, sils, "s-", color="#4CAF50", linewidth=2, markersize=8)
    axes[0].set_title("Silhouette Score  (↑)", fontsize=10)
    axes[0].set_xlabel("Data Volume (% users)")
    axes[0].set_xticks(xs)
    axes[0].grid(alpha=0.3)

    axes[1].plot(xs, seps, "^-", color="#FF9800", linewidth=2, markersize=8)
    axes[1].set_title("Factor Cosine Sim  (↓ = 분리 잘 됨)", fontsize=10)
    axes[1].set_xlabel("Data Volume (% users)")
    axes[1].set_xticks(xs)
    axes[1].grid(alpha=0.3)

    plt.suptitle("Data Volume Ablation  (NMF, K=3)\n"
                 "\"데이터↑ → Factor 안정성↑\" 가설 검증",
                 fontsize=11, y=1.03)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/ablation_volume.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_user_weight_dist(results: dict, K: int):
    W          = results["NMF"]["W"]
    km         = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
    pred       = km.fit_predict(W)
    colors_p   = plt.cm.tab10(np.linspace(0, 0.9, K))

    fig, ax = plt.subplots(figsize=(7, 6))
    for k in range(K):
        mask = pred == k
        ax.scatter(W[mask, 0], W[mask, 1 % K],
                   alpha=0.25, s=4, label=f"Cluster {k+1}",
                   color=colors_p[k])
    ax.set_xlabel("Factor 1 Score")
    ax.set_ylabel("Factor 2 Score")
    ax.set_title(f"User Weight Distribution  (NMF, K={K})\nml-32m",
                 fontsize=10)
    ax.legend(fontsize=8, markerscale=4)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/user_weight_dist_K{K}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ─────────────────────────────────────────────
# 7. 콘솔 리포트
# ─────────────────────────────────────────────
def print_report(all_metrics: dict, ablation: dict):
    print("\n" + "=" * 65)
    print("  ADAP Factor Discovery — 결과 요약  (ml-32m)")
    print("=" * 65)

    print("\n[1] 방법론 × K 비교")
    print(f"{'Method':<6} {'K':>3}  {'Sil':>7}  {'FactorSep':>10}")
    print("-" * 35)
    for K in sorted(all_metrics.keys()):
        for method in ["NMF", "FA", "PCA"]:
            m    = all_metrics[K][method]
            flag = " ★" if method == "NMF" else ""
            print(f"{method:<6} {K:>3}  "
                  f"{m['silhouette_score']:>7.4f}  "
                  f"{m['mean_factor_cosine_sim']:>10.4f}{flag}")

    print("\n[2] Data Volume Ablation  (NMF, K=3)")
    print(f"{'Volume':>8}  {'Users':>8}  {'Ratings':>12}  "
          f"{'Sil':>7}  {'FactorSep':>10}")
    print("-" * 58)
    for frac, r in sorted(ablation.items()):
        print(f"{frac*100:>7.0f}%  {r['n_users']:>8,}  "
              f"{r['n_ratings']:>12,}  "
              f"{r['silhouette']:>7.4f}  "
              f"{r['mean_factor_cosine_sim']:>10.4f}")

    print("\n[3] NMF K=3 — Factor별 상위 변수")
    tf = all_metrics[3]["NMF"]["top_features_per_factor"]
    for fid, feats in tf.items():
        print(f"\n  {fid}:")
        for fname, val in feats:
            bar  = "█" * int(abs(val) * 10)
            sign = "+" if val >= 0 else "-"
            print(f"    {sign}{bar:<12}  {val:>+6.3f}  {fname}")


# ─────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    t0 = time.time()

    print("=" * 55)
    print("  ADAP Factor Discovery  (ml-32m)")
    print("=" * 55)

    # Step 1. 로드
    ratings, movies = load_data()

    # Popularity Tier: 전체 데이터 기준으로 한 번만 계산하고 이후 재사용
    tier_df = compute_popularity_tier(ratings, movies)

    # Step 2. Feature Engineering
    print("\n[Step 1] Feature Engineering...")
    feat_df, all_genres = build_feature_matrix(ratings, movies, tier_df)
    feat_df.to_csv(f"{OUTPUT_DIR}/user_feature_matrix.csv")
    print(f"  저장: {OUTPUT_DIR}/user_feature_matrix.csv")

    # Step 3. Factor Discovery × K
    print("\n[Step 2] Factor Discovery (NMF / FA / PCA × K)...")
    all_metrics       = {}
    best_results_by_K = {}

    for K in K_LIST:
        print(f"\n  K={K}:")
        results = run_factor_discovery(feat_df, K)
        metrics = evaluate_factors(results, K)
        all_metrics[K]       = metrics
        best_results_by_K[K] = results

        plot_loading_heatmap(results, K, "NMF", top_n=20)

        for method in ["NMF", "FA", "PCA"]:
            m = metrics[method]
            print(f"    [{method}]  "
                  f"Sil={m['silhouette_score']:.4f}  "
                  f"FactorSep={m['mean_factor_cosine_sim']:.4f}")

    plot_method_comparison(all_metrics)
    plot_user_weight_dist(best_results_by_K[3], K=3)

    # Step 4. Data Volume Ablation — tier_df 고정값 전달
    print("\n[Step 3] Data Volume Ablation (NMF, K=3)...")
    ablation = data_volume_ablation(ratings, movies, tier_df, K=3)
    plot_ablation_curve(ablation)

    # 리포트 출력
    print_report(all_metrics, ablation)

    # JSON 저장
    summary = {
        "method_comparison": {
            str(K): {
                m: {k: v for k, v in metrics.items()
                    if k != "top_features_per_factor"}
                for m, metrics in mm.items()
            }
            for K, mm in all_metrics.items()
        },
        "ablation":          {str(k): v for k, v in ablation.items()},
        "top_features_K3_NMF": all_metrics[3]["NMF"]["top_features_per_factor"]
    }
    with open(f"{OUTPUT_DIR}/results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[Done]  총 소요: {time.time()-t0:.1f}s")
    print(f"결과 저장 위치: {OUTPUT_DIR}/")
    print("  loading_heatmap_NMF_K*.png  : Factor Loading Heatmap")
    print("  method_comparison.png        : NMF vs FA vs PCA")
    print("  ablation_volume.png          : Data Volume 커브")
    print("  user_weight_dist_K3.png      : User-Factor 공간")
    print("  user_feature_matrix.csv      : 사용자×피처 행렬")
    print("  results_summary.json         : 수치 전체")