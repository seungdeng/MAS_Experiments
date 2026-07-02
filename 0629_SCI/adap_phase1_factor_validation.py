"""
ADAP Phase 1 — Factor 프로필 검증
===================================
목적: NMF Factor가 실제로 해석 가능한가?
      Factor별 대표 사용자들의 실제 고평점 영화 목록을 추출해서
      Loading 해석(F1=작품성 추구자, F2=가족/판타지, F3=오락)과 일치하는지 확인.

확인 사항:
  1. Factor별 대표 사용자(pure user) 추출
     - 각 Factor에 가장 강하게 속한 사용자 (W 벡터에서 해당 Factor 가중치 최상위)
  2. 대표 사용자들의 실제 고평점 영화 추출
  3. 영화 목록이 Factor Loading 해석과 일치하는지 정성 평가

필요 파일:
  - ml-32m/ratings.csv
  - ml-32m/movies.csv
  - adap_results_32m_B_deviation/user_feature_matrix.csv  ← NMF 결과
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import json, os, time

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DATA_DIR      = "ml-32m"
FEAT_CSV      = "adap_results_32m_B_deviation/user_feature_matrix.csv"
OUTPUT_DIR    = "adap_phase1_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

K             = 3          # 앞선 실험과 동일
RANDOM_STATE  = 42
TOP_USERS     = 200        # Factor별 대표 사용자 수 (늘려서 집합 지성 강화)
TOP_MOVIES    = 20         # 사용자별 고평점 영화 추출 수
MIN_RATING    = 4.0        # 고평점 기준
MIN_USER_VOTE = 3          # 영화가 집계되려면 최소 몇 명이 평가해야 하는가
MIN_USER_RATINGS = 100     # Pure user 후보 최소 평점 수 (cold user 제외)


# ─────────────────────────────────────────────
# 1. Feature Matrix 로드 + NMF 재학습
#    (W 행렬을 다시 얻기 위해)
# ─────────────────────────────────────────────
def load_and_refit_nmf(feat_csv: str, K: int):
    print("[1] Feature Matrix 로드...")
    feat_df = pd.read_csv(feat_csv, index_col=0)
    print(f"    shape: {feat_df.shape}")

    # NMF 입력 컬럼만 선택 (앞선 실험과 동일 기준)
    nmf_cols = [c for c in feat_df.columns
                if not c.startswith("NMF_EXCLUDE__")]
    nmf_input = feat_df[nmf_cols]
    print(f"    NMF 입력 변수: {len(nmf_cols)}개")

    # 앞선 실험과 동일한 스케일링
    X_scaled = StandardScaler().fit_transform(nmf_input.values)
    X_nn     = np.clip(X_scaled, 0, None)

    print(f"[2] NMF 재학습 (K={K})...")
    nmf = NMF(n_components=K, random_state=RANDOM_STATE,
              max_iter=500, init="nndsvda")
    W   = nmf.fit_transform(X_nn)   # (users, K)
    H   = nmf.components_           # (K, features)

    W_df = pd.DataFrame(W,
                        index=feat_df.index,
                        columns=[f"F{k+1}" for k in range(K)])

    print(f"    reconstruction_err: {nmf.reconstruction_err_:.4f}")
    return W_df, H, nmf_cols, feat_df


# ─────────────────────────────────────────────
# 2. Factor별 대표 사용자 추출
#    "Pure user": 해당 Factor 가중치가 압도적으로 높은 사용자
#    조건: 해당 Factor 가중치 / 전체 가중치 합 > 0.6 (60% 이상이 한 Factor에 집중)
# ─────────────────────────────────────────────
def extract_pure_users(W_df: pd.DataFrame, feat_df: pd.DataFrame,
                       K: int,
                       purity_threshold: float = 0.5,
                       top_n: int = TOP_USERS) -> dict:
    print(f"\n[3] Factor별 Pure User 추출 (purity>{purity_threshold}, "
          f"min_ratings>={MIN_USER_RATINGS})...")

    # cold user 제거: NMF_EXCLUDE__log_n_ratings 또는 전체 평점 수 기준
    # feat_df에 NMF_EXCLUDE__log_n_ratings가 있으면 활용, 없으면 skip
    log_col = "NMF_EXCLUDE__log_n_ratings"
    if log_col in feat_df.columns:
        # log1p(n) >= log1p(MIN_USER_RATINGS)
        min_log = np.log1p(MIN_USER_RATINGS)
        active_users = feat_df.index[feat_df[log_col] >= min_log]
        W_filtered = W_df.loc[active_users]
        print(f"    cold user 제거 후: {len(W_filtered):,}명 "
              f"(전체 {len(W_df):,}명 중)")
    else:
        W_filtered = W_df
        print(f"    log_n_ratings 컬럼 없음 — cold user 필터 미적용")

    W_norm = W_filtered.div(W_filtered.sum(axis=1) + 1e-9, axis=0)

    pure_users = {}
    for k in range(K):
        factor_col = f"F{k+1}"
        mask   = W_norm[factor_col] >= purity_threshold
        subset = W_norm[mask][factor_col].sort_values(ascending=False)
        top    = subset.head(top_n)
        pure_users[factor_col] = top.index.tolist()
        print(f"    F{k+1}: {mask.sum():,}명 중 상위 {len(top)}명 선택 "
              f"(purity range: {top.min():.3f}~{top.max():.3f}, "
              f"mean: {top.mean():.3f})")

    return pure_users


# ─────────────────────────────────────────────
# 3. ratings 로드 (chunk, 메모리 효율)
# ─────────────────────────────────────────────
def load_ratings_for_users(user_ids: set,
                            ratings_path: str,
                            chunk_size: int = 2_000_000) -> pd.DataFrame:
    print(f"\n[4] ratings 로드 (대상 사용자 {len(user_ids):,}명)...")
    chunks = []
    for chunk in pd.read_csv(
        ratings_path,
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
        usecols=["userId", "movieId", "rating"],
        chunksize=chunk_size
    ):
        sub = chunk[chunk["userId"].isin(user_ids)]
        if len(sub):
            chunks.append(sub)
    ratings = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    print(f"    로드 완료: {len(ratings):,}건")
    return ratings


# ─────────────────────────────────────────────
# 4. Factor별 대표 영화 추출
#    대표 사용자들이 공통으로 높게 평가한 영화
# ─────────────────────────────────────────────
def extract_top_movies(pure_users: dict,
                       ratings: pd.DataFrame,
                       movies: pd.DataFrame,
                       min_rating: float = MIN_RATING,
                       min_votes: int    = MIN_USER_VOTE,
                       top_n: int        = TOP_MOVIES) -> dict:
    print(f"\n[5] Factor별 대표 영화 추출...")
    results = {}

    for factor, user_ids in pure_users.items():
        # 해당 사용자들의 고평점 기록
        sub = ratings[
            ratings["userId"].isin(user_ids) &
            (ratings["rating"] >= min_rating)
        ]

        # 영화별 집계: 평균 평점 + 투표 수
        agg = sub.groupby("movieId")["rating"].agg(
            mean_rating="mean",
            vote_count="count"
        ).reset_index()

        # 최소 투표 수 필터 (소수 의견 제거)
        agg = agg[agg["vote_count"] >= min_votes]

        # 평균 평점 내림차순 정렬 후 상위 N개
        top = (agg.sort_values(["mean_rating", "vote_count"],
                               ascending=[False, False])
                  .head(top_n)
                  .merge(movies[["movieId", "title", "genres"]], on="movieId"))

        results[factor] = top
        print(f"    {factor}: {len(top)}편 추출 "
              f"(전체 {len(agg):,}편 중)")

    return results


# ─────────────────────────────────────────────
# 5. Loading 상위 변수 추출 (해석 근거)
# ─────────────────────────────────────────────
def get_loading_summary(H: np.ndarray, feature_names: list,
                        K: int, top_n: int = 8) -> dict:
    summary = {}
    for k in range(K):
        loading   = H[k]
        top_idx   = np.argsort(loading)[::-1][:top_n]
        summary[f"F{k+1}"] = [
            {"feature": feature_names[i],
             "loading": round(float(loading[i]), 3)}
            for i in top_idx
        ]
    return summary


# ─────────────────────────────────────────────
# 6. 결과 출력 + 저장
# ─────────────────────────────────────────────
def print_and_save(top_movies: dict, loading_summary: dict,
                   pure_users: dict, W_df: pd.DataFrame):

    factor_labels = {
        "F1": "작품성 추구자 (가설)",
        "F2": "가족/판타지 장르 (가설)",
        "F3": "오락/장르 영화 선호자 (가설)"
    }

    report = {}
    for factor in top_movies:
        print(f"\n{'='*60}")
        print(f"  {factor}  {factor_labels.get(factor, '')}")
        print(f"  Pure User 수: {len(pure_users[factor])}명")
        print(f"{'='*60}")

        print(f"\n  [Loading 상위 변수]")
        for item in loading_summary[factor]:
            bar  = "█" * int(item["loading"] / 2)
            print(f"    {bar:<20} {item['loading']:>7.3f}  {item['feature']}")

        print(f"\n  [대표 사용자들의 실제 고평점 영화 Top {TOP_MOVIES}]")
        df = top_movies[factor]
        for i, row in df.iterrows():
            print(f"    {row['mean_rating']:.2f}점 "
                  f"({int(row['vote_count'])}명) "
                  f"{row['title']}  [{row['genres']}]")

        report[factor] = {
            "label": factor_labels.get(factor, ""),
            "n_pure_users": len(pure_users[factor]),
            "loading_top": loading_summary[factor],
            "top_movies": df[["title", "genres",
                               "mean_rating", "vote_count"]].to_dict("records")
        }

    # JSON 저장
    with open(f"{OUTPUT_DIR}/phase1_factor_validation.json",
              "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # W 행렬 저장 (Phase 2에서 재사용)
    W_df.to_csv(f"{OUTPUT_DIR}/user_factor_weights.csv")

    print(f"\n[저장 완료]")
    print(f"  {OUTPUT_DIR}/phase1_factor_validation.json")
    print(f"  {OUTPUT_DIR}/user_factor_weights.csv")


# ─────────────────────────────────────────────
# 7. 검증 요약 — 일치도 정량화
#    Factor Loading의 상위 장르가 실제 영화 목록에 얼마나 등장하는지
# ─────────────────────────────────────────────
def compute_genre_alignment(top_movies: dict,
                             loading_summary: dict) -> None:
    print(f"\n{'='*60}")
    print("  Factor Loading vs 실제 영화 장르 일치도")
    print(f"{'='*60}")
    print("  ※ Loading 상위 장르가 실제 고평점 영화 장르에 얼마나 등장하는지")
    print("    숫자가 높을수록 Factor 해석이 실제 행동과 일치함\n")

    for factor, movies_df in top_movies.items():
        # Loading 상위 장르 추출 (genre_X_dev → X 부분만)
        top_genres = []
        for item in loading_summary[factor]:
            fname = item["feature"]
            if "_dev" in fname:
                genre = fname.replace("genre_", "").replace("_dev", "")
                top_genres.append(genre.lower())

        # 실제 영화 장르에서 등장 빈도
        genre_counter = {}
        total_movies  = len(movies_df)
        for _, row in movies_df.iterrows():
            for g in row["genres"].split("|"):
                genre_counter[g.lower()] = genre_counter.get(g.lower(), 0) + 1

        print(f"  {factor} — Loading 상위 장르 vs 실제 고평점 영화 장르 비율")
        for g in top_genres[:5]:
            count = genre_counter.get(g, 0)
            ratio = count / total_movies if total_movies else 0
            bar   = "█" * int(ratio * 20)
            print(f"    {g:<20} {bar:<20} {ratio:>6.1%} ({count}/{total_movies}편)")
        print()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()

    # 1. NMF 재학습
    W_df, H, nmf_cols, feat_df = load_and_refit_nmf(FEAT_CSV, K)
    loading_summary = get_loading_summary(H, nmf_cols, K)

    # 2. Pure User 추출
    pure_users = extract_pure_users(W_df, feat_df, K,
                                    purity_threshold=0.5,
                                    top_n=TOP_USERS)

    # 3. 전체 pure user ID 수집
    all_pure_ids = set()
    for ids in pure_users.values():
        all_pure_ids.update(ids)

    # 4. ratings 로드
    ratings = load_ratings_for_users(
        all_pure_ids,
        f"{DATA_DIR}/ratings.csv"
    )

    # 5. movies 로드
    movies = pd.read_csv(f"{DATA_DIR}/movies.csv",
                         dtype={"movieId": "int32"})

    # 6. Factor별 대표 영화 추출
    top_movies = extract_top_movies(pure_users, ratings, movies)

    # 7. 결과 출력 + 저장
    print_and_save(top_movies, loading_summary, pure_users, W_df)

    # 8. 장르 일치도 검증
    compute_genre_alignment(top_movies, loading_summary)

    print(f"\n[Done] 총 소요: {time.time()-t0:.1f}s")
    print()
    print("다음 단계 판단 기준:")
    print("  ✅ Factor별 영화 목록이 Loading 해석과 일치 → Phase 2 진행")
    print("     (F1 목록에 쉰들러 리스트·필름누아르 등 작품성 영화가 주를 이루는가)")
    print("  ❌ 일치하지 않음 → Feature Engineering 재검토 or K 재탐색")