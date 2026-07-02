"""
ADAP Phase 2 — Behavior Fidelity 평가
======================================
Factor → LLM 에이전트 프롬프트 변환 후,
사용자의 W 가중치로 앙상블하여 실제 평점 예측 정확도를 측정한다.

흐름:
  1. NMF 재학습 → W(사용자 가중치), H(Factor Loading) 획득
  2. Factor별 few-shot 프롬프트 구성 (Phase 1 대표 영화 활용)
  3. Hold-out 평점 셋 구성 (사용자별 최신 N건)
  4. 각 에이전트에게 평점 예측 요청 → 가중합 → 최종 예측
  5. RMSE, MAE 측정 및 베이스라인 비교

베이스라인:
  B1. 사용자 평균 평점
  B2. 아이템 평균 평점
  B3. 가중치 최대 Factor 단일 에이전트 (hard assignment)

실행:
  export ANTHROPIC_API_KEY=sk-ant-...
  python adap_phase2_fidelity.py

필요 파일:
  ml-32m/ratings.csv
  ml-32m/movies.csv
  adap_results_32m_B_deviation/user_feature_matrix.csv
  adap_phase1_results/phase1_factor_validation.json
"""

import os, json, time, re, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 OPENAI_API_KEY 로드

# ─────────────────────────────────────────────
# 0. 설정
# ─────────────────────────────────────────────
DATA_DIR    = "ml-32m"
FEAT_CSV    = "adap_results_32m_B_deviation/user_feature_matrix.csv"
PHASE1_JSON = "adap_phase1_results/phase1_factor_validation.json"
OUT_DIR     = "adap_phase2_results"
os.makedirs(OUT_DIR, exist_ok=True)

K             = 3
RANDOM_STATE  = 42
N_EVAL_USERS  = 50       # 평가할 사용자 수 (API 비용 고려)
N_HOLDOUT     = 5        # 사용자당 hold-out 평점 수
TOP_MOVIES_FS = 10       # few-shot 예시 영화 수
MODEL         = "gpt-4o-mini"   # 비용 효율적. gpt-4o로 교체 가능
MAX_TOKENS    = 50
CHUNK_SIZE    = 2_000_000

FACTOR_NAMES = {
    "F1": "작품성 추구자",
    "F2": "가족/판타지 장르 선호자",
    "F3": "오락/액션 장르 선호자"
}


# ─────────────────────────────────────────────
# 1. NMF 재학습 → W, H
# ─────────────────────────────────────────────
def load_nmf(feat_csv: str, K: int):
    feat_df  = pd.read_csv(feat_csv, index_col=0)
    nmf_cols = [c for c in feat_df.columns if not c.startswith("NMF_EXCLUDE__")]
    X        = np.clip(StandardScaler().fit_transform(feat_df[nmf_cols].values), 0, None)
    nmf      = NMF(n_components=K, random_state=RANDOM_STATE,
                   max_iter=500, init="nndsvda")
    W        = nmf.fit_transform(X)
    W_df     = pd.DataFrame(W, index=feat_df.index,
                            columns=[f"F{k+1}" for k in range(K)])
    return W_df, nmf.components_, nmf_cols


# ─────────────────────────────────────────────
# 2. Few-shot 프롬프트 구성
#    Phase 1 결과에서 Factor별 대표 영화 로드
# ─────────────────────────────────────────────
def build_agent_prompts(phase1_json: str, top_n: int = TOP_MOVIES_FS) -> dict:
    """
    Factor별 few-shot 프롬프트 생성.
    시스템 프롬프트: 이 에이전트의 취향 설명
    """
    with open(phase1_json, encoding="utf-8") as f:
        phase1 = json.load(f)

    prompts = {}
    for factor, data in phase1.items():
        movies = data["top_movies"][:top_n]
        movie_list = "\n".join(
            f"  - {m['title']} ({m['mean_rating']:.1f}점)"
            for m in movies
        )
        prompts[factor] = f"""당신은 다음과 같은 영화 취향을 가진 사용자입니다.

[선호 영화 예시 - 이 사용자가 높게 평가한 영화들]
{movie_list}

위 취향을 바탕으로, 질문받은 영화에 이 사용자가 줄 평점을 예측하세요.
- 평점 범위: 0.5 ~ 5.0 (0.5 단위)
- 반드시 숫자 하나만 답하세요. 설명 금지.
- 예: 3.5"""

    return prompts


# ─────────────────────────────────────────────
# 3. Hold-out 셋 구성
#    사용자별 최신 N건을 테스트로, 나머지를 학습으로
# ─────────────────────────────────────────────
def build_holdout(ratings_path: str, user_ids: list,
                  n_holdout: int = N_HOLDOUT) -> tuple:
    """
    반환: (holdout_df, movie_ids_needed)
    holdout_df: userId, movieId, rating 컬럼
    """
    print(f"  Hold-out 구성 ({len(user_ids)}명 × {n_holdout}건)...")
    user_set = set(user_ids)
    chunks   = []
    for chunk in pd.read_csv(
        ratings_path,
        usecols=["userId", "movieId", "rating", "timestamp"],
        dtype={"userId": "int32", "movieId": "int32",
               "rating": "float32", "timestamp": "int64"},
        chunksize=CHUNK_SIZE
    ):
        sub = chunk[chunk["userId"].isin(user_set)]
        if len(sub):
            chunks.append(sub)

    all_ratings = pd.concat(chunks, ignore_index=True)

    # 사용자별 최신 n_holdout건 추출
    holdout = (all_ratings
               .sort_values("timestamp", ascending=False)
               .groupby("userId")
               .head(n_holdout)
               .reset_index(drop=True))

    movie_ids = holdout["movieId"].unique().tolist()
    print(f"    hold-out: {len(holdout)}건, 고유 영화: {len(movie_ids)}편")
    return holdout, movie_ids, all_ratings


# ─────────────────────────────────────────────
# 4. LLM 평점 예측
# ─────────────────────────────────────────────
def predict_rating_llm(client: OpenAI,
                        system_prompt: str,
                        movie_title: str,
                        movie_genres: str) -> float | None:
    """단일 에이전트의 단일 영화 평점 예측."""
    user_msg = f"영화: {movie_title} (장르: {movie_genres})\n평점:"
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg}
            ],
            temperature=0.0   # 재현 가능성을 위해 고정
        )
        text = resp.choices[0].message.content.strip()
        match = re.search(r"\d+\.?\d*", text)
        if match:
            rating = float(match.group())
            return float(np.clip(rating, 0.5, 5.0))
        return None
    except Exception as e:
        print(f"    API 오류: {e}")
        return None


# ─────────────────────────────────────────────
# 5. 가중합 앙상블
# ─────────────────────────────────────────────
def ensemble_predict(agent_preds: dict, weights: dict) -> float:
    """
    agent_preds: {"F1": 4.0, "F2": 3.5, "F3": 4.5}
    weights:     {"F1": 0.6, "F2": 0.1, "F3": 0.3}
    """
    total_w, total_wv = 0.0, 0.0
    for factor, pred in agent_preds.items():
        if pred is not None:
            w         = weights.get(factor, 0.0)
            total_wv += w * pred
            total_w  += w
    return total_wv / total_w if total_w > 0 else 3.0


# ─────────────────────────────────────────────
# 6. 베이스라인 계산
# ─────────────────────────────────────────────
def compute_baselines(holdout: pd.DataFrame,
                      all_ratings: pd.DataFrame) -> pd.DataFrame:
    """
    B1: 사용자 평균 평점
    B2: 아이템(영화) 평균 평점
    """
    # B1: 사용자 평균 (hold-out 제외한 나머지에서 계산)
    holdout_idx  = holdout.set_index(["userId", "movieId"]).index
    train        = all_ratings[
        ~all_ratings.set_index(["userId", "movieId"]).index.isin(holdout_idx)
    ]
    user_mean    = train.groupby("userId")["rating"].mean()
    item_mean    = train.groupby("movieId")["rating"].mean()
    global_mean  = train["rating"].mean()

    result = holdout.copy()
    result["b1_user_mean"] = result["userId"].map(user_mean).fillna(global_mean)
    result["b2_item_mean"] = result["movieId"].map(item_mean).fillna(global_mean)
    return result


# ─────────────────────────────────────────────
# 7. 메인 평가 루프
# ─────────────────────────────────────────────
def run_evaluation(W_df, agent_prompts, holdout, movies_df,
                   all_ratings) -> pd.DataFrame:
    client  = OpenAI()   # OPENAI_API_KEY는 .env에서 자동 로드
    results = []

    # 영화 정보 조회용 dict
    movie_info = movies_df.set_index("movieId")[["title", "genres"]].to_dict("index")

    # 베이스라인 계산
    holdout_bl = compute_baselines(holdout, all_ratings)

    users = holdout["userId"].unique()
    total = len(holdout)
    done  = 0

    print(f"\n  평가 시작: {len(users)}명 × 최대 {N_HOLDOUT}건")
    print(f"  API 호출 예상: {len(users) * N_HOLDOUT * K}건\n")

    for uid in users:
        user_holdout = holdout_bl[holdout_bl["userId"] == uid]

        # 사용자 W 가중치
        if uid not in W_df.index:
            continue
        w_row    = W_df.loc[uid]
        w_norm   = w_row / (w_row.sum() + 1e-9)
        weights  = {f"F{k+1}": float(w_norm[f"F{k+1}"]) for k in range(K)}
        dominant = w_norm.idxmax()

        for _, row in user_holdout.iterrows():
            mid   = row["movieId"]
            true_r = row["rating"]

            if mid not in movie_info:
                continue
            title  = movie_info[mid]["title"]
            genres = movie_info[mid]["genres"]

            # 각 에이전트에게 예측 요청
            agent_preds = {}
            for factor, sys_prompt in agent_prompts.items():
                pred = predict_rating_llm(client, sys_prompt, title, genres)
                agent_preds[factor] = pred
                time.sleep(0.3)   # rate limit 방지

            # 앙상블 예측
            pred_ensemble = ensemble_predict(agent_preds, weights)

            # hard assignment (dominant Factor만 사용)
            pred_hard = agent_preds.get(dominant)

            results.append({
                "userId":         uid,
                "movieId":        mid,
                "title":          title,
                "true_rating":    true_r,
                "pred_ensemble":  pred_ensemble,
                "pred_hard":      pred_hard,
                "pred_b1":        row["b1_user_mean"],
                "pred_b2":        row["b2_item_mean"],
                "weight_F1":      weights["F1"],
                "weight_F2":      weights["F2"],
                "weight_F3":      weights["F3"],
                "dominant":       dominant,
                **{f"agent_{f}": v for f, v in agent_preds.items()}
            })

            done += 1
            if done % 10 == 0:
                print(f"    진행: {done}/{total}  "
                      f"(user={uid}, movie={title[:30]})")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# 8. 결과 집계
# ─────────────────────────────────────────────
def compute_metrics(results_df: pd.DataFrame) -> dict:
    def rmse(true, pred):
        return float(np.sqrt(np.mean((true - pred) ** 2)))

    true = results_df["true_rating"].values
    metrics = {}

    for col, label in [
        ("pred_ensemble", "ADAP Ensemble"),
        ("pred_hard",     "Hard Assignment"),
        ("pred_b1",       "Baseline: User Mean"),
        ("pred_b2",       "Baseline: Item Mean"),
    ]:
        valid = results_df[col].notna()
        if valid.sum() == 0:
            continue
        t, p = true[valid], results_df[col][valid].values
        metrics[label] = {
            "RMSE": round(rmse(t, p), 4),
            "MAE":  round(float(mean_absolute_error(t, p)), 4),
            "N":    int(valid.sum())
        }

    return metrics


def print_metrics(metrics: dict):
    print(f"\n{'='*55}")
    print(f"  Behavior Fidelity 평가 결과")
    print(f"{'='*55}")
    print(f"{'Method':<25} {'RMSE':>8}  {'MAE':>8}  {'N':>6}")
    print("-" * 55)
    for label, m in metrics.items():
        marker = " ★" if "ADAP" in label else ""
        print(f"{label:<25} {m['RMSE']:>8.4f}  {m['MAE']:>8.4f}  "
              f"{m['N']:>6}{marker}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    print("=" * 55)
    print("  ADAP Phase 2 — Behavior Fidelity 평가")
    print("=" * 55)

    # 1. NMF
    print("\n[1] NMF 로드...")
    W_df, H, nmf_cols = load_nmf(FEAT_CSV, K)
    print(f"    W: {W_df.shape}")

    # 2. 프롬프트 구성
    print("\n[2] 에이전트 프롬프트 구성...")
    agent_prompts = build_agent_prompts(PHASE1_JSON, top_n=TOP_MOVIES_FS)
    for factor, prompt in agent_prompts.items():
        print(f"\n  [{factor}] {FACTOR_NAMES[factor]}")
        print("  " + "-" * 40)
        print("  " + prompt[:300].replace("\n", "\n  ") + "...")

    # 3. 평가 사용자 선정
    #    W가 고르게 분포한 사용자 선택 (혼합 가중치를 가진 사용자)
    print(f"\n[3] 평가 사용자 {N_EVAL_USERS}명 선정...")
    W_norm = W_df.div(W_df.sum(axis=1) + 1e-9, axis=0)
    # 혼합 가중치 사용자: max purity < 0.9 (한 Factor에 쏠리지 않은 사용자)
    mixed_users = W_norm[W_norm.max(axis=1) < 0.9].index.tolist()
    np.random.seed(RANDOM_STATE)
    eval_users  = np.random.choice(
        mixed_users, size=min(N_EVAL_USERS, len(mixed_users)), replace=False
    ).tolist()
    print(f"    혼합 가중치 사용자 풀: {len(mixed_users):,}명 → {len(eval_users)}명 선정")

    # 4. Hold-out 구성
    print("\n[4] Hold-out 구성...")
    holdout, movie_ids, all_ratings = build_holdout(
        f"{DATA_DIR}/ratings.csv", eval_users
    )

    # 5. 영화 정보 로드
    movies_df = pd.read_csv(f"{DATA_DIR}/movies.csv",
                            dtype={"movieId": "int32"})

    # 6. 평가 실행
    print("\n[5] LLM 평점 예측 및 평가...")
    results_df = run_evaluation(W_df, agent_prompts, holdout,
                                movies_df, all_ratings)

    # 7. 결과
    metrics = compute_metrics(results_df)
    print_metrics(metrics)

    # 저장
    results_df.to_csv(f"{OUT_DIR}/prediction_results.csv", index=False)
    with open(f"{OUT_DIR}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n[저장]")
    print(f"  {OUT_DIR}/prediction_results.csv")
    print(f"  {OUT_DIR}/metrics.json")
    print(f"\n[Done]  총 소요: {time.time()-t0:.1f}s")