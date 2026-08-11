import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    FACTOR_METHODS,
    MAX_ENGINEERED_AXES,
    MIN_AXIS_GAIN,
    N_EVAL_USERS,
    N_FACTORS,
    N_NEGATIVES,
    RANDOM_SEED,
    TOP_K,
)
from data import build_item_genre_matrix, build_rating_matrix, leave_one_out_split, load_movielens
from evaluate import hit_at_k, mrr, ndcg_at_k
from mf import engineer_axes, fit_factorization, top_items_per_factor, user_residual_summary
from profiling import profile_axis, profile_raw
from recommend import rank_candidates


def main(n_factors: int = N_FACTORS, n_eval_users: int = N_EVAL_USERS, out_path: str | None = None):
    rng = np.random.default_rng(RANDOM_SEED)
    ratings, movies = load_movielens()
    train, test = leave_one_out_split(ratings, RANDOM_SEED)

    user_ids = sorted(ratings.userId.unique())
    item_ids = sorted(ratings.movieId.unique())
    user_idx = {u: i for i, u in enumerate(user_ids)}
    item_idx = {m: i for i, m in enumerate(item_ids)}
    item_idx_inv = {i: m for m, i in item_idx.items()}

    R = build_rating_matrix(train, len(user_ids), len(item_ids), user_idx, item_idx)
    mask = R != 0

    title_map = dict(zip(movies.movieId, movies.title))
    genre_map = dict(zip(movies.movieId, movies.genres))
    item_genre_matrix, genre_names = build_item_genre_matrix(movies, item_ids)

    genre_idx_inv = {i: g for i, g in enumerate(genre_names)}

    # 방법별로 축/잔차를 미리 계산
    factorizations = {}
    for method in FACTOR_METHODS:
        if method == "genre":
            U, H, residual, H_genre = fit_factorization(
                R, n_factors, method=method, item_genre_matrix=item_genre_matrix
            )
            # 압축된 축이 어떤 장르 조합인지: "Action(+)+Drama(-)" 같은 라벨로 표시
            genre_summaries = top_items_per_factor(H_genre, genre_idx_inv, n=3)
            axis_labels = [
                "+".join(f"{g}({'+' if w >= 0 else '-'})" for g, w in gs) for gs in genre_summaries
            ]
        else:
            U, H, residual = fit_factorization(R, n_factors, method=method)
            axis_labels = None
        U, H, residual, engineered_desc, axis_labels = engineer_axes(
            U,
            H,
            residual,
            mask,
            max_rounds=MAX_ENGINEERED_AXES,
            min_relative_gain=MIN_AXIS_GAIN,
            axis_labels=axis_labels,
        )
        print(f"[{method}] engineered axes: {engineered_desc}")
        summaries = top_items_per_factor(H, item_idx_inv, n=8)
        summaries = [[(title_map.get(mid, str(mid)), w) for mid, w in fs] for fs in summaries]
        factorizations[method] = {
            "residual": residual,
            "factor_summaries": summaries,
            "axis_labels": axis_labels,
        }

    eval_users = rng.choice(
        test.userId.values, size=min(n_eval_users, len(test)), replace=False
    )

    results = []
    for uid in tqdm(eval_users):
        u_row = user_idx[uid]
        u_train = train[train.userId == uid]
        history = [
            (title_map.get(mid, str(mid)), genre_map.get(mid, ""), r)
            for mid, r in zip(u_train.movieId, u_train.rating)
        ]
        if not history:
            continue

        target_row = test[test.userId == uid].iloc[0]
        target_id = int(target_row.movieId)

        rated_ids = set(u_train.movieId) | {target_id}
        neg_pool = [m for m in item_ids if m not in rated_ids]
        negatives = rng.choice(neg_pool, size=min(N_NEGATIVES, len(neg_pool)), replace=False)

        candidates = [(target_id, title_map.get(target_id, str(target_id)))]
        candidates += [(m, title_map.get(m, str(m))) for m in negatives]
        rng.shuffle(candidates)

        row = {"userId": uid}

        p_raw = profile_raw(history)
        ranked_raw = rank_candidates(p_raw, candidates)
        row["profile_raw"] = p_raw
        row["hit_raw"] = hit_at_k(ranked_raw, target_id, TOP_K)
        row["mrr_raw"] = mrr(ranked_raw, target_id)
        row["ndcg_raw"] = ndcg_at_k(ranked_raw, target_id, TOP_K)

        for method in FACTOR_METHODS:
            residual = factorizations[method]["residual"]
            summaries = factorizations[method]["factor_summaries"]
            axis_labels = factorizations[method]["axis_labels"]
            far_items = user_residual_summary(residual, u_row, item_ids, movies, mask, n=5, mode="far")
            close_items = user_residual_summary(residual, u_row, item_ids, movies, mask, n=5, mode="close")

            p_axis = profile_axis(
                history, summaries, far_items, close_items, method=method, axis_labels=axis_labels
            )
            ranked_axis = rank_candidates(p_axis, candidates)

            row[f"profile_{method}"] = p_axis
            row[f"hit_{method}"] = hit_at_k(ranked_axis, target_id, TOP_K)
            row[f"mrr_{method}"] = mrr(ranked_axis, target_id)
            row[f"ndcg_{method}"] = ndcg_at_k(ranked_axis, target_id, TOP_K)

        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(out_path or "results.csv", index=False)

    methods = ["raw"] + FACTOR_METHODS
    metric_cols = [f"{metric}_{m}" for m in methods for metric in ("hit", "mrr", "ndcg")]
    print(f"[n_factors={n_factors} n_eval_users={n_eval_users}]")
    print(df[metric_cols].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-factors", type=int, default=N_FACTORS)
    parser.add_argument("--n-eval-users", type=int, default=N_EVAL_USERS)
    parser.add_argument("--out", type=str, default=None, help="output CSV path (default: results.csv)")
    args = parser.parse_args()
    main(args.n_factors, args.n_eval_users, args.out)
