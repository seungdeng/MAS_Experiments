import numpy as np
import pandas as pd
from tqdm import tqdm

from config import N_EVAL_USERS, N_FACTORS, N_NEGATIVES, RANDOM_SEED, TOP_K
from data import build_rating_matrix, leave_one_out_split, load_movielens
from evaluate import hit_at_k
from mf import fit_svd, top_items_per_factor, user_residual_summary
from profiling import profile_axis, profile_raw
from recommend import rank_candidates


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    ratings, movies = load_movielens()
    train, test = leave_one_out_split(ratings, RANDOM_SEED)

    user_ids = sorted(ratings.userId.unique())
    item_ids = sorted(ratings.movieId.unique())
    user_idx = {u: i for i, u in enumerate(user_ids)}
    item_idx = {m: i for i, m in enumerate(item_ids)}
    item_idx_inv = {i: m for m, i in item_idx.items()}

    R = build_rating_matrix(train, len(user_ids), len(item_ids), user_idx, item_idx)
    U, S, Vt, user_means, residual = fit_svd(R, N_FACTORS)

    title_map = dict(zip(movies.movieId, movies.title))
    genre_map = dict(zip(movies.movieId, movies.genres))

    factor_summaries = top_items_per_factor(Vt, item_idx_inv, n=8)
    factor_summaries = [
        [(title_map.get(mid, str(mid)), w) for mid, w in fs] for fs in factor_summaries
    ]

    eval_users = rng.choice(
        test.userId.values, size=min(N_EVAL_USERS, len(test)), replace=False
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

        residual_items = user_residual_summary(residual, u_row, item_ids, movies, n=5)

        p_raw = profile_raw(history)
        p_axis = profile_axis(history, factor_summaries, residual_items)

        ranked_raw = rank_candidates(p_raw, candidates)
        ranked_axis = rank_candidates(p_axis, candidates)

        results.append(
            {
                "userId": uid,
                "profile_raw": p_raw,
                "profile_axis": p_axis,
                "hit_raw": hit_at_k(ranked_raw, target_id, TOP_K),
                "hit_axis": hit_at_k(ranked_axis, target_id, TOP_K),
            }
        )

    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)
    print(df[["hit_raw", "hit_axis"]].mean())


if __name__ == "__main__":
    main()
