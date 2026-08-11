import os

import numpy as np
import pandas as pd

from config import DATA_DIR


def load_movielens():
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    return ratings, movies


def leave_one_out_split(ratings: pd.DataFrame, seed: int = 42):
    """유저별 가장 최근 상호작용 1개를 test로 분리."""
    test_rows = ratings.sort_values("timestamp").groupby("userId").tail(1)
    train = ratings.drop(test_rows.index)
    return train.reset_index(drop=True), test_rows.reset_index(drop=True)


def build_rating_matrix(train, n_users, n_items, user_idx, item_idx):
    mat = np.zeros((n_users, n_items))
    for row in train.itertuples(index=False):
        mat[user_idx[row.userId], item_idx[row.movieId]] = row.rating
    return mat


def build_item_genre_matrix(movies_df: pd.DataFrame, item_ids):
    """(n_items, n_genres) 정규화된 장르 멤버십 행렬과 장르 이름 리스트.
    한 영화가 여러 장르면 멤버십 가중치를 장르 수로 나눠 행의 합이 1이 되게 함."""
    genres_by_item = movies_df.set_index("movieId")["genres"].to_dict()
    all_genres = sorted(
        {g for gs in genres_by_item.values() for g in gs.split("|") if g != "(no genres listed)"}
    )
    genre_idx = {g: i for i, g in enumerate(all_genres)}
    M = np.zeros((len(item_ids), len(all_genres)))
    for row, mid in enumerate(item_ids):
        for g in genres_by_item.get(mid, "").split("|"):
            if g in genre_idx:
                M[row, genre_idx[g]] = 1
    row_sums = M.sum(1, keepdims=True)
    M = np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums != 0)
    return M, all_genres
