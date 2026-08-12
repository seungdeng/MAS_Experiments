import os

import numpy as np
import pandas as pd

from config import DATA_DIR


def load_movielens():
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    return ratings, movies


def filter_users_by_history(
    ratings: pd.DataFrame,
    min_history_len: int,
    max_history_len: int | None = None,
    max_users: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """이력 길이(전체 평점 수)가 [min_history_len, max_history_len] 구간인 유저의 평점만 남김
    (max_history_len=None이면 상한 없음). max_users가 주어지면 그중 무작위로 그 수만큼만 뽑는다 --
    이력 긴 유저 전체(수만 명)를 다 넣으면 이들이 합쳐서 본 아이템이 카탈로그 대부분을 덮어
    dense 행렬이 감당 못 할 크기가 되므로, 풀을 좁혀 build_rating_matrix의 유저/아이템 universe를
    함께 줄인다."""
    counts = ratings.groupby("userId").size()
    in_range = counts >= min_history_len
    if max_history_len is not None:
        in_range &= counts <= max_history_len
    selected_users = counts[in_range].index
    if max_users is not None and len(selected_users) > max_users:
        rng = np.random.default_rng(seed)
        selected_users = rng.choice(selected_users, size=max_users, replace=False)
    return ratings[ratings.userId.isin(selected_users)].reset_index(drop=True)


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
