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
