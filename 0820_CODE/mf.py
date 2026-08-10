import numpy as np
from scipy.sparse.linalg import svds


def fit_svd(rating_matrix: np.ndarray, k: int):
    """평균 중심화 후 SVD. rating_matrix의 0은 미관측을 의미."""
    mask = rating_matrix != 0
    counts = mask.sum(1)
    user_means = np.divide(
        rating_matrix.sum(1), counts, out=np.zeros(rating_matrix.shape[0]), where=counts != 0
    )
    centered = np.where(mask, rating_matrix - user_means[:, None], 0)

    U, S, Vt = svds(centered, k=k)
    order = np.argsort(-S)
    U, S, Vt = U[:, order], S[order], Vt[order, :]

    recon = U @ np.diag(S) @ Vt + user_means[:, None]
    residual = np.where(mask, rating_matrix - recon, 0)
    return U, S, Vt, user_means, residual


def top_items_per_factor(Vt: np.ndarray, item_idx_inv: dict, n: int = 8):
    """각 축(factor)에서 loading 절댓값이 큰 상위 아이템 인덱스와 가중치."""
    factors = []
    for f in range(Vt.shape[0]):
        top = np.argsort(-np.abs(Vt[f]))[:n]
        factors.append([(item_idx_inv[i], float(Vt[f, i])) for i in top])
    return factors


def user_residual_summary(residual, user_row, item_ids, movies_df, n=5):
    """해당 유저에서 기존 축으로 설명이 잘 안 되는(잔차 절댓값이 큰) 영화 n개."""
    row = residual[user_row]
    idx = np.argsort(-np.abs(row))[:n]
    out = []
    for i in idx:
        if row[i] == 0:
            continue
        mid = item_ids[i]
        title = movies_df.loc[movies_df.movieId == mid, "title"].values
        if len(title):
            out.append((title[0], round(float(row[i]), 2)))
    return out
