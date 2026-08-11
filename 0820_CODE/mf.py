import numpy as np
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF, FactorAnalysis


def _user_means(rating_matrix):
    mask = rating_matrix != 0
    counts = mask.sum(1)
    means = np.divide(
        rating_matrix.sum(1), counts, out=np.zeros(rating_matrix.shape[0]), where=counts != 0
    )
    return mask, means


def fit_svd(rating_matrix: np.ndarray, k: int):
    """평균 중심화 SVD. 축은 부호 있는 선형 성분이라 회전 불변성 문제가 있음."""
    mask, user_means = _user_means(rating_matrix)
    centered = np.where(mask, rating_matrix - user_means[:, None], 0)

    U, S, Vt = svds(centered, k=k)
    order = np.argsort(-S)
    U, S, Vt = U[:, order], S[order], Vt[order, :]

    recon = U @ np.diag(S) @ Vt + user_means[:, None]
    residual = np.where(mask, rating_matrix - recon, 0)
    return U, Vt, residual


def fit_nmf(rating_matrix: np.ndarray, k: int, max_iter: int = 300, seed: int = 42):
    """비음수 행렬분해. 미관측(0)을 그대로 0 평점으로 취급하는 표준 NMF이므로
    희소성이 심한 데이터에서는 편향될 수 있음 -- 결과 해석 시 유의."""
    model = NMF(n_components=k, init="nndsvda", max_iter=max_iter, random_state=seed)
    U = model.fit_transform(rating_matrix)
    H = model.components_
    recon = U @ H
    mask = rating_matrix != 0
    residual = np.where(mask, rating_matrix - recon, 0)
    return U, H, residual


def fit_factor_analysis(rating_matrix: np.ndarray, k: int, seed: int = 42):
    """평균 중심화 + 결측 0-대치 후 Factor Analysis. 변수별 노이즈 분산을
    분리 추정하므로 SVD/PCA보다 해석 가능성이 다소 높다고 알려짐."""
    mask, user_means = _user_means(rating_matrix)
    centered = np.where(mask, rating_matrix - user_means[:, None], 0)

    model = FactorAnalysis(n_components=k, random_state=seed)
    U = model.fit_transform(centered)
    H = model.components_

    recon = U @ H + user_means[:, None]
    residual = np.where(mask, rating_matrix - recon, 0)
    return U, H, residual


FACTORIZERS = {"svd": fit_svd, "nmf": fit_nmf, "fa": fit_factor_analysis}


def fit_factorization(rating_matrix: np.ndarray, k: int, method: str = "svd"):
    if method not in FACTORIZERS:
        raise ValueError(f"unknown method: {method} (choose from {list(FACTORIZERS)})")
    return FACTORIZERS[method](rating_matrix, k)


def top_items_per_factor(H: np.ndarray, item_idx_inv: dict, n: int = 8):
    """각 축(factor)에서 loading 절댓값이 큰 상위 아이템."""
    factors = []
    for f in range(H.shape[0]):
        top = np.argsort(-np.abs(H[f]))[:n]
        factors.append([(item_idx_inv[i], float(H[f, i])) for i in top])
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
