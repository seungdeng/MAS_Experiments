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


_ENGINEER_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=np.abs(b) > 1e-6),
}


def engineer_axes(U, H, residual, mask, n_rounds: int = 2, ops=("+", "-", "*", "/")):
    """기존 축들을 사칙연산으로 조합해 residual을 가장 잘 설명하는 새 축을
    매 라운드 하나씩 찾아 U/H에 편입시키고 residual을 갱신 (matching pursuit)."""
    U, H, residual = U.copy(), H.copy(), residual.copy()
    descriptions = []
    for _ in range(n_rounds):
        k = H.shape[0]
        best = None
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                for op_name in ops:
                    if op_name in ("+", "*") and j <= i:
                        continue
                    cand = _ENGINEER_OPS[op_name](H[i], H[j])
                    norm = np.linalg.norm(cand)
                    if norm < 1e-9:
                        continue
                    c_hat = cand / norm
                    new_u = residual @ c_hat
                    score = float(new_u @ new_u)
                    if best is None or score > best[0]:
                        best = (score, new_u, c_hat, f"Axis{i + 1} {op_name} Axis{j + 1}")
        if best is None:
            break
        _, new_u, c_hat, desc = best
        U = np.hstack([U, new_u[:, None]])
        H = np.vstack([H, c_hat[None, :]])
        residual = mask * (residual - np.outer(new_u, c_hat))
        descriptions.append(desc)
    return U, H, residual, descriptions


def top_items_per_factor(H: np.ndarray, item_idx_inv: dict, n: int = 8):
    """각 축(factor)에서 loading 절댓값이 큰 상위 아이템."""
    factors = []
    for f in range(H.shape[0]):
        top = np.argsort(-np.abs(H[f]))[:n]
        factors.append([(item_idx_inv[i], float(H[f, i])) for i in top])
    return factors


def user_residual_summary(residual, user_row, item_ids, movies_df, mask, n=5, mode="far"):
    """해당 유저가 실제로 평가한 영화 중 기존 축과의 잔차 기준 상위 n개.
    mode="far": 절댓값이 큰(기존 축이 설명 못 하는) 영화, mode="close": 절댓값이 작은(기존 축이 잘 설명하는) 영화."""
    row = residual[user_row]
    observed = np.where(mask[user_row])[0]
    if len(observed) == 0:
        return []
    order = np.argsort(-np.abs(row[observed]) if mode == "far" else np.abs(row[observed]))
    idx = observed[order][:n]
    out = []
    for i in idx:
        mid = item_ids[i]
        title = movies_df.loc[movies_df.movieId == mid, "title"].values
        if len(title):
            out.append((title[0], round(float(row[i]), 2)))
    return out
