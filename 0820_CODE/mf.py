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


def fit_svd(rating_matrix: np.ndarray, k: int, extra_features: np.ndarray | None = None):
    """평균 중심화 SVD. 축은 부호 있는 선형 성분이라 회전 불변성 문제가 있음.
    extra_features(유저 x n_extra, 이미 중심화된 값)가 주어지면 압축 직전에 이어붙여서
    함께 압축하되, 압축 결과(H)는 원본 아이템 개수만큼만 남긴다 -- 새 feature가 압축(U)에는
    반영되지만 축의 "표현"은 항상 실제 아이템 공간에 남도록 하기 위함. rank-k SVD 근사를
    컬럼 부분집합으로 슬라이스한 것은 그 부분집합만 따로 근사한 것과 수학적으로 동일하므로
    정확하다."""
    mask, user_means = _user_means(rating_matrix)
    centered = np.where(mask, rating_matrix - user_means[:, None], 0)
    n_items = rating_matrix.shape[1]
    F = np.hstack([centered, extra_features]) if extra_features is not None else centered

    U, S, Vt = svds(F, k=k)
    order = np.argsort(-S)
    U, S, Vt = U[:, order], S[order], Vt[order, :]
    Vt = Vt[:, :n_items]

    recon = U @ np.diag(S) @ Vt + user_means[:, None]
    residual = np.where(mask, rating_matrix - recon, 0)
    return U, Vt, residual


def fit_nmf(
    rating_matrix: np.ndarray,
    k: int,
    extra_features: np.ndarray | None = None,
    max_iter: int = 300,
    seed: int = 42,
):
    """비음수 행렬분해. 미관측(0)을 그대로 0 평점으로 취급하는 표준 NMF이므로
    희소성이 심한 데이터에서는 편향될 수 있음 -- 결과 해석 시 유의.
    extra_features는 fit_svd와 같은 방식으로 이어붙였다가 압축 후 슬라이스해서 뺀다.
    NMF는 비음수만 허용하므로 extra_features의 음수값은 0으로 클리핑한다(사칙연산 중
    -, /의 결과가 음수일 수 있음 -- 이 방법에서는 그 정보가 일부 손실되는 한계로 남는다)."""
    n_items = rating_matrix.shape[1]
    F = rating_matrix
    if extra_features is not None:
        F = np.hstack([F, np.clip(extra_features, 0, None)])
    model = NMF(n_components=k, init="nndsvda", max_iter=max_iter, random_state=seed)
    U = model.fit_transform(F)
    H = model.components_[:, :n_items]
    recon = U @ H
    mask = rating_matrix != 0
    residual = np.where(mask, rating_matrix - recon, 0)
    return U, H, residual


def fit_factor_analysis(rating_matrix: np.ndarray, k: int, extra_features: np.ndarray | None = None, seed: int = 42):
    """평균 중심화 + 결측 0-대치 후 Factor Analysis. 변수별 노이즈 분산을
    분리 추정하므로 SVD/PCA보다 해석 가능성이 다소 높다고 알려짐.
    extra_features는 fit_svd와 동일한 방식으로 처리(이어붙여 압축 후 슬라이스)."""
    mask, user_means = _user_means(rating_matrix)
    centered = np.where(mask, rating_matrix - user_means[:, None], 0)
    n_items = rating_matrix.shape[1]
    F = np.hstack([centered, extra_features]) if extra_features is not None else centered

    model = FactorAnalysis(n_components=k, random_state=seed)
    U = model.fit_transform(F)
    H = model.components_[:, :n_items]

    recon = U @ H + user_means[:, None]
    residual = np.where(mask, rating_matrix - recon, 0)
    return U, H, residual


def fit_genre(
    rating_matrix: np.ndarray,
    k: int,
    item_genre_matrix: np.ndarray,
    extra_features: np.ndarray | None = None,
):
    """장르별 평균평점을 '입력 feature'(유저 x 장르)로 구성한 뒤, 그 위에서 SVD로
    k개 축으로 압축한다 -- 축(axis) != feature. 장르 자체가 아니라, 장르들의 선형결합인
    k개 압축 축이 실제 축이 된다. 안 본 장르는 (해당 장르에 대한 관측이 없으므로) 유저
    전체평균 기준 0(중립)으로 취급, 실제로 그 장르를 0점만큼 싫어한다는 뜻이 아님.
    반환하는 H는 top_items_per_factor로 대표 영화를 뽑기 위해 아이템 공간으로 투영한 것이고,
    H_genre(k x n_genres)는 각 압축 축이 어떤 장르들의 조합인지, G_centered(유저 x n_genres)는
    실제 중심화된 장르별 평점 feature 값 자체를 보여주기 위해 함께 반환한다(축 엔지니어링이
    새 장르 조합 feature를 계산하려면 원본 장르 feature 값에 접근해야 하므로 필요)."""
    mask, user_means = _user_means(rating_matrix)
    n_users, n_genres = rating_matrix.shape[0], item_genre_matrix.shape[1]

    weighted_ratings = rating_matrix @ item_genre_matrix  # (n_users, n_genres)
    weighted_counts = mask.astype(float) @ item_genre_matrix  # (n_users, n_genres)
    genre_avg = np.divide(
        weighted_ratings, weighted_counts, out=np.zeros((n_users, n_genres)), where=weighted_counts > 1e-9
    )
    genre_mask = weighted_counts > 1e-9
    G_centered = np.where(genre_mask, genre_avg - user_means[:, None], 0)
    F = np.hstack([G_centered, extra_features]) if extra_features is not None else G_centered

    k = min(k, n_genres - 1)
    U, S, Vt = svds(F, k=k)
    order = np.argsort(-S)
    U, S, Vt = U[:, order], S[order], Vt[order, :]
    H_genre = Vt[:, :n_genres]  # (k, n_genres): 압축된 축이 어떤 장르 조합인지

    H = H_genre @ item_genre_matrix.T  # (k, n_items): 대표영화/잔차 계산을 위해 아이템 공간으로 투영
    recon = U @ H + user_means[:, None]
    residual = np.where(mask, rating_matrix - recon, 0)
    return U, H, residual, H_genre, G_centered


FACTORIZERS = {"svd": fit_svd, "nmf": fit_nmf, "fa": fit_factor_analysis, "genre": fit_genre}


def fit_factorization(rating_matrix: np.ndarray, k: int, method: str = "svd", **kwargs):
    if method not in FACTORIZERS:
        raise ValueError(f"unknown method: {method} (choose from {list(FACTORIZERS)})")
    return FACTORIZERS[method](rating_matrix, k, **kwargs)


_ENGINEER_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=np.abs(b) > 1e-6),
}


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
