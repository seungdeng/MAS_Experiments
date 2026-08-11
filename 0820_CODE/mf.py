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


def fit_genre(rating_matrix: np.ndarray, k: int, item_genre_matrix: np.ndarray):
    """장르별 평균평점을 '입력 feature'(유저 x 장르)로 구성한 뒤, 그 위에서 SVD로
    k개 축으로 압축한다 -- 축(axis) != feature. 장르 자체가 아니라, 장르들의 선형결합인
    k개 압축 축이 실제 축이 된다. 안 본 장르는 (해당 장르에 대한 관측이 없으므로) 유저
    전체평균 기준 0(중립)으로 취급, 실제로 그 장르를 0점만큼 싫어한다는 뜻이 아님.
    반환하는 H는 top_items_per_factor로 대표 영화를 뽑기 위해 아이템 공간으로 투영한 것이고,
    H_genre(k x n_genres)는 각 압축 축이 어떤 장르들의 조합인지 보여주기 위해 함께 반환한다."""
    mask, user_means = _user_means(rating_matrix)
    n_users, n_genres = rating_matrix.shape[0], item_genre_matrix.shape[1]

    weighted_ratings = rating_matrix @ item_genre_matrix  # (n_users, n_genres)
    weighted_counts = mask.astype(float) @ item_genre_matrix  # (n_users, n_genres)
    genre_avg = np.divide(
        weighted_ratings, weighted_counts, out=np.zeros((n_users, n_genres)), where=weighted_counts > 1e-9
    )
    genre_mask = weighted_counts > 1e-9
    G_centered = np.where(genre_mask, genre_avg - user_means[:, None], 0)

    k = min(k, n_genres - 1)
    U, S, Vt = svds(G_centered, k=k)
    order = np.argsort(-S)
    U, S, Vt = U[:, order], S[order], Vt[order, :]
    H_genre = Vt  # (k, n_genres): 압축된 축이 어떤 장르 조합인지

    H = H_genre @ item_genre_matrix.T  # (k, n_items): 대표영화/잔차 계산을 위해 아이템 공간으로 투영
    recon = U @ H + user_means[:, None]
    residual = np.where(mask, rating_matrix - recon, 0)
    return U, H, residual, H_genre


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


def engineer_axes(
    U,
    H,
    residual,
    mask,
    max_rounds: int = 10,
    min_relative_gain: float = 0.01,
    ops=("+", "-", "*", "/"),
    axis_labels=None,
):
    """기존 축들을 사칙연산으로 조합해 residual을 가장 잘 설명하는 새 축을
    매 라운드 하나씩 찾아 U/H에 편입시키고 residual을 갱신 (matching pursuit).
    이번 라운드 최선 조합이 남은 잔차 제곱합의 min_relative_gain 비율도 못 줄이면
    더 추가할 의미가 없다고 보고 멈춘다 -- 축 개수를 미리 정해두지 않고 데이터가 정하게 함.
    max_rounds는 무한/과다 라운드를 막는 안전장치일 뿐, 보통은 그 전에 조기 종료됨.
    axis_labels가 주어지면(예: 장르 이름) 조합식을 "Action - Romance"처럼 해석 가능하게 표시한다.
    새로 추가되는 축에는 "E1", "E2"... 같은 짧은 이름을 붙여서 다음 라운드 조합 재료로 쓴다 --
    조합식 문자열 자체를 라벨로 재사용하면 라운드를 거듭할수록 라벨이 기하급수적으로 길어져
    (이전 조합을 통째로 이어붙이게 됨) 프롬프트에 넣기 부적합해지기 때문. 각 엔지니어링 축의
    실제 유도식은 descriptions에 "E1 = Axis1 + Axis2" 형태로 한 줄씩 남는다."""
    U, H, residual = U.copy(), H.copy(), residual.copy()
    labels = list(axis_labels) if axis_labels is not None else [f"Axis{i + 1}" for i in range(H.shape[0])]
    descriptions = []
    used = set()  # 이미 선택된 조합은 재선택 금지 -- mask 재적용으로 직교분해가 깨져 같은 축이 반복 선택될 수 있음
    engineered_count = 0
    for _ in range(max_rounds):
        total_before = float((residual**2).sum())
        if total_before < 1e-12:
            break
        k = H.shape[0]
        best = None
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                for op_name in ops:
                    if op_name in ("+", "*") and j <= i:
                        continue
                    combo = f"{labels[i]} {op_name} {labels[j]}"
                    if combo in used:
                        continue
                    cand = _ENGINEER_OPS[op_name](H[i], H[j])
                    norm = np.linalg.norm(cand)
                    if norm < 1e-9:
                        continue
                    c_hat = cand / norm
                    new_u = residual @ c_hat
                    score = float(new_u @ new_u)
                    if best is None or score > best[0]:
                        best = (score, new_u, c_hat, combo)
        if best is None:
            break
        score, new_u, c_hat, combo = best
        if score / total_before < min_relative_gain:
            break
        used.add(combo)
        engineered_count += 1
        new_label = f"E{engineered_count}"
        U = np.hstack([U, new_u[:, None]])
        H = np.vstack([H, c_hat[None, :]])
        residual = mask * (residual - np.outer(new_u, c_hat))
        descriptions.append(f"{new_label} = {combo}")
        labels.append(new_label)
    return U, H, residual, descriptions, labels


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
