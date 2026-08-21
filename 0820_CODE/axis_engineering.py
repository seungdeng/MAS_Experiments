import numpy as np

from llm_client import chat
from mf import _ENGINEER_OPS, _user_means, fit_factorization, top_items_per_factor


def _global_far_items(residual, item_idx_inv, title_map, n=10):
    """전체 유저 합산 |잔차| 기준 상위 n개 영화(컬럼 인덱스, 제목, 오차) -- 지금 축들이
    전반적으로 못 설명하는 대상. (유저 개인화된 far와 달리, feature를 설계하기 위한 전역 신호)"""
    agg = np.abs(residual).sum(axis=0)
    idx = np.argsort(-agg)[:n]
    return [
        (int(i), title_map.get(item_idx_inv[i], str(item_idx_inv[i])), round(float(agg[i]), 1))
        for i in idx
        if agg[i] > 0
    ]


def _axis_context(H, labels, item_idx_inv, title_map, n=4):
    summaries = top_items_per_factor(H, item_idx_inv, n=n)
    lines = []
    for label, fs in zip(labels, summaries):
        items = ", ".join(f"{title_map.get(mid, str(mid))}({'+' if w >= 0 else '-'})" for mid, w in fs)
        lines.append(f"{label}: {items}")
    return "\n".join(lines)


def _genre_axis_labels(H_genre, genre_names, n=3):
    genre_idx_inv = {i: g for i, g in enumerate(genre_names)}
    summaries = top_items_per_factor(H_genre, genre_idx_inv, n=n)
    return ["+".join(f"{g}({'+' if w >= 0 else '-'})" for g, w in gs) for gs in summaries]


def _propose_genre_combo(genre_names, far_items, method):
    """장르는 19개 전부 이름이 있으므로 전체를 후보로 보여주고 "GenreA op GenreB" 자유 제안."""
    far_text = "\n".join(f"- {t} (total abs error {e})" for _, t, e in far_items) or "(none)"
    prompt = (
        f"We are building interpretable taste features for a movie recommender ({method} method).\n\n"
        f"[Available genres] {', '.join(genre_names)}\n\n"
        "[Movies poorly explained overall, across all users, by the current model]\n"
        f"{far_text}\n\n"
        "Propose ONE new engineered feature by combining exactly two of the genres above with "
        "exactly one of these operators: +, -, *, /. Pick a combination that seems likely to "
        "help explain the poorly-explained movies above.\n"
        "Respond with EXACTLY one line, using the genre names verbatim, in this format:\n"
        "GENRE_A <op> GENRE_B\n"
        "Example: Action - Romance"
    )
    resp = chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=50)
    line = resp.strip().splitlines()[0].strip() if resp.strip() else ""
    tokens = line.split()
    if len(tokens) != 3:
        return None
    label_a, op_name, label_b = tokens
    if op_name not in _ENGINEER_OPS:
        return None
    if label_a == label_b or label_a not in genre_names or label_b not in genre_names:
        return None
    return label_a, op_name, label_b


def _propose_item_combo(H, axis_labels, item_idx_inv, title_map, far_items, method):
    """영화는 이름에 공백이 있어 자유 텍스트 파싱이 불안정하므로, 가장 못 설명되는 영화
    상위 n개로 후보를 좁혀 번호를 매기고 "3 - 7" 같은 번호 응답을 받는다(recommend.py의
    번호 기반 랭킹과 동일한 패턴)."""
    if len(far_items) < 2:
        return None
    axis_context = _axis_context(H, axis_labels, item_idx_inv, title_map)
    listing = "\n".join(f"{i + 1}. {t} (total abs error {e})" for i, (_, t, e) in enumerate(far_items))
    prompt = (
        f"We are building interpretable taste features for a movie recommender ({method} method).\n\n"
        f"[Current axes] each shown with its top representative movies (+/- = which end of the axis)\n"
        f"{axis_context}\n\n"
        "[Movies the current axes explain poorly overall, across all users -- numbered]\n"
        f"{listing}\n\n"
        "Propose ONE new engineered feature by combining exactly two of the numbered movies above "
        "with exactly one of these operators: +, -, *, /, applied to their rating columns. Pick a "
        "combination that seems likely to reveal a taste dimension the current axes miss.\n"
        "Respond with EXACTLY one line, using the movie numbers, in this format:\n"
        "NUM_A <op> NUM_B\n"
        "Example: 3 - 7"
    )
    resp = chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=20)
    line = resp.strip().splitlines()[0].strip() if resp.strip() else ""
    tokens = line.split()
    if len(tokens) != 3:
        return None
    tok_a, op_name, tok_b = tokens
    if op_name not in _ENGINEER_OPS or not tok_a.isdigit() or not tok_b.isdigit():
        return None
    ia, ib = int(tok_a), int(tok_b)
    if ia == ib or not (1 <= ia <= len(far_items)) or not (1 <= ib <= len(far_items)):
        return None
    return far_items[ia - 1], op_name, far_items[ib - 1]


def _ask_continue(descriptions, last_gain_pct):
    prompt = (
        f"Engineered features accepted so far: {len(descriptions)}\n"
        f"The feature just added reduced the model's overall prediction error by {last_gain_pct:.1f}%.\n\n"
        "Should we propose one more engineered feature, or is the current model already "
        "sufficient? Respond with exactly one word: CONTINUE or STOP."
    )
    resp = chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=10)
    return resp.strip().upper().startswith("CONTINUE")


def engineer_axes_llm(
    rating_matrix,
    k,
    mask,
    method,
    item_idx_inv,
    title_map,
    item_genre_matrix=None,
    genre_names=None,
    max_rounds=10,
):
    """축끼리 직접 조합하는 대신, LLM이 제안한 조합으로 새 feature를 만들어 원본 feature
    행렬(genre는 유저x장르, 나머지는 유저x영화)에 컬럼으로 추가하고, 매 라운드 그 방법
    (svd/nmf/fa/genre)으로 처음부터 다시 압축한다 -- "축은 항상 feature를 압축한 결과"라는
    원칙이 새로 추가되는 정보에도 그대로 유지되도록 하기 위함. 몇 개를 추가할지도 LLM이
    CONTINUE/STOP으로 직접 결정한다. 서버는 제안을 검증만 하고(무효면 조용히 중단,
    fail-safe), 반영 계산(feature 컬럼 생성 + 재압축)은 수치적으로 수행한다."""
    fit_kwargs = {"item_genre_matrix": item_genre_matrix} if method == "genre" else {}
    result = fit_factorization(rating_matrix, k, method=method, **fit_kwargs)
    if method == "genre":
        U, H, residual, H_genre, G_centered = result
        axis_labels = _genre_axis_labels(H_genre, genre_names)
    else:
        U, H, residual = result
        axis_labels = [f"Axis{i + 1}" for i in range(H.shape[0])]
        G_centered = None

    _, user_means = _user_means(rating_matrix)
    engineered_cols = []
    engineered_desc = []
    used = set()

    for _ in range(max_rounds):
        prev_err = float((residual**2).sum())
        if prev_err < 1e-12:
            break

        far_items = _global_far_items(residual, item_idx_inv, title_map, n=10)

        if method == "genre":
            proposal = _propose_genre_combo(genre_names, far_items, method)
            if proposal is None:
                break
            label_a, op_name, label_b = proposal
            key = (label_a, op_name, label_b)
            if key in used:
                break
            ia, ib = genre_names.index(label_a), genre_names.index(label_b)
            new_col = _ENGINEER_OPS[op_name](G_centered[:, ia], G_centered[:, ib])
            desc_text = f"{label_a} {op_name} {label_b}"
        else:
            proposal = _propose_item_combo(H, axis_labels, item_idx_inv, title_map, far_items, method)
            if proposal is None:
                break
            item_a, op_name, item_b = proposal
            key = (item_a[0], op_name, item_b[0])
            if key in used:
                break
            combo = _ENGINEER_OPS[op_name](rating_matrix[:, item_a[0]], rating_matrix[:, item_b[0]])
            new_col = combo - user_means
            desc_text = f"{item_a[1]} {op_name} {item_b[1]}"

        used.add(key)
        engineered_cols.append(new_col)
        extra_features = np.column_stack(engineered_cols)
        result = fit_factorization(rating_matrix, k, method=method, extra_features=extra_features, **fit_kwargs)
        if method == "genre":
            U, H, new_residual, H_genre, G_centered = result
        else:
            U, H, new_residual = result

        new_err = float((new_residual**2).sum())
        gain_pct = 100 * (prev_err - new_err) / prev_err if prev_err > 0 else 0.0
        residual = new_residual
        engineered_desc.append(f"{desc_text} (residual -{gain_pct:.1f}%)")

        if not _ask_continue(engineered_desc, gain_pct):
            break

    return U, H, residual, engineered_desc, axis_labels
