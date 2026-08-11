import math


def _rank_of(ranked, target_movie_id: int):
    """target의 1-based 순위. 못 찾으면 None."""
    for i, (mid, _) in enumerate(ranked):
        if mid == target_movie_id:
            return i + 1
    return None


def hit_at_k(ranked, target_movie_id: int, k: int) -> int:
    top_ids = [mid for mid, _ in ranked[:k]]
    return 1 if target_movie_id in top_ids else 0


def mrr(ranked, target_movie_id: int) -> float:
    """Mean Reciprocal Rank (단일 정답 기준): 1/rank, 못 찾으면 0."""
    rank = _rank_of(ranked, target_movie_id)
    return 1.0 / rank if rank else 0.0


def ndcg_at_k(ranked, target_movie_id: int, k: int) -> float:
    """정답이 1개뿐이므로 IDCG=1, rank<=k일 때 1/log2(rank+1), 아니면 0."""
    rank = _rank_of(ranked, target_movie_id)
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)
