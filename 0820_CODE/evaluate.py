def hit_at_k(ranked, target_movie_id: int, k: int) -> int:
    top_ids = [mid for mid, _ in ranked[:k]]
    return 1 if target_movie_id in top_ids else 0
