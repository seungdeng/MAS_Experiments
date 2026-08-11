import re

from llm_client import chat


def rank_candidates(profile_text: str, candidates):
    """candidates: list of (movieId, title). LLM이 전체 후보를 선호 순서로 재정렬(랭킹 기반 지표용 full ranking)."""
    n = len(candidates)
    listing = "\n".join(f"{i + 1}. {title}" for i, (_, title) in enumerate(candidates))
    prompt = (
        f"User profile:\n{profile_text}\n\n"
        f"Here is a list of {n} candidate movies:\n{listing}\n\n"
        f"Rank all {n} movies from most to least likely for this user to enjoy. "
        "Output the numbers only, separated by commas, no explanation. "
        "Example: 3,1,7,2,5,..."
    )
    resp = chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=200)
    nums = [int(x) for x in re.findall(r"\d+", resp)]

    seen = set()
    ranked = []
    for n in nums:
        if 1 <= n <= len(candidates) and n not in seen:
            seen.add(n)
            ranked.append(candidates[n - 1])
    return ranked
