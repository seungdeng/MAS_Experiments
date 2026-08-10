import re

from llm_client import chat


def rank_candidates(profile_text: str, candidates):
    """candidates: list of (movieId, title). LLM이 선호 순서로 재정렬."""
    listing = "\n".join(f"{i + 1}. {title}" for i, (_, title) in enumerate(candidates))
    prompt = (
        f"유저 프로필:\n{profile_text}\n\n"
        f"다음은 영화 후보 목록입니다:\n{listing}\n\n"
        "이 유저가 가장 좋아할 순서대로 상위 5개 영화의 번호만 쉼표로 구분해 출력하세요. "
        "설명 없이 번호만 출력하세요. 예: 3,1,7,2,5"
    )
    resp = chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=50)
    nums = [int(x) for x in re.findall(r"\d+", resp)]

    seen = set()
    ranked = []
    for n in nums:
        if 1 <= n <= len(candidates) and n not in seen:
            seen.add(n)
            ranked.append(candidates[n - 1])
    return ranked
