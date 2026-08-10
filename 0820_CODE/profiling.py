from llm_client import chat


def profile_raw(user_history):
    """Baseline: 시청 이력만으로 LLM이 직접 프로필 생성 (KAR/ONCE 방식)."""
    lines = "\n".join(f"- {t} ({g}) rated {r}/5" for t, g, r in user_history)
    prompt = (
        "다음은 한 유저가 시청하고 평가한 영화 목록입니다.\n"
        f"{lines}\n\n"
        "이 정보를 바탕으로 이 유저의 영화 취향을 200자 내외로 서술하세요. "
        "장르, 톤, 선호 패턴을 구체적으로 언급하세요."
    )
    return chat([{"role": "user", "content": prompt}])


def profile_axis(user_history, factor_summaries, residual_items):
    """Axis-based: MF로 뽑은 축 + 축이 설명 못 하는 잔차를 조건으로 프로필 생성."""
    factors_text = "\n".join(
        f"축 {i + 1}: 대표 영화 - " + ", ".join(t for t, _ in fs[:5])
        for i, fs in enumerate(factor_summaries)
    )
    residual_text = "\n".join(f"- {t} (예측 오차 {e})" for t, e in residual_items) or "(없음)"
    lines = "\n".join(f"- {t} ({g}) rated {r}/5" for t, g, r in user_history)
    prompt = (
        "다음은 행렬분해로 추출한 유저 취향의 잠재 축, 이 유저의 시청 이력, "
        "그리고 기존 축으로 설명이 잘 안 되는(예측 오차가 큰) 영화 목록입니다.\n\n"
        f"[잠재 축]\n{factors_text}\n\n"
        f"[시청 이력]\n{lines}\n\n"
        f"[기존 축으로 설명 안 되는 영화]\n{residual_text}\n\n"
        "기존 축들이 놓치고 있는 취향의 특성을 짚어내어, "
        "이 유저의 영화 취향을 200자 내외로 서술하세요."
    )
    return chat([{"role": "user", "content": prompt}])
