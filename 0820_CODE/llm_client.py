import requests

from config import MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL


def chat(messages, temperature: float = 0.3, max_tokens: int = 400) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("환경변수 OPENROUTER_API_KEY가 설정되지 않았습니다.")
    resp = requests.post(
        OPENROUTER_BASE_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
