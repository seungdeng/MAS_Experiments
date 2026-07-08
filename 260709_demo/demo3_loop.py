"""
[시연 3] 루프(Loop) 구조 — Generator-Critic 패턴
- 핵심: 생성 → 평가 → (불합격 시) 피드백 반영 재생성. 종료조건 = "합격 판정" or "최대 반복 횟수".
- 오류 귀인 연구와의 연결점: 루프 구조에서는 '몇 번째 반복의 어느 에이전트가 실패 원인인가'가 문제가 됨.
- 라이브러리: openai (시연 1과 동일. 루프는 프레임워크 없이 순수 파이썬 while문으로 구현됨을 보여주는 것이 포인트)
"""
from openai import OpenAI

client = OpenAI()


def call(role: str, user_input: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": role},
                  {"role": "user", "content": user_input}],
        temperature=0.5,
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    task = "초등학생도 이해할 수 있는 '데이터베이스 인덱스' 설명문 3문장"
    MAX_ITER = 3          # 무한루프 방지용 상한 (실무에서 필수)
    feedback = "없음"

    for i in range(1, MAX_ITER + 1):
        # ── Generator: 이전 피드백을 프롬프트에 주입하여 재생성 ──
        draft = call(
            "당신은 작가다. 피드백이 있으면 반드시 반영하라.",
            f"과제: {task}\n이전 피드백: {feedback}",
        )
        print(f"[{i}회차 초안]\n{draft}\n")

        # ── Critic: 첫 단어를 PASS/FAIL로 강제 → 파싱 가능한 종료 신호 확보 ──
        verdict = call(
            "당신은 평가자다. 초등학생 눈높이에 맞으면 'PASS', 아니면 'FAIL: <이유 1문장>' 형식으로만 답하라.",
            draft,
        )
        print(f"[{i}회차 평가] {verdict}\n")

        if verdict.strip().startswith("PASS"):   # 종료조건 1: 합격
            print(f"=> {i}회 만에 수렴")
            break
        feedback = verdict                        # FAIL 사유를 다음 반복의 입력으로 전달
    else:                                         # 종료조건 2: 상한 도달 (for-else: break 없이 끝난 경우)
        print("=> 최대 반복 도달, 마지막 초안 채택")
