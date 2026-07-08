"""
[시연 1] 순차(Sequential) 구조
- 핵심: 에이전트 = "역할(시스템 프롬프트) + LLM 호출"일 뿐. 앞 에이전트의 출력이 뒤 에이전트의 입력이 됨.
- 라이브러리: openai (LLM API 호출용 공식 SDK. pip install openai)
"""
import os
from openai import OpenAI  # OpenAI API 클라이언트. 환경변수 OPENAI_API_KEY 자동 인식
from dotenv import load_dotenv

load_dotenv() 
client = OpenAI()  # api_key=os.environ["OPENAI_API_KEY"] 를 내부에서 읽음


def agent(role: str, user_input: str) -> str:
    """에이전트의 최소 단위: 시스템 프롬프트(role)로 페르소나를 부여한 단일 LLM 호출."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",         
        messages=[
            {"role": "system", "content": role},        # 역할 정의
            {"role": "user", "content": user_input},    # 이전 단계 출력이 여기로 들어옴
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content
# resp                          # ChatCompletion 객체 (id, model, usage 등 메타데이터 포함)
# resp.choices                  # 응답 후보 리스트. n=3으로 요청하면 3개가 옴 (기본 n=1이라 보통 1개)
# resp.choices[0]               # 첫 번째 후보 (Choice 객체: message, finish_reason 등 보유)
# resp.choices[0].message       # assistant의 메시지 객체 (role="assistant", content, tool_calls 등)
# resp.choices[0].message.content  # 그중 텍스트 본문 (str)

if __name__ == "__main__":
    topic = "대학원 연구실의 논문 관리 자동화"

    # Pipeline
    plan = agent("당신은 기획자다. 주제에 대한 3단계 목차만 작성하라.", topic)
    print("[1. Planner]\n", plan, "\n")

    draft = agent("당신은 작가다. 주어진 목차대로 5문장 요약문을 작성하라.", plan)
    print("[2. Writer]\n", draft, "\n")

    review = agent("당신은 검토자다. 글의 문제점 2가지만 지적하라.", draft)
    print("[3. Reviewer]\n", review)
