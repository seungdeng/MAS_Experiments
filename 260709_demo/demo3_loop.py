"""
병렬(Parallel) 구조: anyio 기반 비동기 처리
- 핵심: 서로 의존성 없는 에이전트들은 동시에 호출 → 총 소요시간 = max(개별 시간). 순차 대비 latency 감소
- 라이브러리:
    anyio  : asyncio/trio를 통합하는 비동기 실행 라이브러리
    openai : AsyncOpenAI 클라이언트가 비동기 호출(await)을 지원
"""
import time
import anyio                       # anyio.run(): 이벤트 루프 실행 / create_task_group(): 병렬 태스크 관리
from dotenv import load_dotenv     # .env 파일 로더
from openai import AsyncOpenAI    # 동기 OpenAI와 동일 API, 단 호출 앞에 await 필요

load_dotenv()  # .env의 OPENAI_API_KEY를 환경변수로 주입
client = AsyncOpenAI()

results: dict[str, str] = {}       # 태스크들이 결과를 담을 공유 딕셔너리 (task group 종료 후 접근)


async def agent(name: str, role: str, user_input: str) -> None:
    """비동기 에이전트: await 지점에서 제어권을 양보 → 다른 에이전트가 동시에 진행됨"""
    resp = await client.chat.completions.create(   # await: I/O 대기 중 다른 태스크 실행 허용
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": role},
                  {"role": "user", "content": user_input}],
    )
    results[name] = resp.choices[0].message.content


async def main() -> None:
    query = "멀티에이전트 시스템의 장점"
    t0 = time.perf_counter()

    # ── task group: 블록 내 start_soon()된 태스크가 전부 끝나야 블록 탈출 (구조적 동시성) ──
    async with anyio.create_task_group() as tg:
        tg.start_soon(agent, "기술", "기술 관점에서 2문장으로 답하라.", query)
        tg.start_soon(agent, "비용", "비용 관점에서 2문장으로 답하라.", query)
        tg.start_soon(agent, "리스크", "리스크 관점에서 2문장으로 답하라.", query)
    # 여기 도달 = 3개 에이전트 모두 완료 보장

    print(f"3개 에이전트 병렬 실행: {time.perf_counter() - t0:.1f}초\n")

    # ── 개별 에이전트 결과 확인 (에이전트 내부가 아닌 완료 후 일괄 출력 → 병렬 중 출력 뒤섞임 방지) ──
    for name, text in results.items():
        print(f"[{name} 에이전트]\n{text}\n")

    # ── Aggregator: 병렬 결과를 모아 최종 종합 (병렬 → 순차 결합 패턴) ──
    merged = "\n".join(f"[{k}] {v}" for k, v in results.items())
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "세 관점을 3문장으로 종합하라."},
                  {"role": "user", "content": merged}],
    )
    print("[Aggregator]\n", resp.choices[0].message.content)


if __name__ == "__main__":
    from demo_utils import enable_logging
    log_path = enable_logging("demo2")   # 이후 모든 print가 results/에도 저장됨
    anyio.run(main)   # asyncio.run()과 동일 역할. backend="trio" 지정도 가능
    print(f"\n결과 저장: {log_path}")