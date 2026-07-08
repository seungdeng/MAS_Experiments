"""
RAG(Retrieval-Augmented Generation) 에이전트
- 핵심: LLM이 모르는 사내/연구실 지식을 "벡터 검색으로 찾아 프롬프트에 붙여" 답하게 하는 구조.
  DB 관점 비유: 임베딩 = 인덱싱, 벡터DB = 인덱스가 걸린 테이블, 검색 = kNN 유사도 질의(SELECT ... ORDER BY distance LIMIT k)
- 라이브러리:
    chromadb : 로컬 벡터DB. 임베딩 저장 + 코사인 유사도 검색 담당 (pip install chromadb)
    openai   : (1) text-embedding-3-small로 텍스트→벡터 변환, (2) 답변 생성
"""
import json                     # 표준 라이브러리: JSON 파싱
from pathlib import Path        # 표준 라이브러리: 스크립트 기준 상대경로 처리

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction  # 문서 add/query 시 자동 임베딩
from dotenv import load_dotenv  # .env 파일 로더
from openai import OpenAI

load_dotenv()  # .env의 OPENAI_API_KEY를 환경변수로 주입 (chromadb 임베딩 함수도 이 키를 사용)
client = OpenAI()

TOP_K = 2  # retrieval에서 가져올 최근접 문서 수 (top-k)

# ── 0. 문서 로딩: 하드코딩 대신 JSON 파일에서 읽기 ──
#    형식: [{"id": "doc1", "text": "..."}, ...]  (실무에서는 이 자리에 PDF 파싱/청킹 로직이 들어감)
docs_path = Path(__file__).parent / "lab_docs.json"      # 스크립트 위치 기준 → 실행 위치 무관
docs = json.loads(docs_path.read_text(encoding="utf-8")) # 파일 전체 읽기 + JSON 파싱을 한 번에

# ── 1. 벡터DB 구축 (인덱싱 단계) ──
db = chromadb.Client()  # 인메모리 모드. PersistentClient(path=...)로 바꾸면 디스크 저장
collection = db.create_collection(
    name="lab_docs",
    embedding_function=OpenAIEmbeddingFunction(model_name="text-embedding-3-small"),  # 1536차원 밀집 벡터 → dense retrieval
)
collection.add(  # 문서 삽입 시 임베딩 자동 계산 → 벡터 인덱스에 저장 (INSERT에 해당)
    documents=[d["text"] for d in docs],  # JSON에서 읽은 본문 리스트
    ids=[d["id"] for d in docs],          # JSON에서 읽은 id 리스트
)
print(f"[인덱싱] {docs_path.name}에서 문서 {len(docs)}건 로드 완료\n")


def rag_agent(question: str) -> str:
    # ── 2. Retrieval(dense): 질문을 같은 임베딩 공간으로 변환 → 최근접 문서 TOP_K개 검색 (kNN 질의) ──
    hits = collection.query(query_texts=[question], n_results=TOP_K)  # n_results = top-k
    context = "\n".join(hits["documents"][0])
    print(f"[검색된 문서]\n{context}\n")

    # ── 3. Generation: 검색 결과를 프롬프트에 주입 → 근거 기반 답변 강제 ──
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "아래 [참고문서]에 근거해서만 답하라. 근거가 없으면 '모른다'고 답하라."},  # 환각 억제 장치
            {"role": "user", "content": f"[참고문서]\n{context}\n\n[질문]\n{question}"},
        ],
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    from demo_utils import enable_logging
    log_path = enable_logging("demo4")   # 이후 모든 print가 results/에도 저장됨

    print("[답변]", rag_agent("GPU 서버 사용은 어떻게 해?"))
    print("[답변]", rag_agent("신입생 자료는 어디에서 볼 수 있어?")) 

    print(f"\n결과 저장: {log_path}")