"""
[시연 4] RAG(Retrieval-Augmented Generation) 에이전트
- 핵심: LLM이 모르는 사내/연구실 지식을 "벡터 검색으로 찾아 프롬프트에 붙여" 답하게 하는 구조.
  DB 관점 비유: 임베딩 = 인덱싱, 벡터DB = 인덱스가 걸린 테이블, 검색 = kNN 유사도 질의(SELECT ... ORDER BY distance LIMIT k)
- 라이브러리:
    chromadb : 로컬 벡터DB. 임베딩 저장 + 코사인 유사도 검색 담당 (pip install chromadb)
    openai   : (1) text-embedding-3-small로 텍스트→벡터 변환, (2) 답변 생성
"""
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction  # 문서 add/query 시 자동 임베딩
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv() 
client = OpenAI()

# ── 1. 벡터DB 구축 (인덱싱 단계) ──
db = chromadb.Client()  # 인메모리 모드. PersistentClient(path=...)로 바꾸면 디스크 저장
collection = db.create_collection(
    name="lab_docs",
    embedding_function=OpenAIEmbeddingFunction(model_name="text-embedding-3-small"),  # 1536차원 벡터 생성
)
collection.add(  # 문서 삽입 시 임베딩 자동 계산 → 벡터 인덱스에 저장 (INSERT에 해당)
    documents=[
        "우리 연구실 GPU 서버는 없으며 OpenRouter 기반의 상용 LLM을 API를 이용하여 사용한다.",
        "본 세미나는 매주 목요일 오후 7시에 진행한다.",
        "논문 실험 코드는 Github의 DSLab Organization에 커밋 후 실행한다.",
    ],
    ids=["doc1", "doc2", "doc3"],
)


def rag_agent(question: str) -> str:
    # ── 2. Retrieval: 질문을 같은 임베딩 공간으로 변환 → 최근접 문서 k개 검색 (kNN 질의에 해당) ──
    hits = collection.query(query_texts=[question], n_results=2)
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
    print("[답변]", rag_agent("연구실 GPU 서버가 있어?"))
    print("[답변]", rag_agent("연구실 회식은 언제야?"))  # 문서에 없음 → '모른다' 유도 (환각 억제 확인용)
