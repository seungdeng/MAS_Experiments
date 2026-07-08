"""
Orchestrator 기반 영화 추천 멀티에이전트 시스템
- Planner  : 사용자 질의 분석 및 검색 계획 수립
- RAG      : Dense Retrieval(OpenAI Embedding) + 근거 기반 추천 생성
- Critic   : 추천 품질 평가, 미달 시 피드백과 함께 재생성 요청
- Verifier : 최종 답변의 근거성(grounding) 검증

실행: python movie_recommender.py
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CSV_PATH = os.getenv("CSV_PATH", "2003_final_summarized_clean.csv")
EMBED_CACHE_PATH = os.getenv("EMBED_CACHE_PATH", "embedding_index.pkl")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "2"))

client = OpenAI()  # OPENAI_API_KEY는 .env에서 로드


# ──────────────────────────────────────────────
# Dense Retriever
# ──────────────────────────────────────────────
class DenseRetriever:
    """short_plot 임베딩 인덱스 구축 후 코사인 유사도 기반 Dense Retrieval 수행."""

    def __init__(self, csv_path: str, cache_path: str, batch_size: int = 256):
        self.df = pd.read_csv(csv_path, encoding="utf-8-sig")
        self.df["short_plot"] = self.df["short_plot"].fillna(self.df["plot"])
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.embeddings = self._build_or_load_index()

    def _embed(self, texts: list[str]) -> np.ndarray:
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)

    def _build_or_load_index(self) -> np.ndarray:
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                cache = pickle.load(f)
            if cache["model"] == EMBED_MODEL and len(cache["embeddings"]) == len(self.df):
                return cache["embeddings"]

        texts = self.df["short_plot"].astype(str).tolist()
        chunks = []
        for i in range(0, len(texts), self.batch_size):
            chunks.append(self._embed(texts[i : i + self.batch_size]))
            print(f"[Index] embedded {min(i + self.batch_size, len(texts))}/{len(texts)}")
        embeddings = np.vstack(chunks)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        with open(self.cache_path, "wb") as f:
            pickle.dump({"model": EMBED_MODEL, "embeddings": embeddings}, f)
        return embeddings

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        q = self._embed([query])[0]
        q /= np.linalg.norm(q)
        scores = self.embeddings @ q
        idx = np.argsort(-scores)[:top_k]
        return [
            {
                "movieId": int(self.df.iloc[i]["movieId"]),
                "title": str(self.df.iloc[i]["title"]),
                "genres": str(self.df.iloc[i]["genres"]),
                "short_plot": str(self.df.iloc[i]["short_plot"]),
                "score": float(scores[i]),
            }
            for i in idx
        ]


# ──────────────────────────────────────────────
# LLM 호출 유틸
# ──────────────────────────────────────────────
def call_llm(system: str, user: str, json_mode: bool = False) -> str:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        response_format={"type": "json_object"} if json_mode else {"type": "text"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content


# ──────────────────────────────────────────────
# Agents
# ──────────────────────────────────────────────
class PlannerAgent:
    SYSTEM = (
        "You are a planner for a movie recommendation system. "
        "Analyze the user's query and produce a retrieval plan. "
        "Respond ONLY with JSON: "
        '{"search_query": "<English plot-style query for dense retrieval>", '
        '"top_k": <int between 3 and 10>, '
        '"user_intent": "<one-sentence summary of what the user wants>"}'
    )

    def plan(self, user_query: str) -> dict:
        out = call_llm(self.SYSTEM, user_query, json_mode=True)
        plan = json.loads(out)
        plan["top_k"] = int(plan.get("top_k", 5))
        return plan


class RagAgent:
    SYSTEM = (
        "You are a movie recommender. Recommend movies ONLY from the retrieved documents below. "
        "Never mention a movie that is not in the documents. "
        "For each recommendation, give the title and a one-to-two sentence reason grounded in its plot. "
        "Answer in Korean. Recommend 2-3 movies."
    )

    def __init__(self, retriever: DenseRetriever):
        self.retriever = retriever

    def run(self, plan: dict, user_query: str, feedback: str | None = None) -> tuple[str, list[dict]]:
        docs = self.retriever.retrieve(plan["search_query"], plan["top_k"])
        docs_str = json.dumps(docs, ensure_ascii=False, indent=2)
        user = (
            f"[User query]\n{user_query}\n\n"
            f"[User intent]\n{plan['user_intent']}\n\n"
            f"[Retrieved documents]\n{docs_str}"
        )
        if feedback:
            user += f"\n\n[Critic feedback — revise accordingly]\n{feedback}"
        answer = call_llm(self.SYSTEM, user)
        return answer, docs


class CriticAgent:
    SYSTEM = (
        "You are a critic evaluating a movie recommendation answer. "
        "Check: (1) relevance to the user's query, (2) whether reasons are specific and plot-grounded, "
        "(3) whether 2-3 movies are recommended. "
        'Respond ONLY with JSON: {"passed": true/false, "feedback": "<what to fix, empty if passed>"}'
    )

    def review(self, user_query: str, answer: str) -> dict:
        user = f"[User query]\n{user_query}\n\n[Answer]\n{answer}"
        return json.loads(call_llm(self.SYSTEM, user, json_mode=True))


class VerifierAgent:
    SYSTEM = (
        "You are a verifier. Check whether every movie mentioned in the answer exists in the "
        "retrieved documents and whether the stated reasons are consistent with the documents' plots. "
        'Respond ONLY with JSON: {"verified": true/false, "issues": "<hallucinated titles or claims, empty if verified>"}'
    )

    def verify(self, answer: str, docs: list[dict]) -> dict:
        docs_str = json.dumps(
            [{"title": d["title"], "short_plot": d["short_plot"]} for d in docs],
            ensure_ascii=False,
        )
        user = f"[Retrieved documents]\n{docs_str}\n\n[Answer]\n{answer}"
        return json.loads(call_llm(self.SYSTEM, user, json_mode=True))


# ──────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────
class Orchestrator:
    def __init__(self, retriever: DenseRetriever):
        self.planner = PlannerAgent()
        self.rag = RagAgent(retriever)
        self.critic = CriticAgent()
        self.verifier = VerifierAgent()

    def run(self, user_query: str) -> str:
        plan = self.planner.plan(user_query)
        print(f"[Planner] {plan}")

        feedback = None
        answer, docs = "", []
        for attempt in range(MAX_REVISIONS + 1):
            answer, docs = self.rag.run(plan, user_query, feedback)
            review = self.critic.review(user_query, answer)
            print(f"[Critic] attempt {attempt + 1}: passed={review['passed']}")
            if review["passed"]:
                break
            feedback = review["feedback"]

        verdict = self.verifier.verify(answer, docs)
        print(f"[Verifier] verified={verdict['verified']}")
        if not verdict["verified"]:
            answer += f"\n\n⚠️ 검증 경고: {verdict['issues']}"
        return answer


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def main():
    retriever = DenseRetriever(CSV_PATH, EMBED_CACHE_PATH)
    orch = Orchestrator(retriever)
    print("영화 추천 챗봇입니다. 종료: exit")
    while True:
        query = input("\n질문 > ").strip()
        if not query or query.lower() in {"exit", "quit"}:
            break
        print("\n" + orch.run(query))


if __name__ == "__main__":
    main()
