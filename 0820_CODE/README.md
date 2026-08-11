# MF-Axis vs Raw-History 프로필 비교

LLM이 유저의 영화 취향 프로필을 생성할 때, **시청 이력만 주는 것**과 **행렬분해(SVD/NMF/FA)로 뽑은 잠재 축 + 그 축이 잘 반영하는(close) 영화와 못 하는(far) 영화를 대비시켜 함께 주는 것** 중 어느 쪽이 더 정확한 추천으로 이어지는지 Hit@K/MRR/NDCG@K로 비교하는 실험 코드입니다.

## 동작 방식 (파이프라인)
1. **데이터 로딩 & 분할** (`data.py`) — MovieLens `ratings.csv`/`movies.csv`를 읽고, 유저별 가장 최근 평점 1개를 leave-one-out test로 분리, 나머지를 train으로 사용.
2. **행렬분해 + 축 엔지니어링** (`mf.py`) — train 평점 행렬을 `config.FACTOR_METHODS`에 지정된 기법(svd/nmf/fa)별로 분해해 `N_FACTORS`개의 잠재 축(U, H)과 잔차를 계산. 이어서 `engineer_axes()`가 기존 축들을 **사칙연산(+, -, ×, ÷)으로 조합**한 후보들을 모두 만들어보고, 현재 잔차를 가장 많이 설명하는(=잔차에 가장 크게 투영되는) 조합을 매 라운드 하나씩 찾아 축 집합에 편입시키고 잔차를 갱신 — matching pursuit 방식으로 `N_ENGINEERED_AXES`번 반복(매 라운드 새로 추가된 축도 다음 조합 대상에 포함되어 순환적으로 개선). 이렇게 만들어진 축(기본 축 + 엔지니어링 축)에서 대표 영화(loading 상위 n개)를 추출하고, `user_residual_summary(mode=...)`로 유저가 평가한 영화 중 잔차 절댓값이 **작은(close, 축들이 잘 설명하는)** 영화와 **큰(far, 축들이 설명 못 하는)** 영화를 각각 상위 n개씩 뽑음.
3. **프로필 생성** (`profiling.py`, LLM 호출) — 평가 대상 유저마다 두 방식으로 취향 프로필 텍스트를 생성:
   - `profile_raw`: 시청 이력만 보고 LLM이 직접 서술 (baseline, KAR/ONCE 방식)
   - `profile_axis`: 잠재 축 요약(+/- 부호로 축의 두 방향 표시) + 시청 이력 + **close 영화(축이 잘 반영하는 취향)**와 **far 영화(축이 반영 못 하는 취향)**를 함께 주고, 그 대비를 근거로 기존 축이 놓친 취향을 짚어내도록 서술 — `FACTOR_METHODS`의 기법마다 하나씩 생성
4. **후보 랭킹** (`recommend.py`, LLM 호출) — 정답 영화 1개 + 랜덤 negative `N_NEGATIVES`개(기본 19개, 총 후보 20개)를 섞어(position bias 방지) 각 프로필 텍스트를 조건으로 LLM이 전체 후보를 선호 순서로 재정렬. 프롬프트/응답은 영어(데이터셋이 영어이므로).
5. **평가** (`evaluate.py`) — 재정렬된 순위에서 정답 영화의 순위를 기준으로 세 지표를 계산: Hit@K(상위 `TOP_K` 안에 있으면 1, 아니면 0), MRR(1/rank), NDCG@K(1/log2(rank+1), rank가 `TOP_K` 밖이면 0). 정답이 항상 1개뿐이라 순위 정보를 보존하는 MRR/NDCG가 Hit@K의 "있다/없다" 손실을 보완.
6. **오케스트레이션** (`main.py`) — 위 과정을 `N_EVAL_USERS`명에 대해 반복하고 결과를 취합, 방법별 평균 Hit@K/MRR/NDCG@K를 콘솔에 출력.

LLM 호출은 `llm_client.py`를 통해 OpenRouter Chat Completions API(`config.MODEL`, 기본 `openai/gpt-4o-mini`)로 이루어집니다.

## 준비
```bash
pip install -r requirements.txt
set OPENROUTER_API_KEY=sk-or-...
```

`ml-latest-small` 데이터를 https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
에서 받아 `data/ml-latest-small/` 에 압축 해제 (ratings.csv, movies.csv 필요).

## 실행
```bash
python main.py
# N_FACTORS 민감도 스윕, 결과 파일 분리
python main.py --n-factors 5 --out results_nf5.csv
python main.py --n-factors 8 --out results_nf8.csv
# 평가 유저 수 오버라이드
python main.py --n-eval-users 150
```
`--n-factors`/`--n-eval-users`/`--out`은 `config.py`의 `N_FACTORS`/`N_EVAL_USERS`/`results.csv`를 일시적으로 덮어씁니다(파일은 그대로 둠).

`config.py`에서 조정 가능한 값:
- `FACTOR_METHODS`: 비교할 축 압축 기법 목록 (svd/nmf/fa 중 선택, `mf.FACTORIZERS`에 정의)
- `N_FACTORS`: 기본 잠재 축 개수 (CLI `--n-factors`로 실행 시 오버라이드 가능)
- `N_ENGINEERED_AXES`: 기존 축을 사칙연산으로 조합해 추가로 만들어낼 축 개수 (method당, matching pursuit 라운드 수)
- `N_NEGATIVES`: 랭킹 후보에 섞을 negative 샘플 수
- `N_EVAL_USERS`: 평가할 유저 수 (기본 150, CLI `--n-eval-users`로 오버라이드 가능)
- `TOP_K`: Hit@K, NDCG@K의 K
- `RANDOM_SEED`: 재현성용 시드
- `MODEL`: 사용할 LLM (OpenRouter 모델 ID)

실행 결과는 `results.csv`에 유저별 프로필 텍스트(`profile_raw`, `profile_svd`, `profile_nmf`, `profile_fa` 등 `FACTOR_METHODS`에 따라 결정)와 방법별 hit/mrr/ndcg 값으로 저장되며, 콘솔에는 방법별 평균 Hit@K/MRR/NDCG@K가 출력됩니다.

## 구조
- `config.py`: API 키/모델, 데이터 경로, 실험 하이퍼파라미터
- `data.py`: MovieLens 로딩, leave-one-out split, 평점 행렬 생성
- `mf.py`: SVD/NMF/FA 세 가지 축 압축 기법(config.FACTOR_METHODS 에서 선택), 기존 축을 사칙연산으로 조합해 잔차를 더 잘 설명하는 축을 반복 추가하는 `engineer_axes`(matching pursuit), 축별 대표 아이템, 유저별 잔차 기반 close/far 아이템 추출
- `profiling.py`: profile_raw(이력만) / profile_axis(축 + close/far 대비 조건) 두 프로필 생성 방식
- `llm_client.py`: OpenRouter Chat Completions 호출 래퍼
- `recommend.py`: 프로필 조건으로 후보 영화 LLM 랭킹
- `evaluate.py`: Hit@k, MRR, NDCG@k
- `main.py`: 전체 파이프라인 orchestration
