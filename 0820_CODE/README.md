# MF-Axis vs Raw-History 프로필 비교

LLM이 유저의 영화 취향 프로필을 생성할 때, **시청 이력만 주는 것**과 **축(행렬분해로 압축한 잠재축) + 그 축이 잘 반영하는(close) 영화와 못 하는(far) 영화를 대비시켜 함께 주는 것** 중 어느 쪽이 더 정확한 추천으로 이어지는지 Hit@K/MRR/NDCG@K로 비교하는 실험 코드입니다.

## 동작 방식 (파이프라인)
1. **데이터 로딩 & 분할** (`data.py`) — MovieLens `ratings.csv`/`movies.csv`를 읽고, 유저별 가장 최근 평점 1개를 leave-one-out test로 분리, 나머지를 train으로 사용.
2. **축 계산 + 엔지니어링** (`mf.py`) — train 평점 행렬을 `config.FACTOR_METHODS`에 지정된 기법별로 분해해 잠재 축(U, H)과 잔차를 계산. 어느 기법이든 **축(axis)은 입력 feature 그 자체가 아니라, feature를 압축(압축률 = `N_FACTORS`)해서 나온 결과**라는 원칙은 동일함:
   - `svd`/`nmf`/`fa`: 입력 feature = 유저×영화 평점 행렬. 이걸 그대로 압축해 `N_FACTORS`개의 **추상적인** 잠재 축을 얻음 (사람이 정의 X, "Axis1/Axis2/..."로 표시)
   - `genre`: 입력 feature = 유저×장르 평균평점 행렬(영화-장르 멤버십 `data.build_item_genre_matrix`로 가중 집계, 안 본 장르는 유저 평균 기준 0=중립 처리 — 실제 0점으로 오해되지 않게). 이 장르 feature 행렬을 **SVD로 다시 압축**해 `N_FACTORS`개의 축을 얻음(`fit_genre`) — 장르 하나하나가 축이 되는 게 아니라, 여러 장르의 선형결합("Action(+)+Drama(-)" 식)이 압축된 축이 됨. 대표영화/잔차 계산을 위해 이 압축 축을 다시 아이템 공간으로 투영

   이어서 `engineer_axes()`가 이렇게 얻은 축들을 **사칙연산(+, -, ×, ÷)으로 조합**한 후보를 모두 만들어보고, 현재 잔차를 가장 많이 설명하는(=잔차에 가장 크게 투영되는) 조합을 매 라운드 하나씩 찾아 축 집합에 편입시키고 잔차를 갱신 — matching pursuit 방식. 몇 개를 추가할지 미리 정해두지 않고, **이번 라운드 최선 조합의 잔차 감소율이 `MIN_AXIS_GAIN`(기본 1%) 미만이면 자동으로 멈춘다**(더 추가해도 의미 있는 정보가 없다고 판단). `MAX_ENGINEERED_AXES`(기본 10)는 무한 반복을 막는 안전장치일 뿐, 실제로는 그 전에 조기 종료되는 경우가 대부분(예: svd/fa는 보통 1라운드, genre는 이미 SVD로 압축된 축이라 더 줄일 게 없어 0라운드, nmf는 초기 적합이 상대적으로 나빠 10라운드까지 채우기도 함). 이미 선택된 조합은 재선택하지 않고, 새로 추가된 축에는 `E1`, `E2`... 같은 짧은 이름을 붙여 다음 라운드 조합 재료로 씀(조합식 텍스트를 그대로 라벨로 재사용하면 라운드가 거듭될수록 라벨이 기하급수적으로 길어지는 문제가 있어 방지). `genre` 방법의 경우 기본 축 라벨은 장르 조합("Action(+)+Drama(-)")으로 표시됨(`axis_labels`).

   이렇게 만들어진 축(기본 축 + 엔지니어링 축)에서 대표 영화(loading 상위 n개)를 추출하고, `user_residual_summary(mode=...)`로 유저가 평가한 영화 중 잔차 절댓값이 **작은(close, 축들이 잘 설명하는)** 영화와 **큰(far, 축들이 설명 못 하는)** 영화를 각각 상위 n개씩 뽑음.
3. **프로필 생성** (`profiling.py`, LLM 호출) — 평가 대상 유저마다 두 방식으로 취향 프로필 텍스트를 생성:
   - `profile_raw`: 시청 이력만 보고 LLM이 직접 서술 (baseline, KAR/ONCE 방식)
   - `profile_axis`: 잠재 축 요약(+/- 부호로 축의 두 방향 표시, `genre` 방법은 축 라벨이 그 축을 구성하는 장르 조합) + 시청 이력 + **close 영화(축이 잘 반영하는 취향)**와 **far 영화(축이 반영 못 하는 취향)**를 함께 주고, 그 대비를 근거로 기존 축이 놓친 취향을 짚어내도록 서술 — `FACTOR_METHODS`의 기법마다 하나씩 생성
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
- `FACTOR_METHODS`: 비교할 축 추출 기법 목록 (svd/nmf/fa/genre 중 선택, `mf.FACTORIZERS`에 정의). 네 기법 모두 `N_FACTORS`를 그대로 압축 축 개수로 씀(genre도 예외 아님 — 19개 장르 feature를 SVD로 `N_FACTORS`개로 압축)
- `N_FACTORS`: 기본 잠재 축 개수 (CLI `--n-factors`로 실행 시 오버라이드 가능)
- `MAX_ENGINEERED_AXES`: 축 엔지니어링 라운드 상한(안전장치, 기본 10) — 실제 추가 개수는 `MIN_AXIS_GAIN` 조건으로 그 전에 조기 종료되는 경우가 대부분
- `MIN_AXIS_GAIN`: 이번 라운드 최선 조합의 잔차 감소율이 이 값(기본 0.01=1%) 미만이면 축 추가 중단
- `N_NEGATIVES`: 랭킹 후보에 섞을 negative 샘플 수
- `N_EVAL_USERS`: 평가할 유저 수 (기본 150, CLI `--n-eval-users`로 오버라이드 가능)
- `TOP_K`: Hit@K, NDCG@K의 K
- `RANDOM_SEED`: 재현성용 시드
- `MODEL`: 사용할 LLM (OpenRouter 모델 ID)

실행 결과는 `results.csv`에 유저별 프로필 텍스트(`profile_raw`, `profile_svd`, `profile_nmf`, `profile_fa`, `profile_genre` 등 `FACTOR_METHODS`에 따라 결정)와 방법별 hit/mrr/ndcg 값으로 저장되며, 콘솔에는 방법별 평균 Hit@K/MRR/NDCG@K가 출력됩니다. 실행 시작 시 방법별로 `[svd] engineered axes: ['E1 = Axis1 + Axis3']`처럼 실제로 몇 개가, 어떤 조합으로 추가됐는지 로그로 출력됩니다.

## 구조
- `config.py`: API 키/모델, 데이터 경로, 실험 하이퍼파라미터
- `data.py`: MovieLens 로딩, leave-one-out split, 평점 행렬 생성, 아이템×장르 멤버십 행렬 생성(`build_item_genre_matrix`)
- `mf.py`: SVD/NMF/FA/genre 네 가지 축 추출 기법(config.FACTOR_METHODS 에서 선택 — genre는 유저×장르 평균평점 feature를 SVD로 `N_FACTORS`개 축으로 압축), 기존 축을 사칙연산으로 조합해 잔차를 더 잘 설명하는 축을 잔차 감소율이 임계치 밑으로 떨어질 때까지 반복 추가하는 `engineer_axes`(matching pursuit, 중복 조합 재선택 방지, 라벨 길이 폭발 방지), 축별 대표 아이템, 유저별 잔차 기반 close/far 아이템 추출
- `profiling.py`: profile_raw(이력만) / profile_axis(축 + close/far 대비 조건) 두 프로필 생성 방식
- `llm_client.py`: OpenRouter Chat Completions 호출 래퍼
- `recommend.py`: 프로필 조건으로 후보 영화 LLM 랭킹
- `evaluate.py`: Hit@k, MRR, NDCG@k
- `main.py`: 전체 파이프라인 orchestration
