# MF-Axis vs Raw-History 프로필 비교

LLM이 유저의 영화 취향 프로필을 생성할 때, **시청 이력만 주는 것**과 **축(행렬분해로 압축한 잠재축) + 그 축이 잘 반영하는(close) 영화와 못 하는(far) 영화를 대비시켜 함께 주는 것** 중 어느 쪽이 더 정확한 추천으로 이어지는지 Hit@K/MRR/NDCG@K로 비교하는 실험 코드입니다.

## 동작 방식 (파이프라인)
1. **데이터 로딩 & 필터링 & 분할** (`data.py`) — MovieLens `ratings.csv`/`movies.csv`를 읽고, `filter_users_by_history`로 이력 길이(전체 평점 수)가 `[MIN_HISTORY_LEN, MAX_HISTORY_LEN]`(CLI `--min-history-len`/`--max-history-len`, 기본 300~상한없음) 구간 밖인 유저를 제외한 뒤, 남은 유저 중 `MAX_POOL_USERS`(기본 1500)명만 무작위로 샘플링(`RANDOM_SEED` 고정)해 실제 행렬 계산에 쓸 풀을 만듦. 조건을 만족하는 유저 전체(`ml-latest` 기준 ≥300 이력 유저만 해도 26,610명)를 다 넣으면 이들이 합쳐서 본 영화가 카탈로그 대부분(8만+ 편)을 덮어 dense 평점 행렬이 감당 못 할 크기(수십 GB)가 되므로, 풀을 좁혀 유저/아이템 universe를 함께 줄인다. 이 구간 필터는 "이력이 매우 긴 유저(예: ≥300)만" 볼 때뿐 아니라, "이력이 짧은 유저(예: 20~50)만" 뽑아 raw-vs-axis 격차가 이력 길이에 따라 어떻게 달라지는지 비교하는 데도 씀(`history_len` 컬럼으로 결과에 유저별 이력 길이가 함께 저장됨). 이후 유저별 가장 최근 평점 1개를 leave-one-out test로 분리, 나머지를 train으로 사용.
2. **축 계산 + feature 엔지니어링** (`mf.py`, `axis_engineering.py`) — 어느 기법이든 **축(axis)은 입력 feature 그 자체가 아니라, feature 행렬을 압축(압축률 = `N_FACTORS`)해서 나온 결과**라는 원칙이 항상 유지됨. 새로 만들어내는 정보도 예외가 아니라 반드시 "feature로 추가 → 전체를 다시 압축"을 거친다(축끼리 직접 조합해서 압축 없이 축 목록에 얹지 않음):
   - **base feature**: `svd`/`nmf`/`fa`는 유저×영화 평점 행렬 자체, `genre`는 유저×장르 평균평점 행렬(영화-장르 멤버십 `data.build_item_genre_matrix`로 가중 집계, 안 본 장르는 유저 평균 기준 0=중립 처리)이 각 방법의 feature. `fit_svd`/`fit_nmf`/`fit_factor_analysis`/`fit_genre`가 이 feature를 `N_FACTORS`개 축으로 압축
   - **`axis_engineering.engineer_axes_llm()`**: 매 라운드 LLM에게 "지금 축들이 전체 유저 기준으로 가장 못 설명하는 영화들"(`_global_far_items`)과 현재 축 요약을 보여주고, **어떤 두 base feature를 어떤 연산(+,-,×,÷)으로 조합할지, 그리고 몇 라운드나 계속할지(CONTINUE/STOP)를 모두 LLM이 결정**함:
     - `genre`: 19개 장르 이름 전체를 후보로 보여주고 `"Action - Romance"`처럼 이름으로 자유 제안받음
     - `svd`/`nmf`/`fa`: 영화가 수천~수만 개라 이름 그대로는 후보로 보여줄 수 없어, far 영화 상위 10개로 후보를 좁히고 번호를 매겨 `"3 - 7"`처럼 번호로 응답받음(`recommend.py`의 번호 기반 랭킹과 동일한 패턴)
   - 서버가 제안을 검증 — 존재하지 않는 이름/번호나 허용 안 된 연산자를 쓰면 **그 즉시 조용히 중단**(재시도 없음, fail-safe). 방법에 따라 엔지니어링 feature가 0개일 수 있음(버그 아니라 의도된 동작)
   - 유효한 제안이면 그 두 feature의 컬럼을 실제로 사칙연산해서 **새 feature 컬럼 하나**를 만들고, 이걸 원본 feature 행렬에 이어붙여서 **그 방법으로 처음부터 다시 압축**(재적합). 압축 결과에서 새로 추가한 컬럼에 해당하는 부분은 버리고 원본 feature(영화/장르) 부분만 축으로 남기므로, 축 개수는 항상 `N_FACTORS`로 고정되고 새 feature의 영향은 압축 자체에 반영됨
   - 재적합으로 잔차가 얼마나 줄었는지(음수면 오히려 나빠졌다는 뜻 — LLM이 도움 안 되는 조합을 제안했을 수 있음)를 LLM에게 알려주고 "더 추가할지" 물어봄(`_ask_continue`). 같은 조합을 다시 제안하면(레퍼토리 소진으로 보고) 자동 중단
   - `MAX_ENGINEERED_AXES`(기본 10)는 LLM이 계속 CONTINUE를 골라도 라운드가 무한정 늘지 않게 막는 안전 상한. 매 라운드 해당 방법을 처음부터 재학습하므로(특히 `nmf`는 반복최적화라 상대적으로 느림) 라운드가 많아지면 실행 시간이 늘어날 수 있음

   이렇게 압축된 축에서 대표 영화(loading 상위 n개)를 추출하고, `user_residual_summary(mode=...)`로 유저가 평가한 영화 중 잔차 절댓값이 **작은(close, 축들이 잘 설명하는)** 영화와 **큰(far, 축들이 설명 못 하는)** 영화를 각각 상위 n개씩 뽑음.
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
set OPENROUTER_API_KEY=sk-or-v1-...
```

`ml-latest`(전체) 데이터를 https://files.grouplens.org/datasets/movielens/ml-latest.zip
에서 받아 `data/ml-latest/` 에 압축 해제 (ratings.csv, movies.csv 필요; zip 335MB, 압축 해제 시 ratings.csv만 933MB — 유저 33만/평점 3383만). `data/`는 `.gitignore`에 포함되어 있어 커밋되지 않음.

작은 데이터셋(`ml-latest-small`, 610명/10만 평점)으로 빠르게 돌리고 싶으면 https://files.grouplens.org/datasets/movielens/ml-latest-small.zip 을 받아 `data/ml-latest-small/`에 풀고 `config.DATA_DIR`을 바꾸면 됨(단, 이 경우 유저 수가 적어 `MIN_HISTORY_LEN`을 낮춰야 함 — ≥300 기준 84명뿐).

## 실행
```bash
python main.py
# N_FACTORS 민감도 스윕, 결과 파일 분리
python main.py --n-factors 5 --out results_nf5.csv
python main.py --n-factors 8 --out results_nf8.csv
# 평가 유저 수 오버라이드
python main.py --n-eval-users 300
# 짧은 이력 vs 긴 이력 유저 비교 (같은 N_FACTORS/N_EVAL_USERS로 두 그룹 각각 실행)
python main.py --min-history-len 20 --max-history-len 50 --n-eval-users 300 --out results_light.csv
python main.py --min-history-len 300 --n-eval-users 300 --out results_heavy.csv
```
`--n-factors`/`--n-eval-users`/`--out`/`--min-history-len`/`--max-history-len`은 `config.py`의 `N_FACTORS`/`N_EVAL_USERS`/`results.csv`/`MIN_HISTORY_LEN`/(상한 없음)을 일시적으로 덮어씁니다(파일은 그대로 둠). 두 그룹을 비교할 땐 결과 CSV의 `history_len` 컬럼으로 유저별 실제 이력 길이도 확인 가능.

`config.py`에서 조정 가능한 값:
- `FACTOR_METHODS`: 비교할 축 추출 기법 목록 (svd/nmf/fa/genre 중 선택, `mf.FACTORIZERS`에 정의). 네 기법 모두 `N_FACTORS`를 그대로 압축 축 개수로 씀(genre도 예외 아님 — 19개 장르 feature를 SVD로 `N_FACTORS`개로 압축)
- `N_FACTORS`: 기본 잠재 축 개수 (CLI `--n-factors`로 실행 시 오버라이드 가능)
- `MAX_ENGINEERED_AXES`: 축 엔지니어링 라운드 상한(안전장치, 기본 10) — 실제로 몇 개가 추가될지는 LLM이 CONTINUE/STOP으로 결정하며, 이 값은 LLM이 계속 CONTINUE를 골라도 과다 라운드로 새지 않게 막는 안전망
- `MIN_HISTORY_LEN`: 이 값(기본 300) 미만으로 평점을 남긴 유저는 평가 대상에서 제외 (CLI `--min-history-len`으로 오버라이드 가능)
- `MAX_HISTORY_LEN`: config 기본값은 없음(상한 없음); 짧은 이력 유저만 뽑고 싶을 때 CLI `--max-history-len`으로만 지정
- `MAX_POOL_USERS`: 이력 길이 조건을 만족하는 유저 중 행렬 계산에 실제로 쓸 유저 수 상한(기본 1500) — dense 행렬 크기를 제어하기 위한 값이라 `N_EVAL_USERS`보다는 넉넉히 커야 함
- `N_NEGATIVES`: 랭킹 후보에 섞을 negative 샘플 수
- `N_EVAL_USERS`: 평가할 유저 수 (기본 150, CLI `--n-eval-users`로 오버라이드 가능; `MAX_POOL_USERS`를 넘을 수 없음)
- `TOP_K`: Hit@K, NDCG@K의 K
- `RANDOM_SEED`: 재현성용 시드
- `MODEL`: 사용할 LLM (OpenRouter 모델 ID)

실행 결과는 `results.csv`에 유저별 `history_len`(실제 이력 길이), 프로필 텍스트(`profile_raw`, `profile_svd`, `profile_nmf`, `profile_fa`, `profile_genre` 등 `FACTOR_METHODS`에 따라 결정), 방법별 hit/mrr/ndcg 값으로 저장되며, 콘솔에는 방법별 평균 Hit@K/MRR/NDCG@K가 출력됩니다. 실행 시작 시 방법별로 `[svd] engineered features: ['Movie A - Movie B (residual -3.2%)']`처럼 LLM이 실제로 제안해 채택된 feature 조합이 로그로 출력됩니다.

## 구조
- `config.py`: API 키/모델, 데이터 경로, 실험 하이퍼파라미터
- `data.py`: MovieLens 로딩, 이력 길이 구간 필터링+풀 샘플링(`filter_users_by_history`), leave-one-out split, 평점 행렬 생성, 아이템×장르 멤버십 행렬 생성(`build_item_genre_matrix`)
- `mf.py`: SVD/NMF/FA/genre 네 가지 축 추출 기법(config.FACTOR_METHODS 에서 선택 — genre는 유저×장르 평균평점 feature를 SVD로 `N_FACTORS`개 축으로 압축). 각 `fit_*`는 `extra_features`(LLM이 엔지니어링한 feature 컬럼)를 받아 원본 feature에 이어붙여 함께 압축하되, 축 표현(H)은 항상 원본 feature 개수만큼만 남김. 축별 대표 아이템, 유저별 잔차 기반 close/far 아이템 추출
- `axis_engineering.py`: **어떤 두 feature를 어떤 연산으로 조합할지, 몇 라운드나 계속할지를 LLM이 직접 결정**하고 서버는 검증 + 재압축만 수행하는 `engineer_axes_llm` — genre는 이름 기반, svd/nmf/fa는 far 영화 번호 기반으로 후보 제시 방식이 다름, 중복 조합 재선택 방지
- `profiling.py`: profile_raw(이력만) / profile_axis(축 + close/far 대비 조건) 두 프로필 생성 방식
- `llm_client.py`: OpenRouter Chat Completions 호출 래퍼
- `recommend.py`: 프로필 조건으로 후보 영화 LLM 랭킹
- `evaluate.py`: Hit@k, MRR, NDCG@k
- `main.py`: 전체 파이프라인 orchestration
