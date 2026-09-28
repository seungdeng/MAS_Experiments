> 보관용 상세 기록입니다. 현재 실행 명령과 경로는 [README](../README.md)를 기준으로 합니다.

# AFD Checklist Attribution — 체크리스트 3단계 판정 + 오류 귀인 (260920)

260831 파이프라인(진단가능성 커버리지 측정)의 후속 실험. 체크리스트가 **L0 6 / L1 3 / L2 9항목**으로 개편되어 새로 구성했다.
LLM 호출은 **OpenRouter**(여러 모델 비교), 프롬프트는 **영어**(데이터셋이 영어), 설정은 **`.env`**.

## 현재 데이터 구성 (TRAIL → AgentRx τ-bench)

**413건: Who&When 184 + AgentErrorBench 200 + AgentRx τ-bench 29.**
원본은 모두 이 폴더의 `data/raw/`에 있다. 기본 실험은 이전 `260831` 폴더에 의존하지 않는다.

```text
data/
├─ raw/
│  ├─ WhoWhen/                 # Algorithm-Generated/, Hand-Crafted/
│  ├─ AgentErrorBench/         # Label/, Original_Failure_Trajectory/
│  └─ AgentRx/                # 내려받은 원본 전체
│     ├─ tau_retail/tau_dataset_failed.json
│     └─ ground_truth/tau_ground_truth.json
└─ traces.jsonl
```

- `python normalize.py`: 세 벤치마크를 합쳐 기본 입력을 생성한다. 기본 원본 경로는 스크립트 위치 기준이다.
- `python normalize.py --agentrx data/raw/AgentRx -o data/tau_only.jsonl`: τ-bench만 생성한다. 원본 옵션을 하나라도 지정하면 지정한 원본만 사용한다.
- `--subset AgentRx/tau_retail`: 판정·귀인을 τ-bench로 제한한다.
- 실패 로그 **29건**과 `tau_ground_truth.json` **29건**을 일대일 연결한다. `tau_retail.json`은 24건짜리 다른 라벨 파일이므로 사용하지 않는다.
- 골드는 `root_cause.failure_id`가 가리키는 실패다. system/user/assistant/tool 메시지를 모두 단계로 유지하고, 원본 `index`를 0-based로 매핑한다. 도구 호출 인자·응답·연결 ID도 보존한다.
- 제공 라벨 중 **9건은 root cause 단계가 user 메시지를 가리킨다**. 번호를 임의 보정하지 않았다. 결과 해석 시 이 라벨 특성을 밝혀야 한다. `gold.source_step`과 각 단계의 `source_index`로 대조할 수 있다.
- AgentRx 귀인 정의는 **최초의 회복 불가능한 결정적 실패**다. 앞선 회복 가능한 오류를 정답으로 취하지 않는다. 단계 정확도를 평가하며 단일 에이전트의 이름 일치율은 집계하지 않는다.
- 정책은 원본 system 메시지에 포함된다. 숨겨진 사용자 시뮬레이터 지시, 기대 행동·출력, reward 정보, 실패 설명은 진단 입력에서 제외한다. 렌더링 길이 제한은 기존 공통 설정을 따른다.
- Who&When·AEB는 이전 실험 보존을 위해 복사했다. TRAIL parquet과 구판 재집계 입력은 `260831`에 남아 있다. `recount_260831.py`만 의도적으로 과거 경로를 참조한다.
- 기존 `results/`는 TRAIL 포함 실험 결과다. 새 실행은 별도 출력 이름을 사용한다. 예를 들어 `--out ckl_v2_<model>.jsonl`, `--out attr_aao_v2_<model>.jsonl`, `--out attr_ckl_v2_<model>.jsonl`로 저장하고, 귀인 입력과 평가 glob도 해당 파일만 지정한다. 기존 전체 표본의 `--sample N`은 새 표본과 다르다.
- 아래 260831 비교 및 TRAIL 비용 표는 과거 기록이다. 현재 비용은 새 데이터로 `--estimate`를 실행해 다시 계산한다.

## 무엇을 하는가

```
data/traces.jsonl ─► ckl_judge.py ─► results/ckl/ckl_<model>.jsonl ─┐
   (413건)            L0 → L1 → L2 (계층당 1콜)                  ├─► attribute.py --method ckl ─┐
      │                                                      │                              ├─► evaluate.py
      └────────────────────────────────────────────────────► attribute.py --method aao ───┘
                                       (베이스라인: 로그만)      → results/attribution/attr_*.jsonl
```

1. **체크리스트 판정 (`ckl_judge.py`)** — 궤적마다 LLM을 L0, L1, L2 각 한 번씩 호출한다.
   - **L0**: 로그에 6개 정보(과제 명세·단계 경계·모듈별 출력·행동 결과·원시 도구 응답·시스템 정보)가 *기록돼 있는가*.
   - **L1/L2**: 시스템 오류 3개 / 인지 모듈 오류 9개가 *발생하는가*.
2. **종속(상속) 로직** — L1/L2 항목은 표의 "요구 L0"에 종속된다. **요구 L0가 하나라도 충족되지 않으면(false·null·응답 누락) 그 항목은 LLM 답과 무관하게 `판단 불가(INDETERMINATE)`** 이며 "오류 없음"으로 세지 않는다.
   요구 L0가 모두 충족된 항목만 LLM에 묻고, 물을 항목이 없는 계층은 호출을 생략한다 (`checklist.py`).
3. **오류 귀인 (`attribute.py`)** — 결정적 오류 단계(+ Who&When은 에이전트, AEB는 모듈)를 지목.
   - `aao` 베이스라인: 전체 로그만.
   - `ckl` 제안: 전체 로그 + **체크리스트 결과(L0 기록 여부 + 판단 가능했던 L1/L2 항목)**를 같은 프롬프트에 포함. 계층·표시 범위는 인자로 조절(아래 "귀인 방식 · 옵션").
4. **평가 (`evaluate.py`)** — 골드 대비 단계 정확일치/±1/±3/±5, 에이전트·모듈 일치율, 같은 모델의 ckl vs aao 쌍대 비교(McNemar).

## 파일

| 파일 | 역할 |
|---|---|
| `checklist.py` | 18항목 정의(한글=논문 정의, `EN`=프롬프트용 영어), 종속 관계, 상태 로직. `--selftest` |
| `normalize.py` | 벤치마크 3종 → `data/traces.jsonl` (413건, UTF-8 안전) |
| `common.py` | 로그 렌더링, OpenRouter 클라이언트, `.env` 로딩, 캐시, 재개. `python common.py`로 환경 점검 |
| `ckl_judge.py` | L0→L1→L2 판정 → `results/ckl/` |
| `attribute.py` | 최종 귀인 (`--method aao|ckl`) → `results/attribution/` |
| `evaluate.py` | 귀인 성능 / 체크리스트 판정 통계 (기본은 요약, `--detail`은 전체 표). `--ckl` 파일이 2개 이상이면 **판정 모델 간 일치(일치율·κ·AC1)** 표가 추가된다 |
| `test_offline.py` | API 없이 배선 검증 (가짜 LLM) |
| `recount_260831.py` | 구판(260831) 결과 중 사후 보정 가능한 표(Table 8·9)를 다시 센다. LLM 호출 없음 → `results/recount_260831/` |
| `.env.example` | 설정 템플릿 (`.env`는 git 제외) |

현재 구성은 표준 라이브러리만 사용한다. 과거 TRAIL 옵션을 직접 사용할 때만 `pyarrow`가 필요하다.

## 폴더 구조

```
260920_agent failure diagnosis/
├─ README.md
├─ .env / .env.example / .gitignore
├─ checklist.py  normalize.py  common.py          ← 스크립트는 루트에 둔다 (명령어가 짧다)
├─ ckl_judge.py  attribute.py  evaluate.py
├─ recount_260831.py  test_offline.py
├─ data/
│   └─ traces.jsonl                                ← normalize.py 산출 (413건, git 제외)
└─ results/                                        ← 모든 실험 결과
    ├─ ckl/               ckl_<모델슬러그>.jsonl          체크리스트 판정 (ckl_judge.py)
    ├─ attribution/       attr_<방식>_<모델슬러그>.jsonl  귀인 결과 (attribute.py)
    └─ recount_260831/    recount_260831_report.txt, .json  구판 결과 사후 보정
```

- **경로 기본값**: 입력은 `data/traces.jsonl`, 출력은 `results/<종류>/`. 그래서 대부분의 명령에 경로를 쓰지 않는다.
- **파일명만 써도 된다**: `--out ckl_test.jsonl`은 `results/ckl/ckl_test.jsonl`로 저장되고, `--ckl-file ckl_x.jsonl`·`--ckl ckl_*.jsonl`·`--attr attr_*.jsonl`은 `results/` 아래에서 찾는다. 디렉터리가 들어간 경로를 주면 그대로 쓴다.
- 예전처럼 `traces.jsonl`을 위치 인자로 줘도 `data/`에서 찾아 준다.

## 준비

```
copy .env.example .env        # 이미 만들어져 있음. OPENROUTER_API_KEY 만 채우면 된다
python common.py --model anthropic/claude-sonnet-5 --search haiku
python test_offline.py
python checklist.py --selftest
```

`.env`

| 키 | 의미 |
|---|---|
| `OPENROUTER_API_KEY` | 필수 (https://openrouter.ai/keys) |
| `DEFAULT_MODEL` | `--model` 생략 시 모델 슬러그 |
| `OPENROUTER_BASE_URL` / `OPENROUTER_HTTP_REFERER` / `OPENROUTER_APP_TITLE` | 선택 |

모델은 OpenRouter 슬러그로 지정한다 (예: `anthropic/claude-sonnet-5`, `anthropic/claude-haiku-4.5`, `openai/gpt-5.6-luna`, `deepseek/deepseek-v4-flash-0731`). `python common.py --search <텍스트>`로 검색.

**모델별 파라미터는 자동 조절된다.** OpenRouter 모델 목록의 `supported_parameters`를 읽어 `temperature`(기본 0), `response_format=json_object`, `reasoning`을 **지원하는 모델에만** 보낸다
(예: `anthropic/claude-sonnet-5`는 temperature 미지원 → 자동 제외). 실행 시작 시 `[llm] ... params sent: ...` 로 확인할 수 있다.

## 실행 순서 (Windows cmd — 한 줄씩, `\` 이어쓰기 불가)

```
cd "C:\Users\user\Documents\MAS_Experiments\260920_agent failure diagnosis"

:: 0) 데이터 → data/traces.jsonl (이미 생성돼 있음. 다시 만들려면)
python normalize.py

:: 1) 프롬프트 확인 / 비용 견적 (API 미사용)
python ckl_judge.py --dry-run
python ckl_judge.py --estimate --model anthropic/claude-sonnet-5

:: 2) 소규모 점검 (10건) — 판정 → 귀인 → 평가
python ckl_judge.py --run --model anthropic/claude-sonnet-5 --sample 10 --out ckl_test.jsonl
python attribute.py --method aao --model anthropic/claude-sonnet-5 --sample 10 --run --out attr_aao_test.jsonl
python attribute.py --method ckl --ckl-file ckl_test.jsonl --model anthropic/claude-sonnet-5 --run --out attr_ckl_test.jsonl
python evaluate.py --ckl ckl_test.jsonl --attr attr_*_test.jsonl

:: 3) 본 실행 (모델별로 반복. 중단되면 같은 명령 재실행 → 이어서)
python ckl_judge.py --run --model anthropic/claude-sonnet-5
python attribute.py --method aao --model anthropic/claude-sonnet-5 --run
python attribute.py --method ckl --ckl-file ckl_anthropic_claude-sonnet-5.jsonl --model anthropic/claude-sonnet-5 --run

:: 4) 집계
python evaluate.py --ckl ckl_*.jsonl
python evaluate.py
python evaluate.py --detail          :: 서브셋별·항목별 전체 표 (기본 출력은 한 화면 요약)
```

`--sample N`은 시드 고정 무작위(같은 N이면 같은 궤적), `--limit N`은 앞 N건, `--subset WhoWhen/AG`(반복 가능)로 일부 서브셋만.
체크리스트 판정을 만든 모델과 귀인 모델은 달라도 된다 (`--ckl-file`의 모델 ≠ `--model`). 결과 행에 `ckl_model`이 기록된다.

## 판정 결과 형식

`results/ckl/ckl_<모델>.jsonl` (궤적당 1행)

```json
{"trace_id": "...", "model": "...",
 "l0": {"L0-03": {"v": true, "why": "..."}, ...},
 "asked": {"L1": ["L1-01"], "L2": []},
 "l1": {"L1-02": {"status": "INDETERMINATE", "reason": "dep:L0-03", "step": null, "why": ""}, ...},
 "l2": {"L2-P2": {"status": "ERROR", "reason": null, "step": 5, "why": "..."}, ...},
 "usage": {"cost": 0.031, ...}, "errors": {"L2": "json_parse_failed"}}
```

- `status`: `ERROR` / `NO_ERROR` / `INDETERMINATE` (`reason`: `dep:<미충족 L0>` 또는 `llm`).
- `errors`가 있는 행은 재실행하면 그 궤적만 다시 호출된다. `usage.cost`는 OpenRouter가 응답에 담아주는 실제 청구액(USD).

**다른 베이스라인 결과를 함께 비교**하려면 `results/attribution/`에 `attr_*.jsonl` 형식으로 저장하면 된다:
`{"trace_id","benchmark","subset","method":"<이름>","model":"<이름>","pred":{"step":int,"agent":...,"module":...}}`

## 귀인 방식 · 옵션 (바꿔가며 실험하는 부분)

`attribute.py`의 방식은 등록표 `METHODS`로 관리한다. 프롬프트 골격·로그 렌더링은 같고 **추가 블록만 다르다.**

| 인자 | 의미 | 기본 |
|---|---|---|
| `--method aao` | 베이스라인: 로그만 | |
| `--method ckl` | 로그 + 체크리스트 결과 블록 (`--ckl-file` 필요) | |
| `--ckl-layers L0,L1,L2` | 블록에 넣을 계층 | 전부 |
| `--ckl-view applicable\|errors` | 판단 가능했던 L1/L2 항목(오류+오류없음) / 오류 항목만 | applicable |
| `--ckl-indet` | 판단 불가 항목도 `not assessable (missing L0-03)`로 표시 | 끔(생략) |
| `--method-label 이름` | 결과에 기록되는 방식 이름. 변형마다 다르게 줘야 `evaluate.py`가 구분한다 | method |

블록 형식 (각 줄이 `질문 -> 답`):
```
Log properties (what this log records):
- L0-03: Are the outputs of each module ... recorded separately at each step? -> NO. Evidence: ...
Error findings:
- L2-P2: Does the agent plan an action that is infeasible ...? (C-P2) -> ERROR at step 5. Evidence: ...
- L2-A1: ... -> no error found.
```
- L0는 yes/no 모두 넣는다(null은 생략). L1/L2의 **판단 불가는 기본적으로 생략**한다.
- 변형 예: `python attribute.py --method ckl --ckl-file ckl_x.jsonl --ckl-layers L1,L2 --ckl-view errors --method-label ckl_err --run`
- **새 방식 추가**: `attribute.METHODS`에 `이름: (체크리스트 필요 여부, 추가 블록을 만드는 함수)` 한 줄을 넣으면 `--method` 선택지에 자동 반영된다.
  다른 베이스라인(step-by-step 등)은 호출 구조가 다르므로 별도 스크립트로 만들고, 아래 형식으로 저장하면 `evaluate.py`가 함께 비교한다.

## 결정 사항 · 주의

1. **결정적 오류의 정의(확정)** — "실패로 이어진 가장 이른 오류 (the earliest error that led to the failure)". AgentRx는 위의 회복 불가능한 실패 정의를 사용하고, 나머지는 이 정의가 프롬프트에 들어간다. 벤치마크별 골드 정의(최초 실패/반사실 등)는 다르므로 결과 해석 시 함께 밝힌다.
2. **베이스라인은 all-at-once만 구현** — Who&When의 step-by-step, binary search, AgentDebug 등은 없다.
3. **단계 번호는 0-based로 통일** — 로그의 `[step N]`, 골드, 예측 모두 0-based.
4. **재현성** — 지원 모델은 temperature 0이지만, 미지원 모델(예: Sonnet 5)은 비결정적이다. OpenRouter는 같은 모델도 제공자를 바꿀 수 있다.
5. **출력 토큰 상한은 기본이 "모델 최대"** — `--max-tokens max`(기본)는 OpenRouter가 알려주는 모델의 최대 출력(예: Sonnet 5 128K, gpt-5-mini 128K, Haiku 4.5 64K)을 쓰고, 프롬프트와 합쳐 컨텍스트를 넘으면 그만큼만 줄인다. 숫자를 주면 그 값으로 고정된다. 주의: OpenRouter는 `max_tokens`만큼 크레딧을 먼저 잡아 두므로 잔액이 적으면 402가 나온다(안내 메시지가 뜬다 → 잔액을 채우거나 `--max-tokens 32000` 등으로 낮춘다). 모델이 폭주하면 비용 상한은 "최대 출력 × 단가"다.

## 260831 대비 바뀐 점과 구판 결과에 미치는 영향

구판(`260831_.../afd_full_pipeline`)의 코드에는 아래 문제가 있었고, 신판은 모두 고쳤다.
**구판 E3(4개 모델 판정, 09-14 커밋)는 이 코드로 이미 실행되었으므로 일부 결과에 영향이 있다.** (아래 "영향" 열은 재계산으로 확인한 내용이며, 재계산한 원래 수치는 `실험결과_정리.txt`와 정확히 일치한다.)

| 구판의 문제 | 신판 | 구판 결과에 미치는 영향 |
|---|---|---|
| AEB 단계 번호가 1-based(`i=len+1`)인데 골드는 0-based | 세 벤치마크 모두 0-based | **Table 8(L3-01 정답 정합)의 AEB 부분이 1칸 어긋남.** 사후 보정 완료(아래 "구판 결과 처리 방침") |
| AEB 관찰(obs)을 "다음 user 메시지 앞 400자"로 취함 → 실제로는 프롬프트 서두(`You are an expert agent...`)만 들어감 | 환경 관찰만 정규식으로 추출 | AEB 판정은 실제 관찰을 보지 못한 채 이루어짐. Table 7(모델 간 일치)·11(발생률)·8에 미친 크기는 **미측정 — 재실행해야 알 수 있음** |
| 스텝당 1,200자 절단 → 긴 assistant 턴의 끝(`<action>`)이 잘림 | 스텝 6,000자·관찰 2,000자(머리 70%+꼬리 30% 보존). 개별 스텝이 잘리는 비율: Who&When 1.7%, AEB 13.2%, TRAIL 0% | 위와 같음 (미측정) |
| TRAIL 스팬은 logs 본문만 보존 → 모델·토큰·지연·`max_steps`가 판정 입력에서 사라짐 | 스팬 이름·상태·지속시간·모델/토큰/`max_steps`·입출력 렌더링 | TRAIL 판정(특히 시스템 정보 관련)에 영향 가능 (미측정). E1(Table 9)은 원본 필드로 계산되어 무관 |
| E1의 AEB 모듈 태그 검출이 `content + obs` 전체에서 이루어짐 → obs(=프롬프트 템플릿)에 들어 있는 `<plan>` 등 안내문까지 "기록됨"으로 셈 | (신판은 E1 대신 L0를 LLM이 판정) | **Table 9의 AEB-GAIA가 과대 산정됨**(45.8% → 33.3%). 사후 보정 완료(아래) |
| 모든 `open()`이 기본 인코딩 → Windows에서 `PYTHONUTF8=1` 필요 | UTF-8 명시 | 없음 |
| `temperature=0` 고정 → 일부 최신 모델에서 400 | 지원 모델에만 전송 | 없음 |

### 구판 결과 처리 방침 (확정)

| 구판 표 | 처리 |
|---|---|
| Table 8 (L3-01 정답 정합) | **사후 보정** — AEB 예측을 −1 (재실행 불필요) |
| Table 9 (E1 커버리지) | **사후 보정** — AEB 모듈 태그를 에이전트 출력 + 실제 관찰에서만 검출해 같은 규칙으로 재산출 |
| Table 7 (모델 간 일치)·Table 11 (발생률) | 판정 입력이 열화돼 사후 보정 불가 → **새 체크리스트·새 파이프라인으로 재실행하여 대체** |

```
python recount_260831.py            :: 구판 폴더는 읽기만 함. 산출: results/recount_260831/recount_260831_report.txt, recount_260831.json
```

보정 결과 (구판 → 보정):

| 모델 | n | exact | ±1 | ±3 | ±5 |
|---|---|---|---|---|---|
| claude-haiku | 331 | 15.4 → 14.8 | 39.3 → 39.9 | 58.9 → 57.1 | 70.4 → 68.3 |
| deepseek-flash | 159 | 8.2 → 14.5 | 39.6 → 38.4 | 60.4 → 59.7 | 69.8 → 68.6 |
| gemini-flash | 294 | 28.9 → 32.0 | 46.9 → 47.6 | 67.0 → 64.6 | 75.9 → 74.1 |
| gpt5-mini | 179 | 8.9 → 13.4 | 39.1 → 40.8 | 60.9 → 62.6 | 72.6 → 70.9 |

- 위 표의 "구판" 열은 이 스크립트가 재현한 값이며 `실험결과_정리.txt`의 Table 8과 일치함을 스크립트가 검증한다(불일치하면 경고).
- AEB만 보정 대상이라 Who&When·TRAIL 행은 그대로다. 벤치마크별 분해는 `results/recount_260831/recount_260831_report.txt` 참조.
- 가정: 판정 LLM이 AEB 로그에 표시된 1-based `[step k]` 라벨을 그대로 답했다.
- Table 9: AEB-GAIA만 변한다. **45.8%(11/24) → 33.3%(8/24)**, L2 판정 가능 항목 4→1개(L2-P1~P3 탈락. plan 태그가 GAIA 50건 중 5건에서 에이전트 출력에는 없었음). ALFWorld·WebShop·AEB 전체·TRAIL·Who&When은 변화 없음.

## 과거 TRAIL 포함 532건의 비용 견적 (현재 구성에 적용하지 않음)

| 모델 | 판정(L0~L2, 상한) | 귀인 1방식 |
|---|---|---|
| `anthropic/claude-sonnet-5` | 캐시 적중 시 ≈ $31 / 미적중 ≈ $55 | ≈ $17 |
| `deepseek/deepseek-v4-flash-0731` | ≈ $0.6 ~ $1 | ≈ $0.3 |

정확한 값은 `--estimate`로 확인한다. 실행 중에는 진행 로그에 실제 누적 청구액이 표시된다.
