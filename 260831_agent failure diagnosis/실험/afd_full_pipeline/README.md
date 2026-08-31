# AFD Full Pipeline — 에이전트 오류 진단가능성 측정 실험

논문 **「LLM 에이전트 오류 진단 가능성 측정 방법론」**(한국IT서비스학회지 투고본 v0.5) 부속 실험 코드.
상세 명세는 상위 폴더의 `metrics_reproduction_guide_v0.4.pdf` 참조.

## 목적

에이전트 실패의 자동 진단·귀인 방법들은 모두 "실행 로그에 판정 근거가 기록돼 있다"는 전제 위에서 작동한다.
이 파이프라인은 그 전제를 **진단가능성(diagnosability)**이라는 측정 가능한 속성으로 조작화한다.

- 4계층 26항목 체크리스트: 항목마다 (진단 질문 / 필요 로그 필드 / 판정 방식)
- 필요 필드가 없으면 그 항목은 판정불가 → 로그의 진단가능성을 정량화
- 공개 벤치마크 3종(Who&When, AgentErrorBench, TRAIL)의 실패 궤적 532건 전수에 적용

핵심 결과: 진단가능성은 과제가 아니라 **로그 형식**이 결정한다
(AEB 76.9% / TRAIL 61.5% / Who&When 42.3%. 같은 GAIA 과제가 형식에 따라 50.0% vs 61.5%).

## 파일 구성

### 스크립트

| 파일 | 역할 | 산출 |
|---|---|---|
| `normalize.py` | 이질적 벤치마크 로그 → 공통 스키마 `traces.jsonl` (532행). 궤적마다 18개 로그 필드의 가용성을 `E`(구조 필드) / `B`(자유텍스트 내 패턴 근거) / `null`(부재)로 판정 | `traces.jsonl` |
| `e1_coverage.py` | E1 진단가능성 커버리지. 항목별 필요 필드가 요구 수준으로 존재하고 서브셋 내 p≥0.9면 "진단가능" | Table 7 (`table7.json`) |
| `l3_derive.py` | 골드 주석의 최초 오류 단계 t0 분포 (위치 3분위, 전파 길이) | Table 8a (`table8_gold.json`) |
| `kappa.py` | 두 평가자(모델) 판정 시트 → Cohen's κ + Gwet AC1. `--selftest` 내장 | Table 6 성분 |
| `e3_judge.py` | E3 LLM 판정 파이프라인. 모듈군 5분할 프롬프트(M/R/P/A/T), 궤적당 5콜, temperature=0, 골드 미포함. **미실행 상태** | Table 6 / 8b / 9 |

### 데이터 · 산출물

| 파일 | 내용 |
|---|---|
| `traces.jsonl` | 공통 스키마 궤적 532행 (`normalize.py` 산출, 이후 모든 스크립트의 입력) |
| `table7.json` / `e1_report.txt` | E1 커버리지 결과 + 콘솔 캡처 |
| `table8_gold.json` / `l3_report.txt` | 골드 앵커 분포 결과 + 콘솔 캡처 |
| `pilot_gold.json` | E2 판정 파일럿용 골드 9건 (궤적 본문과 분리 — 심사자 격리) |
| `pilot_judgments.json` | 파일럿 수기 판정 결과 (최초 오류 단계·모듈·L2 항목·확신도) |
| `judg_*.jsonl` | (E3 실행 후 생성) LLM 판정 원본 |

## 사전 준비

- Python 3.12
- `pyarrow` (TRAIL parquet 읽기용) — `pip install pyarrow`
- E3 실행 시에만 `ANTHROPIC_API_KEY` (그 외 API·외부 SDK 불필요, 표준 라이브러리만 사용)
- 난수 시드: 20260817

### Windows 실행 주의 (필수)

1. **인코딩** — 모든 스크립트가 `open()`에 인코딩을 지정하지 않아, Windows 기본 코드페이지(cp949)에서는
   UTF-8 로그를 읽다가 `UnicodeDecodeError`로 죽는다. 코드 수정 없이 다음 중 하나로 해결한다:
   - 세션당 한 번 `set PYTHONUTF8=1` (cmd) / `$env:PYTHONUTF8=1` (PowerShell) / `export PYTHONUTF8=1` (bash)
   - 또는 매 명령을 `python -X utf8 ...`로 실행
2. **줄바꿈** — 줄 끝 `\` 이어쓰기는 bash 문법. cmd.exe에서는 `^`를 쓰거나 한 줄로 실행한다.
3. **`&` 이스케이프** — `Who&When` 경로의 `&`는 셸 특수문자이므로 **반드시 따옴표로 감싼다** (`"../Who&When"`).

macOS/Linux(UTF-8 로케일)에서는 위 3가지 모두 신경 쓸 필요 없다.

### 데이터 취득

| 벤치마크 | 취득 | 위치 |
|---|---|---|
| Who&When | `git clone github.com/mingyin1/Agents_Failure_Attribution` | `실험/Who&When/` 하위 `Algorithm-Generated/` (126개), `Hand-Crafted/` (58개). 파일명은 `1.json`, `2.json` … 형식 |
| AgentErrorBench | 저장소에 포함 | `실험/AgentErrorBench-20260818T084720Z-1-001/AgentErrorBench/` |
| TRAIL | 저장소에 포함 (HuggingFace `PatronusAI/TRAIL` 게이트 동의본) | `실험/gaia-*.parquet`, `실험/swe_bench-*.parquet` |

## 실행 순서

> **셸별 줄바꿈 주의**: 아래 명령은 한 줄씩 그대로 붙여넣는다. 줄 끝 `\`(백슬래시) 이어쓰기는
> bash 문법이라 **Windows cmd.exe에서는 동작하지 않는다**(`unrecognized arguments: \` 에러).
> cmd.exe에서 여러 줄로 나누려면 `\` 대신 `^`를 쓰거나, 그냥 한 줄로 실행한다.
>
> **인코딩**: cmd.exe에서 세션당 한 번 `set PYTHONUTF8=1` 실행 후 아래 명령을 그대로 쓴다
> (안 하면 `UnicodeDecodeError: 'cp949'...`). 매 명령에 `python -X utf8 ...`를 붙여도 된다.
> macOS/Linux(UTF-8 로케일)에서는 둘 다 불필요.

### 1. 정규화 → traces.jsonl

```
cd "c:\Users\user\Documents\MAS_Experiments\260831_agent failure diagnosis\실험\afd_full_pipeline"
set PYTHONUTF8=1

python normalize.py --whowhen "../Who&When" --aeb "../AgentErrorBench-20260818T084720Z-1-001/AgentErrorBench" --trail-gaia "../gaia-00000-of-00001-33a2e72d362d688a.parquet" --trail-swe "../swe_bench-00000-of-00001-91aa04220f7198b4.parquet" -o traces.jsonl
```

→ `traces.jsonl` (532행) + 서브셋별 필드 가용률(E/B 판정 감사 근거) 콘솔 출력.
이후 모든 스크립트는 이 파일 하나만 입력으로 받는다.

### 2. 무API 지표 (독립 실행, 순서 무관)

```
python e1_coverage.py traces.jsonl --p 0.9 -o table7.json
python e1_coverage.py traces.jsonl --p 0.85
python l3_derive.py traces.jsonl -o table8_gold.json
python kappa.py --selftest
```

여기까지가 현재 저장소에 산출물이 포함된 "실측 완료" 범위.
재실행 시 `table7.json` / `table8_gold.json`은 기존 커밋본과 바이트 단위로 동일하게 재현된다.

### 3. LLM 판정 (API 키 필요 · 미실행 · 약 $64~128)

```
set ANTHROPIC_API_KEY=sk-ant-...

python e3_judge.py traces.jsonl --estimate
python e3_judge.py traces.jsonl --run --model sonnet --limit 20 --out test.jsonl
python e3_judge.py traces.jsonl --run --model sonnet --out judg_sonnet.jsonl
python e3_judge.py traces.jsonl --run --model haiku --out judg_haiku.jsonl
python e3_judge.py traces.jsonl --run --model sonnet --variant b --limit 106 --out judg_vb.jsonl
python e3_judge.py traces.jsonl --aggregate judg_sonnet.jsonl judg_haiku.jsonl judg_vb.jsonl
```

- `--estimate`: 비용 견적만 (API 미사용) / `--limit 20`: 소액 점검 / `--variant b`: 자기일관성 표본
- `--run`은 출력 jsonl의 `(trace_id, group)` 기준으로 재개를 지원한다.
  중단되면 **같은 명령을 다시 실행**하면 이어서 진행되고 중복 과금이 없다.
- 마지막 `--aggregate`가 Table 6(모델 간 κ·AC1) / 8b(발생률) / 9(골드 앵커 정합)를 출력한다.

## 의존 관계

```
데이터 취득
   └─ normalize.py ──► traces.jsonl ──┬─ e1_coverage.py       → table7.json
                                      ├─ l3_derive.py         → table8_gold.json
                                      ├─ kappa.py --selftest
                                      └─ e3_judge.py --run    → judg_*.jsonl
                                            └─ --aggregate (kappa.py 내부 호출) → Table 6 / 8b / 9
```

- `normalize.py`가 항상 먼저.
- 2단계 3개는 서로 순서 무관.
- 3단계는 `--run` 3종(sonnet / haiku / variant-b)이 끝난 뒤 `--aggregate`.

## 현재 상태

| 지표 | 상태 |
|---|---|
| Table 7 커버리지 (E1) | 실측 완료 — AEB 76.9 / TRAIL 61.5 / W&W 42.3% (AG 38.5 · HC 53.8) |
| Table 8a 골드 앵커 분포 | 실측 완료 — 골드 526건 (범위 초과 등 6건 제외) |
| 판정 파일럿 (E2) | 실측 완료 — 9건, 단계 exact 4/9 · ±3 8/9 · 에이전트 4/4 · 모듈 1/5 |
| Table 6 판정 안정성 (E2) | **미실행** — `e3_judge.py --aggregate` + `kappa.py` |
| Table 8b 발생률 (E3) | **미실행** — `e3_judge.py --run` |
| Table 9 모델 민감도 | **미실행** — `e3_judge.py --aggregate` |
