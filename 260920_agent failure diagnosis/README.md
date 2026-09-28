# AFD 체크리스트 기반 오류 귀인

전체 로그만 보는 **AAO**와 체크리스트를 함께 보는 **CKL**의 실패 단계 귀인 성능을 비교한다.
체크리스트는 L0 6개·L1 3개·L2 9개이며, 필요한 L0 정보가 없으면 해당 오류는 **판단 불가**로 처리한다.

데이터 **413건**: Who&When 184 + AgentErrorBench 200 + AgentRx τ-bench 29.
원본은 `data/raw/`, 공통 입력은 `data/traces.jsonl`, 결과는 `results/ckl/`와 `results/attribution/`에 저장된다.

## 1. 준비 — Windows CMD

아래 명령은 **CMD에서 한 줄씩** 실행한다. 새 CMD 창을 열면 경로와 변수를 다시 설정한다.
`MODEL`은 호출 모델, `TAG`는 결과 파일 이름이다. 모델이나 실험 조건을 바꾸면 둘을 함께 바꾼다.

```bat
cd /d "C:\Users\sgrhe\OneDrive\문서\MAS_Experiments\260920_agent failure diagnosis"
set "PYTHONUTF8=1"
set "MODEL=openai/gpt-6-luna"
set "TAG=v3_fit_gpt6luna"
if not exist .env copy .env.example .env
notepad .env
```

`.env`에 `OPENROUTER_API_KEY`를 입력하고 저장한다. 현재 데이터 구성은 Python 표준 라이브러리만 사용한다.

```bat
python normalize.py
python test_offline.py
python checklist.py --selftest
```

## 2. 프롬프트·비용 확인

모델 추론을 호출하지 않는다. 비용 견적은 OpenRouter 모델·요금 목록을 조회한다.

```bat
python ckl_judge.py --dry-run
python ckl_judge.py --estimate --model %MODEL%
python attribute.py --method aao --estimate --model %MODEL%
```

## 3. 10건 시험 실행

아래 판정·귀인 명령은 **유료 LLM 호출**을 실행한다. 동일한 10건으로 두 방식을 비교한다.

```bat
python ckl_judge.py --run --model %MODEL% --sample 10 --out ckl_test_%TAG%.jsonl
python attribute.py --method aao --run --model %MODEL% --sample 10 --out attr_aao_test_%TAG%.jsonl
python attribute.py --method ckl --run --model %MODEL% --sample 10 --ckl-file ckl_test_%TAG%.jsonl --out attr_ckl_test_%TAG%.jsonl
python evaluate.py --ckl ckl_test_%TAG%.jsonl --attr attr_aao_test_%TAG%.jsonl attr_ckl_test_%TAG%.jsonl
```

## 4. 전체 413건 실행

판정 → AAO 귀인 → CKL 귀인 순서다. 중단되면 같은 명령을 다시 실행해 이어간다.

```bat
python ckl_judge.py --run --model %MODEL% --out ckl_%TAG%.jsonl
python attribute.py --method aao --run --model %MODEL% --out attr_aao_%TAG%.jsonl
python attribute.py --method ckl --run --model %MODEL% --ckl-file ckl_%TAG%.jsonl --out attr_ckl_%TAG%.jsonl
```

## 5. 결과 확인

단계 exact/±1/±3/±5, 에이전트·모듈 일치율, AAO–CKL 쌍대 비교를 집계한다.

```bat
python evaluate.py --ckl ckl_%TAG%.jsonl --attr attr_aao_%TAG%.jsonl attr_ckl_%TAG%.jsonl

python evaluate.py --ckl ckl_google_gemini-3.7-flash.jsonl --attr attr_aao_google_gemini-3.7-flash.jsonl attr_ckl_google_gemini-3.7-flash.jsonl

python evaluate.py --ckl ckl_%TAG%.jsonl --attr attr_aao_%TAG%.jsonl attr_ckl_%TAG%.jsonl --detail
python evaluate.py --ckl ckl_%TAG%.jsonl --coverage

python evaluate.py --ckl ckl_google_gemini-3.7-flash.jsonl --coverage
```

**기존 TRAIL 결과와 섞지 않도록** 위처럼 `v3_fit_` 태그와 파일명을 명시한다. 시험 실행과 전체 실행 파일도 분리된다.

## 자주 쓰는 옵션

| 옵션 | 용도 |
|---|---|
| `--subset AgentRx/tau_retail` | τ-bench만 실행. 위의 판정·AAO·CKL 세 명령에 모두 추가하고 별도 `TAG` 사용 |
| `--sample N` / `--limit N` | 고정 시드 무작위 N건 / 앞 N건 |
| `--max-tokens 32000` | 출력 토큰 상한 지정. 기본은 모델 최대 |
| `--ckl-layers L1,L2` | 귀인에 넣을 체크리스트 계층 선택 |
| `--ckl-view errors` | 오류가 있다고 판정된 항목만 포함 |
| `--ckl-indet` | 판단 불가 항목도 포함 |
| `--method-label ckl_err` | 변형을 별도 방식으로 집계. 출력 파일명도 변경 |

## 데이터 해석 시 참고

- 기본은 정규화된 전체 궤적을 입력한다. 컨텍스트 예산을 넘으면 로그의 앞 70%·뒤 30%를 남기고 중간을 생략하여 **계속 실행**한다. 프롬프트에 생략 표시를 넣으며 단계 번호는 바꾸지 않는다. 예산 점검은 UTF-8 바이트 수 + 2,048을 사용한 보수적 추정이다.
- API가 컨텍스트 초과를 반환해도 로그 예산을 절반씩 줄여 최대 8회 재시도한다. 지시문·출력 예산조차 들어가지 않거나 재시도 후에도 거절되면 오류를 기록한다. `usage.input_truncated_calls`·`input_omitted_chars`·`context_retries`에 절단 호출 수·생략 문자 수·재시도 수가 남는다(체크리스트는 계층별 합계).
- 절단을 사용했던 결과와 섞이지 않도록 새 `v3_fit_` 태그로 실행한다. 아래 상세 기록의 절단 설명은 과거 동작이다.

- 모든 단계 번호는 **0-based**다. AgentRx는 system/user/assistant/tool 메시지 각각이 한 단계다.
- AgentRx는 `tau_ground_truth.json` 29건의 root cause를 사용한다. 24건짜리 `tau_retail.json`과 구분한다.
- AgentRx 정답은 **최초의 회복 불가능한 결정적 실패**다. 라벨 9건이 user 메시지를 가리키며, 임의 보정 없이 유지했다. 기대 행동·출력과 실패 설명은 진단 입력에서 제외한다.
- Who&When·AEB 원본은 이전 실험 보존을 위해 복사했다. `recount_260831.py`만 과거 폴더를 참조한다.

결과 스키마, 세부 동작, 구판 보정 기록과 과거 비용 표는 [상세 기록](docs/experiment_details.md)에 보관했다. 현재 실행 명령은 이 README를 기준으로 한다.
