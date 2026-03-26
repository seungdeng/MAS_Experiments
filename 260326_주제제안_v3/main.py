"""
멀티에이전트 상호작용 로그 생성기 - 메인 실행 스크립트

GAIA, GSM8K, MATH 데이터셋의 문제를 멀티에이전트 파이프라인으로 풀고,
1.json 형식의 상호작용 로그를 기록합니다.

사용법:
    python main.py --provider openai --model gpt-4o-mini --dataset gsm8k --max-problems 10
    python main.py --provider google --model gemini-2.0-flash --dataset gsm8k --max-problems 5
    python main.py --provider ollama --model ollama/llama3 --dataset all --max-problems 10
"""
import json
import os
import re
import time
import traceback
from pathlib import Path

from config import get_args, build_llm_config
from agents import build_group_chat


# ============================================================
# 데이터 로딩
# ============================================================

def load_dataset(dataset_name, data_dir="data"):
    """
    데이터셋을 로드하고 공통 형식으로 정규화합니다.

    Args:
        dataset_name: 로드할 데이터셋 이름 ("gsm8k", "math", "gaia", "all")
        data_dir: 데이터 파일이 위치한 디렉터리 (기본값: "data")

    Returns:
        list[dict]: 정규화된 문제 리스트. 각 항목은 다음 키를 가짐:
            - task_id: 문제 고유 ID
            - question: 문제 텍스트
            - ground_truth: 정답
            - level: 난이도 레벨
            - source: 데이터셋 출처 ("GSM8K", "MATH", "GAIA")
    """
    items = []

    # ── GSM8K 로드 (초등 수학 문제, 8,792개) ──
    if dataset_name in ("gsm8k", "all"):
        path = os.path.join(data_dir, "GSM8K.json")
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for i, item in enumerate(raw):
            items.append({
                "task_id": item.get("task_id", str(i + 1)),
                "question": item["question"],
                "ground_truth": str(item["answer"]).strip(),
                "level": "gsm8k",
                "source": "GSM8K",
            })

    # ── MATH 로드 (고급 수학 벤치마크, 12,500개) ──
    # 필드: problem(문제), answer(정답), subject(주제), level(난이도 1~5)
    if dataset_name in ("math", "all"):
        path = os.path.join(data_dir, "MATH.json")
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for item in raw:
            items.append({
                "task_id": item.get("task_id", ""),
                "question": item["problem"],       # MATH는 "problem" 키 사용
                "ground_truth": str(item["answer"]).strip(),
                "level": str(item.get("level", "")),
                "source": "MATH",
            })

    # ── GAIA 로드 (다양한 추론 문제, 165개) ──
    # 필드: task_id, Question(문제), Final answer(정답), Level(난이도)
    if dataset_name in ("gaia", "all"):
        path = os.path.join(data_dir, "GAIA.json")
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for item in raw:
            items.append({
                "task_id": item["task_id"],
                "question": item["Question"],       # GAIA는 "Question" 키 사용 (대문자)
                "ground_truth": str(item["Final answer"]).strip(),
                "level": str(item.get("Level", "")),
                "source": "GAIA",
            })

    return items


# ============================================================
# 답변 추출 및 비교
# ============================================================

def extract_final_answer(chat_history):
    """
    에이전트 대화 기록에서 최종 답변을 추출합니다.

    FinalAnswer 에이전트가 "FINAL ANSWER: <답변>" 형식으로 출력한 답변을
    정규표현식으로 찾아 반환합니다. 찾지 못하면 마지막 메시지에서 추출을 시도합니다.
    """
    # 대화 기록을 역순으로 탐색하여 "FINAL ANSWER:" 패턴을 찾음
    for msg in reversed(chat_history):
        content = msg.get("content", "")
        if content is None:
            continue

        # "FINAL ANSWER: <답변>" 패턴 매칭 (대소문자 무시)
        match = re.search(r"FINAL\s*ANSWER\s*:\s*(.+?)(?:\n|TERMINATE|$)", content, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            answer = answer.strip('"\'`*.')   # 따옴표, 백틱 등 서식 문자 제거
            return answer

    # 폴백: TERMINATE가 없는 마지막 메시지에서 짧은 답변을 추출
    for msg in reversed(chat_history):
        content = msg.get("content", "")
        if content and "TERMINATE" not in content:
            lines = content.strip().split("\n")
            last_line = lines[-1].strip()
            if len(last_line) < 100:  # 합리적인 답변 길이인 경우만
                return last_line.strip('"\'`*.')
            break

    return ""


def normalize_answer(answer):
    """
    답변을 비교 가능한 형태로 정규화합니다.

    MATH 데이터셋 12,500개 답변 분석 결과 반영:
    - 72.3%: 순수 숫자/텍스트 (예: "2", "18", "1.36")
    - 27.7%: LaTeX 수식 포함 (2,121개 \\frac, 687개 \\sqrt, 414개 ^ 등)

    처리 순서:
    1. \\boxed{}, \\text{}, \\mbox{} 등 래퍼 제거
    2. \\frac, \\dfrac, \\tfrac, \\cfrac → a/b 변환
    3. \\sqrt → sqrt() 변환
    4. \\left, \\right, 간격 명령(\\!, \\,, \\;) 제거
    5. 수학 기호 변환 (\\cdot, \\times, \\pm, \\infty, \\pi 등)
    6. \\cup, \\cap 등 집합 연산자 변환
    7. \\$, \\% 등 이스케이프 문자 처리
    8. 중괄호 제거, 소문자 변환, 공백 정리
    """
    if answer is None:
        return ""
    answer = str(answer).strip()

    # ── 1단계: 래퍼 명령 제거 ──
    # \boxed{...} → 내용만 추출
    answer = re.sub(r'\\boxed\{(.+)\}', r'\1', answer)
    # \text{}, \textbf{}, \mathrm{}, \mathbf{}, \mbox{}, \textnormal{} → 내용만
    answer = re.sub(
        r'\\(?:text|textbf|textnormal|mathrm|mathbf|mathit|operatorname|mbox)\{([^}]*)\}',
        r'\1', answer
    )
    # \overline{...} → 내용만 (예: \overline{CD} → CD)
    answer = re.sub(r'\\overline\{([^}]*)\}', r'\1', answer)

    # ── 2단계: 분수 변환 ──
    # \frac{a}{b}, \dfrac{a}{b}, \tfrac{a}{b}, \cfrac{a}{b} → a/b
    answer = re.sub(r'\\[dtc]?frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', answer)

    # ── 3단계: 루트 변환 ──
    # \sqrt[n]{x} → x^(1/n)  (n제곱근)
    answer = re.sub(r'\\sqrt\[([^]]+)\]\{([^}]+)\}', r'(\2)^(1/\1)', answer)
    # \sqrt{x} → sqrt(x)
    answer = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', answer)
    # \sqrt2 같은 약식 (중괄호 없는 경우)
    answer = re.sub(r'\\sqrt(\d)', r'sqrt(\1)', answer)

    # ── 4단계: 간격/서식 명령 제거 ──
    answer = answer.replace('\\left', '')
    answer = answer.replace('\\right', '')
    answer = answer.replace('\\!', '')     # 87개: 음수 간격 (예: 10,\!010)
    answer = answer.replace('\\,', '')     # 4개: 작은 간격
    answer = answer.replace('\\;', '')     # 2개: 중간 간격
    answer = answer.replace('\\:', '')
    answer = answer.replace('\\quad', ' ')
    answer = answer.replace('\\qquad', ' ')
    answer = answer.replace('{,}', ',')   # 9개: 숫자 내 쉼표 (예: 10{,}000)

    # ── 5단계: 수학 기호 변환 ──
    answer = answer.replace('\\cdot', '*')
    answer = answer.replace('\\times', '*')
    answer = answer.replace('\\div', '/')
    answer = answer.replace('\\pm', '±')
    answer = answer.replace('\\mp', '∓')
    answer = answer.replace('\\leq', '<=')
    answer = answer.replace('\\geq', '>=')
    answer = answer.replace('\\le', '<=')
    answer = answer.replace('\\ge', '>=')
    answer = answer.replace('\\neq', '!=')
    answer = answer.replace('\\ne', '!=')
    answer = answer.replace('\\approx', '≈')
    answer = answer.replace('\\infty', 'infty')   # 138개
    answer = answer.replace('\\pi', 'pi')          # \pi → pi

    # ── 6단계: 집합/논리 연산자 ──
    answer = answer.replace('\\cup', 'U')    # 84개: 합집합
    answer = answer.replace('\\cap', '∩')
    answer = answer.replace('\\setminus', '\\')
    answer = answer.replace('\\subset', '⊂')
    answer = answer.replace('\\in', '∈')

    # ── 7단계: 이스케이프 문자 ──
    answer = answer.replace('\\$', '$')    # 61개
    answer = answer.replace('\\%', '%')    # 47개
    answer = answer.replace('\\&', '&')

    # ── 8단계: 삼각/기타 함수 (이름만 유지) ──
    # \sin, \cos, \tan, \sec, \csc, \cot, \log, \ln 등
    answer = re.sub(r'\\(sin|cos|tan|sec|csc|cot|log|ln|exp|arcsin|arccos|arctan)\b', r'\1', answer)

    # ── 9단계: 환경 제거 ──
    # \begin{pmatrix}...\end{pmatrix} 등
    answer = re.sub(r'\\begin\{[^}]+\}', '', answer)
    answer = re.sub(r'\\end\{[^}]+\}', '', answer)

    # ── 10단계: 남은 LaTeX 명령 제거 ──
    # \circ, \theta, \alpha 등 → 이름만 유지
    answer = re.sub(r'\\([a-zA-Z]+)', r'\1', answer)

    # ── 11단계: 최종 정리 ──
    answer = answer.replace('{', '').replace('}', '')   # 중괄호 제거
    answer = answer.strip('"\'`*.')
    answer = answer.lower().strip()
    answer = re.sub(r'\s+', ' ', answer)   # 연속 공백 → 단일 공백

    return answer


def eval_math_expr(s):
    """
    정규화된 수학 표현식을 숫자로 평가합니다.

    처리 가능한 형식:
    - 단순 숫자: "6.375"
    - 분수: "51/8" → 6.375
    - sqrt 표현: "sqrt(10)" → 3.162...
    - 혼합: "3sqrt(3)" → 3 * 1.732... = 5.196...
    - 음수 분수: "-3/8" → -0.375
    """
    try:
        s = s.strip().replace(",", "").replace(" ", "")

        # 단순 float 변환 시도
        return float(s)
    except ValueError:
        pass

    try:
        import math

        # "sqrt(x)" → math.sqrt(x) 로 변환
        expr = s.replace("sqrt", "math.sqrt")
        # "pi" → math.pi
        expr = expr.replace("pi", str(math.pi))

        # "3sqrt(3)" 같은 패턴 → "3*math.sqrt(3)"
        expr = re.sub(r'(\d)(math\.sqrt)', r'\1*\2', expr)

        # "2/3" 같은 분수 계산 허용 (eval 사용, 안전한 범위에서만)
        # 허용 문자: 숫자, +, -, *, /, (, ), ., math.sqrt, math.pi
        allowed = set("0123456789+-*/().emath.sqrtpi ")
        if all(c in allowed for c in expr.replace("math.sqrt", "").replace("math.pi", "")):
            result = eval(expr)
            return float(result)
    except Exception:
        pass

    return None


def check_answer(predicted, ground_truth):
    """
    예측 답변과 정답을 비교하여 정답 여부를 판단합니다.

    비교 방법 (우선순위 순):
    1. 정규화 후 정확히 일치하는지 확인
    2. 숫자인 경우 수치적으로 비교 (쉼표 제거 후 float 변환)
    3. 분수/sqrt 등 수학 표현식을 수치적으로 평가하여 비교
       예: Ground Truth "\\frac{51}{8}" → 51/8 = 6.375
           LLM 출력 "6.375" → 6.375  ✅ 일치!
    4. 한쪽이 다른 쪽에 포함되는지 확인 (부분 문자열 매칭)
    """
    pred = normalize_answer(predicted)
    truth = normalize_answer(ground_truth)

    if not pred or not truth:
        return False

    # 1. 정확히 일치
    if pred == truth:
        return True

    # 2. 단순 숫자 비교 (예: "72" vs "72.0", "1,000" vs "1000")
    try:
        pred_num = float(pred.replace(",", "").replace(" ", ""))
        truth_num = float(truth.replace(",", "").replace(" ", ""))
        if abs(pred_num - truth_num) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass

    # 3. 수학 표현식 평가 비교
    #    분수(51/8 = 6.375), sqrt(10 = 3.162...) 등을 수치로 환산하여 비교
    pred_val = eval_math_expr(pred)
    truth_val = eval_math_expr(truth)
    if pred_val is not None and truth_val is not None:
        if abs(pred_val - truth_val) < 1e-4:
            return True

    # 4. 부분 문자열 매칭 (예: "72 clips" 안에 "72"가 포함)
    if truth in pred or pred in truth:
        return True

    return False


# ============================================================
# 로그 생성
# ============================================================

def build_log_entry(item, chat_history, predicted_answer, is_correct):
    """
    1.json 형식에 맞는 로그 항목을 생성합니다.

    에이전트 대화 기록을 history 배열로 변환하고,
    로그 길이를 5~15 범위로 조절합니다.
    """
    # 대화 기록에서 history 배열 구성
    history = []
    for msg in chat_history:
        name = msg.get("name", "Unknown")
        role = msg.get("role", "assistant")
        content = msg.get("content", "")

        # 빈 메시지 건너뛰기
        if not content:
            continue

        # UserProxy의 초기 메시지는 question 필드에 이미 포함되므로 제외
        if name == "UserProxy" and role == "assistant":
            continue

        history.append({
            "content": content,
            "role": role,
            "name": name,
        })

    # 전체 로그 반영 (트리밍 제거)
    # 최대 길이는 max_round 설정으로 제어됩니다.

    # 1.json과 동일한 형식으로 반환
    return {
        "is_correct": is_correct,
        "question": item["question"],
        "task_id": item["task_id"],
        "level": item["level"],
        "ground_truth": item["ground_truth"],
        "predicted_answer": predicted_answer,
        "source": item["source"],
        "history": history,
    }


# ============================================================
# 메인 처리 로직
# ============================================================

def process_single_problem(item, llm_config, max_round):
    """
    단일 문제를 멀티에이전트 파이프라인으로 처리합니다.

    Planner → Drafter → Critic → Editor → FinalAnswer 순서로
    에이전트가 협력하여 문제를 풀고, 결과 로그를 반환합니다.
    """
    # 에이전트 그룹챗 구성
    user_proxy, group_chat, manager = build_group_chat(llm_config, max_round=max_round)

    # 문제를 에이전트에게 전달할 메시지 구성
    task_message = (
        f"Please solve the following problem step by step.\n\n"
        f"Question: {item['question']}\n\n"
        f"Provide the answer as a precise, short value (number, word, or phrase)."
    )

    # 그룹챗 실행 (에이전트 간 대화 시작)
    user_proxy.initiate_chat(
        manager,
        message=task_message,
    )

    # 대화 기록 수집
    chat_history = group_chat.messages

    # 최종 답변 추출
    predicted_answer = extract_final_answer(chat_history)

    # 정답 여부 판정
    is_correct = check_answer(predicted_answer, item["ground_truth"])

    # 1.json 형식 로그 생성
    log_entry = build_log_entry(item, chat_history, predicted_answer, is_correct)

    return log_entry


def main():
    """메인 실행 함수: 데이터 로드 → 파이프라인 실행 → 로그 저장 → 정확도 리포트"""
    args = get_args()

    # ── 실행 설정 출력 ──
    print("=" * 60)
    print("Multi-Agent Interaction Log Generator")
    print("=" * 60)
    print(f"  Provider:      {args.provider}")
    print(f"  Model:         {args.model}")
    print(f"  Dataset:       {args.dataset}")
    print(f"  Max Problems:  {args.max_problems if args.max_problems > 0 else 'ALL'}")
    print(f"  Start Index:   {args.start_index}")
    print(f"  Max Rounds:    {args.max_round}")
    print(f"  Temperature:   {args.temperature}")
    print(f"  Output Dir:    {args.output_dir}")
    print("=" * 60)

    # ── LLM 설정 빌드 ──
    llm_config = build_llm_config(args)

    # ── 데이터셋 로드 ──
    items = load_dataset(args.dataset)
    print(f"\nLoaded {len(items)} problems total.")

    # ── 처리 범위 설정 (start_index ~ start_index + max_problems) ──
    start = args.start_index
    end = start + args.max_problems if args.max_problems > 0 else len(items)
    items = items[start:end]
    print(f"Processing {len(items)} problems (index {start} to {start + len(items) - 1}).\n")

    # ── 출력 디렉터리 생성 (벤치마크별 폴더 구분) ──
    # 예: results/GSM8K/correct/, results/MATH/incorrect/
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 처리할 데이터의 소스별 디렉터리를 미리 생성
    sources_in_data = set(item["source"] for item in items)
    for source in sources_in_data:
        (output_dir / source / "correct").mkdir(parents=True, exist_ok=True)
        (output_dir / source / "incorrect").mkdir(parents=True, exist_ok=True)

    # ── 문제별 처리 루프 ──
    results = []          # 전체 결과 리스트
    correct_count = 0     # 정답 수
    error_count = 0       # 에러 수

    for idx, item in enumerate(items):
        task_id = item["task_id"]
        print(f"\n[{idx + 1}/{len(items)}] Processing task_id={task_id} ({item['source']})")
        print(f"  Question: {item['question'][:80]}...")
        print(f"  Ground Truth: {item['ground_truth']}")

        try:
            # 멀티에이전트 파이프라인 실행
            log_entry = process_single_problem(item, llm_config, args.max_round)
            results.append(log_entry)

            # ── 개별 로그 저장 (벤치마크별 폴더에 정답/오답 분리) ──
            source = log_entry["source"]
            if log_entry["is_correct"]:
                correct_count += 1
                save_dir = output_dir / source / "correct"
                status = "✅ CORRECT"
            else:
                save_dir = output_dir / source / "incorrect"
                status = "❌ INCORRECT"

            # task_id에 슬래시(/) 등이 포함될 수 있으므로 안전한 파일명으로 변환
            # task_id 자체가 .json으로 끝나는 경우 중복 방지
            safe_task_id = str(task_id).replace("/", "_").replace("\\", "_")
            if safe_task_id.endswith(".json"):
                filename = safe_task_id
            else:
                filename = f"{safe_task_id}.json"
                
            filepath = save_dir / filename
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(log_entry, f, ensure_ascii=False, indent=4)

            print(f"  Predicted: {log_entry['predicted_answer']}")
            print(f"  Result: {status}")
            print(f"  Log saved: {filepath}")
            print(f"  History length: {len(log_entry['history'])}")

        except Exception as e:
            # ── 에러 발생 시 에러 로그 저장 ──
            error_count += 1
            print(f"  ⚠️ ERROR: {str(e)}")
            traceback.print_exc()

            source = item["source"]
            error_log = {
                "is_correct": False,
                "question": item["question"],
                "task_id": task_id,
                "level": item["level"],
                "ground_truth": item["ground_truth"],
                "predicted_answer": "",
                "source": source,
                "history": [{"content": f"Error: {str(e)}", "role": "system", "name": "Error"}],
                "error": str(e),
            }
            safe_task_id = str(task_id).replace("/", "_").replace("\\", "_")
            if safe_task_id.endswith(".json"):
                filename = safe_task_id
            else:
                filename = f"{safe_task_id}.json"
                
            filepath = output_dir / source / "incorrect" / filename
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(error_log, f, ensure_ascii=False, indent=4)

        # API 요청 속도 제한 방지를 위한 대기
        time.sleep(1)

    # ============================================================
    # 결과 요약 리포트
    # ============================================================
    total = len(items)
    processed = total - error_count
    accuracy = (correct_count / processed * 100) if processed > 0 else 0

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Total Problems:   {total}")
    print(f"  Processed:        {processed}")
    print(f"  Errors:           {error_count}")
    print(f"  Correct:          {correct_count}")
    print(f"  Incorrect:        {processed - correct_count}")
    print(f"  Accuracy:         {accuracy:.2f}%")
    print("=" * 60)
    print(f"  Correct logs:     {output_dir}/<benchmark>/correct")
    print(f"  Incorrect logs:   {output_dir}/<benchmark>/incorrect")
    print("=" * 60)

    # ── 전체 요약을 JSON으로 저장 ──
    summary = {
        "provider": args.provider,
        "model": args.model,
        "dataset": args.dataset,
        "total_problems": total,
        "processed": processed,
        "errors": error_count,
        "correct": correct_count,
        "incorrect": processed - correct_count,
        "accuracy": round(accuracy, 2),
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
