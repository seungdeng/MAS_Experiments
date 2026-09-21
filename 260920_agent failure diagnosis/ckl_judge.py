#!/usr/bin/env python3
"""ckl_judge.py — 체크리스트 3단계 LLM 판정 (L0 → L1 → L2, 계층당 1회 호출). OpenRouter 기반, 프롬프트는 영어.

흐름 (궤적 1건)
  1) L0 호출: 로그에 6개 정보가 '기록돼 있는가' 판정 → 기록이 확인된(v=true) L0 집합
  2) L1 호출: 요구 L0 가 전부 충족된 L1 항목만 질문 (없으면 호출 생략)
  3) L2 호출: 요구 L0 가 전부 충족된 L2 항목만 질문 (없으면 호출 생략)
  4) 요구 L0 미충족 항목은 LLM 답과 무관하게 '판단 불가(INDETERMINATE, reason=dep:...)'.
     '오류 없음'이 되지 않는다. (checklist.resolve 참조)

사용법 (Windows cmd 는 한 줄로)
  python ckl_judge.py --dry-run                                   # 프롬프트 확인 (API 미사용)
  python ckl_judge.py --estimate --model anthropic/claude-sonnet-5   # 비용 견적 (API 미사용)
  python ckl_judge.py --run --model anthropic/claude-sonnet-5 --limit 10 --out ckl_test.jsonl
  python ckl_judge.py --run --model openai/gpt-5.6-luna --workers 4
입력 기본값 data/traces.jsonl. 출력: results/ckl/ckl_<모델슬러그>.jsonl (궤적당 1행). 오류가 남은 행은 재실행하면 그 궤적만 다시 호출된다.
"""
import argparse
import os
import sys

from checklist import EN, L0, UPPER, askable, l0_present, resolve
from common import (FatalAPIError, add_llm_args, add_select_args, add_usage, done_ids, make_llm, model_prices,
                    render_log, result_path, run_jobs, safe_name, select_traces)


# ---------------------------------------------------------------- prompts (English)
def instr_l0():
    qs = '\n'.join(f'- {i}: {EN[i]}' for i in L0)
    return ('[Instruction] L0 - Is each piece of information recorded in the log?\n'
            'For each item below, decide whether the execution log above RECORDS that information. '
            'This is not about whether the agent made an error; it is about whether the information exists in the log.\n'
            '- v=true: the log clearly contains it.\n'
            '- v=false: it is absent or insufficient.\n'
            'Base your verdict on fields and text that actually appear in the log, and state the evidence in "why" '
            'in one short sentence.\n\n'
            f'### Items\n{qs}\n\n'
            'Output format (include every item): '
            '{"items": {"L0-01": {"v": true|false, "why": "<=15 words"}, "L0-02": {...}, ...}}')


def instr_upper(layer, ids):
    qs = '\n'.join(f'- {i}: {EN[i]}' for i in ids)
    return (f'[Instruction] {layer} - Does each error occur?\n'
            'For each item below, decide whether the error occurs in the execution log above.\n'
            '- v=true: the error occurs. Set "step" to the FIRST step where it occurs, as the integer N of "[step N]" '
            'in the log (0-based).\n'
            '- v=false: the error does not occur.\n'
            '- v=null: the log does not contain enough information to decide.\n'
            'Answer true only when the evidence is clear, and state the evidence in "why" in one short sentence.\n\n'
            f'### Items\n{qs}\n\n'
            'Output format (include every item): '
            '{"items": {"' + ids[0] + '": {"v": true|false|null, "step": <int or null>, "why": "<=15 words"}, ...}}')


# ---------------------------------------------------------------- judging
def judge_trace(llm, trace):
    log = render_log(trace)
    usage, errors = {}, {}

    res, u = llm.ask(log, instr_l0())
    usage['L0'] = u
    if '_error' in res:
        errors['L0'] = res['_error']
    items0 = res.get('items') if isinstance(res.get('items'), dict) else {}
    l0 = {}
    for i in L0:
        r = items0.get(i)
        l0[i] = {'v': r.get('v') if isinstance(r, dict) else None, 'why': r.get('why', '') if isinstance(r, dict) else ''}
    present = l0_present(l0)

    row = {'trace_id': trace['trace_id'], 'benchmark': trace['benchmark'], 'subset': trace['subset'],
           'model': llm.model, 'l0': l0, 'asked': {}}
    for layer in ('L1', 'L2'):
        ids = askable(layer, present)
        row['asked'][layer] = ids
        llm_items = {}
        if ids:                                    # 요구 L0 를 충족한 항목이 있을 때만 호출
            res, u = llm.ask(log, instr_upper(layer, ids))
            usage[layer] = u
            if '_error' in res:
                errors[layer] = res['_error']
            elif isinstance(res.get('items'), dict):
                llm_items = res['items']
        row[layer.lower()] = resolve(layer, present, llm_items)
    total = {}
    for u in usage.values():
        add_usage(total, u)
    row['usage'] = total
    if errors:
        row['errors'] = errors
    return row


# ---------------------------------------------------------------- estimate / dry run
def estimate(traces, model):
    pr = model_prices(model)
    if not pr:
        sys.exit(f'No pricing for "{model}" (offline, or wrong slug). Check with: python common.py --model {model}')
    tok = sum(len(render_log(t)) for t in traces) / 3.5      # chars -> tokens (rough)
    n, OUT = len(traces), 600                                # assumed output tokens per call (excluding reasoning)
    print(f'{n} traces x up to 3 calls = up to {3 * n} calls (layers whose prerequisites fail are skipped -> upper bound)')
    print(f'log ~ {tok / 1e6:.2f}M tokens (chars/3.5) | assumed output {OUT} tok/call (reasoning tokens NOT included)')
    out = 3 * n * OUT * pr['out']
    nocache = tok * 3 * pr['inp'] + out
    cached = tok * pr['cache_write'] + tok * 2 * pr['cache_read'] + out
    print(f'{model}: no cache hit  ~ ${nocache:.2f}')
    print(f'{model}: cache hits    ~ ${cached:.2f}   (L0 writes the prefix, L1/L2 read it)')


def dry_run(traces):
    t = traces[0]
    log = render_log(t)
    print(f'trace={t["trace_id"]} steps={t["n_steps"]} log={len(log)} chars\n{"=" * 70}')
    print('--- first 800 chars of the log ---\n' + log[:800] + '\n...\n--- last 500 chars ---\n' + log[-500:])
    print('=' * 70 + '\n--- L0 instruction ---\n' + instr_l0())
    print('=' * 70 + '\n--- L1 instruction (assuming all prerequisites met) ---\n' + instr_upper('L1', list(UPPER['L1'])))
    print('=' * 70 + '\n--- L2 instruction (assuming all prerequisites met) ---\n' + instr_upper('L2', list(UPPER['L2'])))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    add_select_args(ap)
    add_llm_args(ap)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--estimate', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--out')
    a = ap.parse_args()
    traces = select_traces(a)
    if a.dry_run:
        dry_run(traces)
    elif a.estimate:
        estimate(traces, a.model or os.environ.get('DEFAULT_MODEL', ''))
    elif a.run:
        try:
            llm = make_llm(a)
            out = result_path('ckl', a.out or f'ckl_{safe_name(llm.model)}.jsonl')
            skip = done_ids(out)
            todo = [t for t in traces if t['trace_id'] not in skip]
            print(f'{len(traces)} selected, {len(traces) - len(todo)} already done, {len(todo)} to run -> {out}', file=sys.stderr)
            run_jobs(todo, lambda t: judge_trace(llm, t), out, a.workers)
        except FatalAPIError as e:
            sys.exit(f'STOPPED: {e}\n(finished rows are kept; re-run the same command to resume)')
    else:
        ap.print_help()
