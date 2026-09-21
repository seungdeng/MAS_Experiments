#!/usr/bin/env python3
"""attribute.py — 최종 오류 귀인 (결정적 오류 단계 + 에이전트/모듈 지목). OpenRouter 기반, 프롬프트는 영어.

결정적 오류의 정의(프롬프트에 명시): "실패로 이어진 가장 이른 오류 (the earliest error that led to the failure)".

방식(method)은 등록표 METHODS 로 관리한다 — 같은 프롬프트 골격·같은 로그 렌더링에 '추가 블록'만 다르다.
  aao  베이스라인: 전체 로그만 (Who&When 의 all-at-once)
  ckl  제안: 전체 로그 + 체크리스트 판정 결과 블록 (ckl_judge.py 가 만든 ckl_*.jsonl 필요: --ckl-file)
       블록 내용은 인자로 바꾼다:
         --ckl-layers L0,L1,L2   넣을 계층 (기본 전부)
         --ckl-view applicable   판단 가능했던 L1/L2 항목만(오류+오류없음). errors = 오류 항목만
         --ckl-indet             판단 불가 항목도 'not assessable' 로 표시 (기본: 뺌)
새 방식 추가: METHODS 에 (needs_ckl, extra_fn) 한 줄을 넣으면 --method 선택지에 자동 반영된다.

사용법 (Windows cmd 는 한 줄로)
  python attribute.py --method aao --model anthropic/claude-sonnet-5 --run --limit 10
  python attribute.py --method ckl --ckl-file ckl_openai_gpt-5-mini.jsonl --model anthropic/claude-sonnet-5 --run
  python attribute.py --method ckl --ckl-file ckl_x.jsonl --ckl-layers L1,L2 --ckl-view errors --run --out attr_ckl_err.jsonl
  python attribute.py --method ckl --ckl-file ckl_x.jsonl --dry-run      # 프롬프트 확인 (API 미사용)
--ckl-file 은 파일명만 줘도 results/ckl/ 에서 찾는다. 출력: results/attribution/attr_<method>_<모델슬러그>.jsonl. 다른 베이스라인도 같은 형식(아래)으로 저장하면 evaluate.py 가 함께 비교한다.
  {"trace_id":..., "benchmark":..., "subset":..., "method":"...", "model":"...", "pred":{"step":int,"agent":str|null,"module":str|null}}
※ 옵션을 바꿔 여러 변형을 돌릴 때는 --out 을 다르게 주고, evaluate.py 에서 method 이름이 겹치면 구분이 안 되므로
  변형마다 --method-label 로 이름을 붙인다 (예: --method-label ckl_errors).
"""
import argparse
import os
import sys

from checklist import EN, ERROR, INDET, L0, NO_ERROR, UPPER
from common import (FatalAPIError, add_llm_args, add_select_args, done_ids, find_result, latest, make_llm, model_prices,
                    read_jsonl, render_log, result_path, run_jobs, safe_name, select_traces)
from common import RESULTS_DIR


# ---------------------------------------------------------------- checklist block
def checklist_block(ckl_row, view='applicable', layers=('L0', 'L1', 'L2'), indet=False):
    """ckl_judge 결과 행 → 프롬프트 블록. 각 줄은 '질문 -> 답' 형태.
    L0: yes(기록됨)/no(없음), null 은 생략.  L1/L2: ERROR at step N / no error found / (indet 옵션) not assessable."""
    out = []
    if 'L0' in layers:
        lines = []
        for iid, r in (ckl_row.get('l0') or {}).items():
            if r.get('v') is None:
                continue
            why = f' Evidence: {r["why"]}' if r.get('why') else ''
            lines.append(f'- {iid}: {EN[iid]} -> {"YES" if r["v"] else "NO"}.{why}')
        if lines:
            out.append('Log properties (what this log records):\n' + '\n'.join(lines))
    lines = []
    for layer in ('L1', 'L2'):
        if layer not in layers:
            continue
        for iid, r in (ckl_row.get(layer.lower()) or {}).items():
            if r['status'] == INDET:
                if indet:
                    need = (r.get('reason') or '').replace('dep:', 'missing ')
                    lines.append(f'- {iid}: {EN[iid]} -> not assessable ({need or "insufficient information"}).')
                continue
            if r['status'] == NO_ERROR and view == 'errors':
                continue
            if r['status'] == ERROR:
                res = 'ERROR' + (f' at step {r["step"]}' if r.get('step') is not None else '')
            else:
                res = 'no error found'
            why = f' Evidence: {r["why"]}' if r.get('why') else ''
            lines.append(f'- {iid}: {EN[iid]} -> {res}.{why}')
    if lines:
        out.append('Error findings:\n' + '\n'.join(lines))
    if not out:
        return ''
    return ('### Checklist findings (from a separate pass over this same log)\n'
            'Use these findings as leads and verify them against the log yourself; they may be wrong.\n\n'
            + '\n\n'.join(out))


# ---------------------------------------------------------------- methods
# name -> (needs_ckl, extra_fn(ckl_row, opts) -> str inserted before the output format)
METHODS = {
    'aao': (False, lambda ckl, o: ''),
    'ckl': (True, lambda ckl, o: checklist_block(ckl, o['view'], o['layers'], o['indet'])),
}


def task_instruction(trace, extra=''):
    bm = trace['benchmark']
    if bm == 'WhoWhen':
        agents = sorted({s['agent'] for s in trace['steps'] if s.get('agent')})
        who = (f'This log is a conversation among multiple agents ({", ".join(agents)}).\n'
               'Identify (1) the step where the decisive error occurred and (2) the agent responsible for it.')
        fmt = '{"step": <int>, "agent": "<one of the agent names above>", "reason": "<=40 words"}'
    elif bm == 'AEB':
        who = ('This log is a single agent with a modular reasoning loop (memory, reflection, plan, action).\n'
               'Identify (1) the step where the decisive error occurred and (2) the module in which the error originated.')
        fmt = '{"step": <int>, "module": "memory|reflection|plan|action|system", "reason": "<=40 words"}'
    else:  # TRAIL: the log is a flattened OpenTelemetry span tree
        who = ('This log is a flattened OpenTelemetry span tree of an agent run (each "[step N]" is one span; ">" marks nesting depth).\n'
               'Identify the step (span) where the decisive error occurred.')
        fmt = '{"step": <int>, "reason": "<=40 words"}'
    return ('[Instruction] Failure attribution\n'
            'The agent run in the log above failed to accomplish its task. '
            'The decisive error is the earliest error that led to the failure.\n'
            + who + '\n'
            '"step" must be the integer N of the "[step N]" marker in the log (0-based).'
            + ('\n\n' + extra if extra else '')
            + f'\n\nOutput format: {fmt}')


DEFAULT_OPTS = {'view': 'applicable', 'layers': ('L0', 'L1', 'L2'), 'indet': False}


def build_instruction(trace, method='aao', ckl_row=None, opts=None):
    needs_ckl, extra_fn = METHODS[method]
    if needs_ckl and ckl_row is None:
        raise ValueError(f'method "{method}" needs a checklist row for {trace["trace_id"]}')
    return task_instruction(trace, extra_fn(ckl_row, {**DEFAULT_OPTS, **(opts or {})}) if needs_ckl else extra_fn(None, {}))


# ---------------------------------------------------------------- run
def attribute_trace(llm, trace, method, ckl_row, opts, label=None):
    res, usage = llm.ask(render_log(trace), build_instruction(trace, method, ckl_row, opts))
    row = {'trace_id': trace['trace_id'], 'benchmark': trace['benchmark'], 'subset': trace['subset'],
           'method': label or method, 'model': llm.model, 'usage': usage}
    if ckl_row is not None:
        row['ckl_model'] = ckl_row.get('model')
        row['ckl_opts'] = {**DEFAULT_OPTS, **(opts or {})}
        row['ckl_opts']['layers'] = list(row['ckl_opts']['layers'])
    if '_error' in res:
        row['errors'] = {'attribute': res['_error']}
        row['pred'] = {'step': None, 'agent': None, 'module': None}
        return row
    try:
        step = int(res.get('step'))
    except (TypeError, ValueError):
        step = None
    mod = res.get('module')
    mod = {'planning': 'plan'}.get(str(mod).lower(), str(mod).lower()) if mod else None
    row['pred'] = {'step': step, 'agent': res.get('agent'), 'module': mod, 'reason': res.get('reason', '')}
    if step is None:
        row['errors'] = {'attribute': 'no integer step in the answer'}
    return row


def estimate(traces, model, method, ckl_by_id, opts):
    pr = model_prices(model)
    if not pr:
        sys.exit(f'No pricing for "{model}". Check with: python common.py --search <text>')
    chars = sum(len(render_log(t)) + len(build_instruction(t, method, ckl_by_id.get(t['trace_id']), opts)) for t in traces)
    tok, n, OUT = chars / 3.5, len(traces), 400
    print(f'{n} traces x 1 call | input ~ {tok / 1e6:.2f}M tokens | assumed output {OUT} tok/call (reasoning NOT included)')
    print(f'{model} {method}: ~ ${tok * pr["inp"] + n * OUT * pr["out"]:.2f}  (no cache; the log prefix is only reused if calls come within minutes)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    add_select_args(ap)
    add_llm_args(ap)
    ap.add_argument('--method', choices=list(METHODS), required=True)
    ap.add_argument('--method-label', help='name recorded in the output / shown by evaluate.py (default: the method)')
    ap.add_argument('--ckl-file', help='ckl_judge.py output (required for methods that need the checklist)')
    ap.add_argument('--ckl-layers', default='L0,L1,L2', help='comma list of layers put in the prompt (default L0,L1,L2)')
    ap.add_argument('--ckl-view', choices=['applicable', 'errors'], default='applicable')
    ap.add_argument('--ckl-indet', action='store_true', help='also list not-assessable (INDETERMINATE) L1/L2 items')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--estimate', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--out')
    a = ap.parse_args()
    layers = tuple(x.strip().upper() for x in a.ckl_layers.split(',') if x.strip())
    bad = [x for x in layers if x not in ('L0', 'L1', 'L2')]
    if bad:
        sys.exit(f'--ckl-layers: unknown layer(s) {bad}')
    opts = {'view': a.ckl_view, 'layers': layers, 'indet': a.ckl_indet}
    traces = select_traces(a)
    ckl_by_id = {}
    if METHODS[a.method][0]:
        if not a.ckl_file:
            sys.exit(f'--method {a.method} needs --ckl-file')
        ckl_path = find_result('ckl', a.ckl_file)
        if not os.path.exists(ckl_path):
            have = sorted(os.listdir(os.path.join(RESULTS_DIR, 'ckl'))) if os.path.isdir(os.path.join(RESULTS_DIR, 'ckl')) else []
            names = ', '.join(have) if have else '(none yet)'
            sys.exit(f'checklist file not found: {a.ckl_file}\n'
                     f'  available in results/ckl/: {names}\n'
                     '  create one first:  python ckl_judge.py --run --model <slug>')
        ckl_by_id = {r['trace_id']: r for r in latest(read_jsonl(ckl_path))}
        missing = [t['trace_id'] for t in traces if t['trace_id'] not in ckl_by_id]
        if missing:
            print(f'[warn] {len(missing)} selected traces have no checklist row and are skipped (e.g. {missing[0]})', file=sys.stderr)
        traces = [t for t in traces if t['trace_id'] in ckl_by_id]
    if not traces:
        sys.exit('no traces selected')
    model = a.model or os.environ.get('DEFAULT_MODEL', '')
    if a.dry_run:
        t = traces[0]
        print(f'trace={t["trace_id"]}\n' + '=' * 70 + '\n' + build_instruction(t, a.method, ckl_by_id.get(t['trace_id']), opts))
    elif a.estimate:
        estimate(traces, model, a.method, ckl_by_id, opts)
    elif a.run:
        try:
            llm = make_llm(a)
            label = a.method_label or a.method
            out = result_path('attribution', a.out or f'attr_{label}_{safe_name(llm.model)}.jsonl')
            skip = done_ids(out)
            todo = [t for t in traces if t['trace_id'] not in skip]
            print(f'{len(traces)} selected, {len(traces) - len(todo)} already done, {len(todo)} to run -> {out}', file=sys.stderr)
            run_jobs(todo, lambda t: attribute_trace(llm, t, a.method, ckl_by_id.get(t['trace_id']), opts, label),
                     out, a.workers)
        except FatalAPIError as e:
            sys.exit(f'STOPPED: {e}\n(finished rows are kept; re-run the same command to resume)')
    else:
        ap.print_help()
