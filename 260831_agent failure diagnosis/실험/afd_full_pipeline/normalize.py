#!/usr/bin/env python3
"""normalize.py — 벤치마크 로그 → 공통 스키마 traces.jsonl 변환기.

공통 스키마 (1행 = 1 trace):
{
  trace_id, benchmark, subset, task_spec, n_steps,
  outcome: {label, rationale},
  gold: {agent, step(0-based int|null), reason, type},
  steps: [{i, agent, content}],
  avail: {필드 가용성 플래그: "E"(explicit 구조 필드) | "B"(embedded 자유텍스트 근거) | null(부재)}
}

E1 커버리지 판정 규칙(판정 방식별 요구 수준)은 e1_coverage.py 참조.
Embedded(B) 판정은 문서화된 패턴 근거로 trace별 자동 산출(감사 가능).
"""
import json, re, glob, os, sys, argparse

TOOL_ROLES = {'WebSurfer', 'FileSurfer', 'ComputerTerminal', 'Assistant'}
PLAN_PAT = re.compile(r'\b(initial plan|plan:|we will|steps?:|next step|계획)\b', re.I)
ACTION_PAT = re.compile(r'(```|`\w+\(|->\s*\w+|perform(ed)? the (following )?(search|action)|click|type[d]? |navigate|execute)', re.I)
OBS_PAT = re.compile(r'(output:|result:|returned|viewport|screenshot|http[s]?://|I found|according to the page)', re.I)
REFL_PAT = re.compile(r'\b(verify|check(ed|ing)?|however|it seems|not (yet )?(complete|correct)|termination condition|재검|확인)\b', re.I)
ERR_PAT = re.compile(r'\b(error|exception|traceback|failed to|invalid)\b', re.I)


def norm_whowhen(root):
    traces = []
    for subset, sub in (('Algorithm-Generated', 'AG'), ('Hand-Crafted', 'HC')):
        for p in sorted(glob.glob(os.path.join(root, subset, '*.json')),
                        key=lambda x: int(os.path.basename(x)[:-5])):
            d = json.load(open(p, encoding='utf-8'))
            hist = d['history']
            steps = []
            for i, h in enumerate(hist):
                agent = h.get('name') or h.get('role')
                steps.append({'i': i, 'agent': agent, 'content': h.get('content', '')})
            blob = '\n'.join(s['content'] for s in steps)
            has_name = any('name' in h for h in hist)
            avail = {
                # L0 explicit 구조 필드 (스키마 검사)
                'task_spec':        'E' if d.get('question') else None,
                'step_index':       'E',                       # 순서 보존 리스트
                'module_tags':      None,                      # 모듈 구분 기록 없음
                'observation':      'B' if OBS_PAT.search(blob) else None,   # 별도 필드 없음, 발화 내 인용
                'outcome_label':    'E' if ('is_correct' in d or 'is_corrected' in d) else None,
                'outcome_rationale':'E' if d.get('ground_truth') else None,
                'raw_tool_response':None,
                'sys_metadata':     None,                      # 모델·토큰·지연 없음
                'step_limit_cfg':   None,
                'error_msg':        'B' if ERR_PAT.search(blob) else None,
                # 모듈 필드 (explicit 없음 → embedded 패턴 근거)
                'action_field':     'B' if ACTION_PAT.search(blob) else None,
                'plan_field':       'B' if PLAN_PAT.search(blob) else None,
                'memory_field':     'B' if len(steps) > 1 else None,  # 전 단계 이력=메시지 스트림
                'reflection_field': 'B' if REFL_PAT.search(blob) else None,
                'agent_id':         'E' if (has_name or sub == 'HC') else None,
                # 골드 주석
                'gold_step':        'E' if d.get('mistake_step') is not None else None,
                'gold_agent':       'E' if d.get('mistake_agent') else None,
                'gold_type':        'E' if d.get('mistake_type') else None,
            }
            traces.append({
                'trace_id': f'WW-{sub}-{os.path.basename(p)[:-5]}',
                'benchmark': 'WhoWhen', 'subset': sub,
                'task_spec': d.get('question', ''), 'n_steps': len(steps),
                'outcome': {'label': d.get('is_correct', d.get('is_corrected')),
                            'rationale': d.get('ground_truth')},
                'gold': {'agent': d.get('mistake_agent'),
                         'step': int(d['mistake_step']) if d.get('mistake_step') is not None else None,
                         'reason': d.get('mistake_reason'), 'type': d.get('mistake_type')},
                'steps': steps, 'avail': avail,
            })
    return traces


MODULE_TAG = re.compile(r'<(plan|memory|reflection|action|think)>')
AEB_ERR = {'ALFWorld': 'Nothing happens', 'WebShop': 'Invalid action', 'GAIA': None}
AEB_LIMIT = re.compile(r'(step limit|maximum (number of )?steps|steps? remaining|한도)', re.I)


def norm_aeb(root):
    """AgentErrorBench: Modular-ReAct 태그 로그 + 별도 골드 라벨(1-based).
    step = assistant 턴(모듈 태그 블록) + 직후 user 관찰. 라벨 'planning'→'plan' 정규화.
    골드 step이 궤적 범위(1..n)를 벗어나면 gold.step=None + 사유 기록 (1건 존재: GAIA_003)."""
    labels = {}
    for f in ('alfworld', 'gaia', 'webshop'):
        for x in json.load(open(os.path.join(root, 'Label', f'{f}_labels.json'), encoding='utf-8')):
            labels[x['trajectory_id']] = x
    traces = []
    for sub in ('ALFWorld', 'WebShop', 'GAIA'):
        for p in sorted(glob.glob(os.path.join(root, 'Original_Failure_Trajectory', sub, '*.json'))):
            tid = os.path.basename(p)[:-5]
            d = json.load(open(p, encoding='utf-8'))
            msgs, md = d['messages'], d.get('metadata', {})
            steps = []
            for i, m in enumerate(msgs):
                if m['role'] != 'assistant':
                    continue
                obs = msgs[i + 1]['content'] if i + 1 < len(msgs) and msgs[i + 1]['role'] == 'user' else ''
                steps.append({'i': len(steps) + 1, 'agent': md.get('model', 'agent'),
                              'content': m['content'], 'obs': obs})
            blob = '\n'.join(s['content'] + s['obs'] for s in steps)
            lab = labels.get(tid, {})
            gstep = lab.get('critical_failure_step')
            gnote = None
            if gstep is not None and not (1 <= gstep <= len(steps)):
                gnote = f'label step {gstep} > n_steps {len(steps)} — 범위 초과로 제외'
                gstep = None
            gmod = lab.get('critical_failure_module')
            gmod = 'plan' if gmod == 'planning' else gmod
            gtype = None
            if lab.get('step_annotations'):
                v = [x for k, x in lab['step_annotations'][0].items() if k != 'step']
                if v and isinstance(v[0], dict):
                    gtype = v[0].get('failure_type')
            err_marker = AEB_ERR[sub]
            avail = {
                'task_spec':        'E' if 'task' in (msgs[0]['content'] if msgs else '').lower() else None,
                'step_index':       'E',
                'module_tags':      'E' if MODULE_TAG.search(blob) else None,
                'observation':      'E' if all(s['obs'] for s in steps[:-1]) else ('B' if any(s['obs'] for s in steps) else None),
                'outcome_label':    'E' if 'won' in md else None,
                'outcome_rationale': None,                       # won 불리언만, 근거 없음
                'raw_tool_response': None,                       # 관찰이 프롬프트 템플릿에 포섭됨
                'sys_metadata':     'B' if md.get('model') else None,  # 모델·스텝수만(토큰·지연 없음)
                'step_limit_cfg':   'B' if AEB_LIMIT.search(blob) else None,
                'error_msg':        'E' if (err_marker and err_marker in blob) else ('B' if ERR_PAT.search(blob) else None),
                'action_field':     'E' if '<action>' in blob else None,
                'plan_field':       'E' if '<plan>' in blob else None,
                'memory_field':     'E' if '<memory>' in blob else None,
                'reflection_field': 'E' if '<reflection>' in blob else None,
                'agent_id':         'E' if md.get('model') else None,
                'gold_step':        'E' if gstep is not None else None,
                'gold_agent':       None,                        # 단일 에이전트 — 개념상 N/A성
                'gold_type':        'E' if gtype else None,
            }
            traces.append({
                'trace_id': f'AEB-{sub}-{tid}',
                'benchmark': 'AEB', 'subset': sub,
                'task_spec': msgs[0]['content'][:2000] if msgs else '', 'n_steps': len(steps),
                'outcome': {'label': md.get('won'), 'rationale': None},
                'gold': {'agent': None, 'step': (gstep - 1) if gstep is not None else None,  # 1-based→0-based 'reason': gnote or (lab.get('step_annotations', [{}])[0] and None),
                         'module': gmod, 'type': gtype},
                'steps': steps, 'avail': avail,
            })
    return traces



def _pyeval(s):
    """OTel 필드는 파이썬 repr 문자열 — ast.literal_eval, 실패 시 빈값."""
    import ast
    if not isinstance(s, str): return s
    try: return ast.literal_eval(s)
    except Exception: return None


def norm_trail(parquets):
    """TRAIL (OTel 스팬 트리, parquet). step = DFS 평탄화 스팬(0-based).
    골드: errors[].location(span_id) → 평탄화 인덱스 사상, gold.step = 최초 오류 인덱스.
    스팬 content는 800자 절단 저장(E1/패턴 검출용; E3는 원본 재독)."""
    import pyarrow.parquet as pq
    TRAIL_LIMIT = re.compile(r"max_steps", re.I)
    OUTCOME_PAT = re.compile(r"true_answer|is_correct|final_answer", re.I)
    traces = []
    for sub, path in parquets:
        for row in pq.read_table(path).to_pylist():
            try:
                tr = json.loads(row['trace'])
            except Exception:
                tr = json.loads(row['trace'].replace('NaN', 'null'))
            try:
                lb = json.loads(row['labels'])
            except Exception:
                try: lb = json.loads(row['labels'].replace('NaN', 'null'))
                except Exception: lb = {'errors': [], 'scores': None, '_parse_fail': True}
            # DFS 평탄화
            flat = []
            def rec(s):
                flat.append(s)
                for c in (_pyeval(s.get('child_spans')) or []):
                    if isinstance(c, dict): rec(c)
            for s in tr.get('spans', []): rec(s)
            steps, blob_parts, has_status, has_dur, has_raw = [], [], False, False, False
            id2idx = {}
            for i, s in enumerate(flat):
                id2idx[s.get('span_id')] = i
                logs = _pyeval(s.get('logs')) or []
                body = ' '.join(str(l.get('body', ''))[:400] for l in logs[:3] if isinstance(l, dict))
                content = f"[{s.get('span_name','')}] {body}"[:800]
                steps.append({'i': i, 'agent': s.get('service_name', ''), 'span': s.get('span_name', ''),
                              'content': content})
                blob_parts.append(content)
                if s.get('status_code') is not None: has_status = True
                if s.get('duration'): has_dur = True
                if body.strip(): has_raw = True
            blob = '\n'.join(blob_parts)
            errors = lb.get('errors') or []
            locs = sorted(id2idx[e['location']] for e in errors if e.get('location') in id2idx)
            unmapped = sum(1 for e in errors if e.get('location') not in id2idx)
            avail = {
                'task_spec':        'E' if any('function.arguments' in p for p in blob_parts[:5]) else ('B' if blob_parts else None),
                'step_index':       'E',                       # 타임스탬프 + 트리 순서
                'module_tags':      None,                      # 컴포넌트 스팬은 있으나 인지 모듈 태그 아님
                'observation':      'E' if has_raw else None,  # 도구 스팬 로그 = 구조화 관찰
                'outcome_label':    'B' if OUTCOME_PAT.search(blob) else None,
                'outcome_rationale': None,
                'raw_tool_response':'E' if has_raw else None,  # logs.body 원시 보존
                'sys_metadata':     'E' if has_dur else None,  # duration·timestamp·service
                'step_limit_cfg':   'E' if TRAIL_LIMIT.search(blob) else None,
                'error_msg':        'E' if has_status else None,  # status_code/message 구조 필드
                'action_field':     'E' if any(s['span'] not in ('main','') for s in steps) else None,  # 도구/스텝 스팬=행동
                'plan_field':       'B' if PLAN_PAT.search(blob) else None,
                'memory_field':     'B' if len(steps) > 1 else None,
                'reflection_field': 'B' if REFL_PAT.search(blob) else None,
                'agent_id':         'E',                       # service_name/스팬 계층
                'gold_step':        'E' if locs else None,
                'gold_agent':       None,
                'gold_type':        'E' if any(e.get('category') for e in errors) else None,
            }
            traces.append({
                'trace_id': f"TRAIL-{sub}-{tr.get('trace_id','')[:12]}",
                'benchmark': 'TRAIL', 'subset': sub,
                'task_spec': '', 'n_steps': len(steps),
                'outcome': {'label': None, 'rationale': None},
                'gold': {'agent': None, 'step': locs[0] if locs else None,
                         'errors': [{'idx': id2idx.get(e.get('location')), 'category': e.get('category')} for e in errors],
                         'n_errors': len(errors), 'unmapped': unmapped,
                         'type': errors[0].get('category') if errors else None},
                'steps': steps, 'avail': avail,
            })
    return traces


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--whowhen', help='Who&When 루트 디렉토리')
    ap.add_argument('--aeb', help='AgentErrorBench 루트 디렉토리')
    ap.add_argument('--trail-gaia'); ap.add_argument('--trail-swe')
    ap.add_argument('-o', '--out', default='traces.jsonl')
    a = ap.parse_args()
    traces = []
    if a.whowhen:
        traces += norm_whowhen(a.whowhen)
    if a.aeb:
        traces += norm_aeb(a.aeb)
    pq_list = [(s, p) for s, p in (('GAIA', a.trail_gaia), ('SWE', a.trail_swe)) if p]
    if pq_list:
        traces += norm_trail(pq_list)
    with open(a.out, 'w', encoding='utf-8') as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')
    print(f'{len(traces)} traces -> {a.out}')
    # embedded 근거 감사 출력
    from collections import Counter
    groups = sorted({(t['benchmark'], t['subset']) for t in traces})
    for bm, sub in groups:
        ts = [t for t in traces if t['benchmark'] == bm and t['subset'] == sub]
        if not ts: continue
        c = Counter()
        for t in ts:
            for k, v in t['avail'].items():
                if v: c[f'{k}:{v}'] += 1
        n = len(ts)
        print(f'\n[{bm}/{sub}] n={n} 필드 가용률(감사 근거):')
        for k in sorted(c): print(f'  {k:26s} {c[k]:4d} ({c[k]/n:5.1%})')
