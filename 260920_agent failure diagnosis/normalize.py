#!/usr/bin/env python3
"""normalize.py — 벤치마크 로그 → 공통 스키마 traces.jsonl (1행 = 1 궤적).

스키마
{
  trace_id, benchmark, subset, task_spec, n_steps,
  meta:  {로그에 기록된 실행 메타(모델·한도 등). L0-06 판정 근거로 렌더링에 노출됨},
  gold:  {step(0-based|null), agent|null, module|null, type|null},
  steps: [{i(0-based), agent, content, obs}]
}

260831 판과의 차이 (판정 입력의 정확성 때문에 바꾼 부분)
- 단계 인덱스 i 를 세 벤치마크 모두 0-based 로 통일한다. (구판은 AEB 의 i 가 1-based 라서
  LLM 이 답한 단계와 0-based gold 를 비교하면 1씩 어긋났다.)
- AEB 관찰(obs)을 '다음 user 메시지 앞 400자'(= 프롬프트 서두)가 아니라 실제 환경 관찰로 추출한다.
- TRAIL 스팬은 이름·상태·지속시간·모델/토큰/한도 속성과 입출력을 함께 렌더링한다
  (구판은 logs 본문 400자만 보존 → 시스템 정보(L0-06)가 입력에서 사라졌음).
- 모든 파일 I/O 는 UTF-8 (Windows cp949 기본값에 의존하지 않음).
- E/B 가용성 플래그(avail)는 만들지 않는다. L0 는 LLM 이 판정한다.
"""
import argparse
import ast
import glob
import json
import os
import re
from collections import Counter


# ---------------------------------------------------------------- Who&When
def norm_whowhen(root):
    traces = []
    for subset, sub in (('Algorithm-Generated', 'AG'), ('Hand-Crafted', 'HC')):
        paths = sorted(glob.glob(os.path.join(root, subset, '*.json')),
                       key=lambda x: int(os.path.basename(x)[:-5]))
        for p in paths:
            with open(p, encoding='utf-8') as f:
                d = json.load(f)
            steps = [{'i': i, 'agent': h.get('name') or h.get('role'),
                      'content': h.get('content', '') or '', 'obs': ''}
                     for i, h in enumerate(d['history'])]
            ms = d.get('mistake_step')
            traces.append({
                'trace_id': f'WW-{sub}-{os.path.basename(p)[:-5]}',
                'benchmark': 'WhoWhen', 'subset': sub,
                'task_spec': d.get('question', ''), 'n_steps': len(steps),
                'meta': {},
                'gold': {'agent': d.get('mistake_agent'), 'step': int(ms) if ms is not None else None,
                         'module': None, 'type': d.get('mistake_type')},
                'steps': steps,
            })
    return traces


# ---------------------------------------------------------------- AgentErrorBench
_OBS_ENV = re.compile(r"current observation is:\s*(.*?)\n\s*Now it's your turn", re.S)
_OBS_HIST = re.compile(r"\[Observation (\d+): '(.*?)', Action \1: ", re.S)


def aeb_obs(user_msg):
    """다음 user 메시지(프롬프트 템플릿)에서 직전 행동의 결과 관찰만 추출."""
    m = _OBS_ENV.search(user_msg)                 # ALFWorld / WebShop
    if m:
        return m.group(1).strip()
    hist = _OBS_HIST.findall(user_msg)            # GAIA: 최근 관찰 목록의 마지막이 최신
    return hist[-1][1].strip() if hist else ''


def norm_aeb(root):
    labels = {}
    for f in ('alfworld', 'gaia', 'webshop'):
        with open(os.path.join(root, 'Label', f'{f}_labels.json'), encoding='utf-8') as fh:
            for x in json.load(fh):
                labels[x['trajectory_id']] = x
    traces = []
    for sub in ('ALFWorld', 'WebShop', 'GAIA'):
        for p in sorted(glob.glob(os.path.join(root, 'Original_Failure_Trajectory', sub, '*.json'))):
            tid = os.path.basename(p)[:-5]
            with open(p, encoding='utf-8') as fh:
                d = json.load(fh)
            msgs, md = d['messages'], d.get('metadata', {})
            steps = []
            for i, m in enumerate(msgs):
                if m['role'] != 'assistant':
                    continue
                nxt = msgs[i + 1] if i + 1 < len(msgs) and msgs[i + 1]['role'] == 'user' else None
                steps.append({'i': len(steps), 'agent': md.get('model', 'agent'),
                              'content': m['content'], 'obs': aeb_obs(nxt['content']) if nxt else ''})
            lab = labels.get(tid, {})
            gstep = lab.get('critical_failure_step')            # 원본은 1-based
            if gstep is not None and not (1 <= gstep <= len(steps)):
                gstep = None                                     # 범위 초과 1건(GAIA_003)은 제외
            gmod = lab.get('critical_failure_module')
            gmod = 'plan' if gmod == 'planning' else gmod
            gtype = None
            if lab.get('step_annotations'):
                v = [x for k, x in lab['step_annotations'][0].items() if k != 'step']
                if v and isinstance(v[0], dict):
                    gtype = v[0].get('failure_type')
            meta = {k: v for k, v in md.items() if k not in ('gamefile', 'won')}
            traces.append({
                'trace_id': f'AEB-{sub}-{tid}', 'benchmark': 'AEB', 'subset': sub,
                'task_spec': msgs[0]['content'] if msgs else '', 'n_steps': len(steps),
                'meta': meta,
                'gold': {'agent': None, 'step': (gstep - 1) if gstep is not None else None,
                         'module': gmod, 'type': gtype},
                'steps': steps,
            })
    return traces


# ---------------------------------------------------------------- AgentRx / tau-bench
def norm_agentrx(root):
    """Keep all messages and map explicit source indices to zero-based steps.

    Only runtime messages are exposed. Reward info, reference actions/outputs,
    and failure explanations must never enter task_spec, meta, or steps.
    """
    def read(relative):
        with open(os.path.join(root, relative), encoding='utf-8') as fh:
            return json.load(fh)

    rows = read('tau_retail/tau_dataset_failed.json')
    labels = read('ground_truth/tau_ground_truth.json')
    by_id = {str(row['task_id']): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError('AgentRx: duplicate task_id')
    label_ids = [str(lab['trajectory_id']) for lab in labels]
    if len(set(label_ids)) != len(labels) or set(label_ids) != set(by_id):
        raise ValueError('AgentRx: failed trajectories and labels must match one-to-one')
    traces = []
    for lab in labels:
        tid = str(lab['trajectory_id'])
        row = by_id[tid]
        messages = row['traj']
        id2idx = {}
        steps = []
        for i, msg in enumerate(messages):
            index = msg['index']
            if not isinstance(index, int) or index in id2idx:
                raise ValueError(f'AgentRx {tid}: invalid or duplicate message index')
            id2idx[index] = i
            parts = [msg.get('content') or '']
            for key in ('tool_calls', 'tool_call_id', 'name'):
                if msg.get(key):
                    parts.append(key + ': ' + json.dumps(msg[key], ensure_ascii=False))
            steps.append({'i': i, 'source_index': index, 'agent': msg['role'],
                          'content': '\n'.join(part for part in parts if part), 'obs': ''})
        cause_id = lab['root_cause']['failure_id']
        causes = [failure for failure in lab['failures'] if failure['failure_id'] == cause_id]
        if len(causes) != 1 or causes[0]['step_number'] not in id2idx:
            raise ValueError(f'AgentRx {tid}: root cause cannot be mapped to a message')
        cause = causes[0]
        traces.append({
            'trace_id': f'AgentRx-tau_retail-{tid}', 'benchmark': 'AgentRx', 'subset': 'tau_retail',
            'task_spec': '', 'meta': {}, 'n_steps': len(steps),
            'gold': {'step': id2idx[cause['step_number']], 'agent': None, 'module': None,
                     'type': cause['failure_category'], 'source_step': cause['step_number'],
                     'failed_agent': cause.get('failed_agent')},
            'steps': steps,
        })
    return traces


# ---------------------------------------------------------------- TRAIL
def _pyeval(s):
    """OTel 필드는 파이썬 repr 문자열 — ast.literal_eval, 실패 시 None."""
    if not isinstance(s, str):
        return s
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def _clip(s, n, tail=False):
    s = str(s).replace('\r', '')
    if len(s) <= n:
        return s
    return ('…' + s[-n:]) if tail else (s[:n] + '…')


def render_span(s, depth):
    a = _pyeval(s.get('span_attributes')) or {}
    head = f"{'>' * depth}[{s.get('span_name', '')}] status={s.get('status_code')} dur={s.get('duration')}"
    if s.get('status_message'):
        head += f" status_message={_clip(s['status_message'], 200)}"
    parts = [head]
    sysinfo = [f'{k.split(".")[-1]}={a[k]}' for k in
               ('llm.model_name', 'llm.token_count.prompt', 'llm.token_count.completion', 'smolagents.max_steps')
               if k in a]
    if sysinfo:
        parts.append('  sys: ' + ' '.join(sysinfo))
    if a.get('input.value'):
        parts.append('  in: ' + _clip(a['input.value'], 500, tail=True))   # 입력은 최근 대화가 뒤에 있음
    if a.get('output.value'):
        parts.append('  out: ' + _clip(a['output.value'], 800))
    logs = _pyeval(s.get('logs')) or []
    body = ' | '.join(_clip(l.get('body', ''), 400) for l in logs[:3] if isinstance(l, dict) and l.get('body'))
    if body:
        parts.append('  log: ' + body)
    return '\n'.join(parts)


def norm_trail(parquets):
    import pyarrow.parquet as pq   # TRAIL 을 쓸 때만 필요
    traces = []
    for sub, path in parquets:
        for row in pq.read_table(path).to_pylist():
            tr = json.loads(row['trace'].replace('NaN', 'null')) if not _is_json(row['trace']) else json.loads(row['trace'])
            try:
                lb = json.loads(row['labels'])
            except Exception:
                try:
                    lb = json.loads(row['labels'].replace('NaN', 'null'))
                except Exception:
                    lb = {'errors': []}
            flat = []

            def rec(s, depth):
                flat.append((s, depth))
                for c in (_pyeval(s.get('child_spans')) or []):
                    if isinstance(c, dict):
                        rec(c, depth + 1)
            for s in tr.get('spans', []):
                rec(s, 0)
            steps, id2idx = [], {}
            for i, (s, depth) in enumerate(flat):
                id2idx[s.get('span_id')] = i
                steps.append({'i': i, 'agent': s.get('service_name', ''), 'content': render_span(s, depth), 'obs': ''})
            errors = lb.get('errors') or []
            locs = sorted(id2idx[e['location']] for e in errors if e.get('location') in id2idx)
            traces.append({
                'trace_id': f"TRAIL-{sub}-{tr.get('trace_id', '')[:12]}", 'benchmark': 'TRAIL', 'subset': sub,
                'task_spec': '', 'n_steps': len(steps), 'meta': {},
                'gold': {'agent': None, 'step': locs[0] if locs else None, 'module': None,
                         'type': errors[0].get('category') if errors else None},
                'steps': steps,
            })
    return traces


def _is_json(s):
    try:
        json.loads(s)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- main
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--whowhen', help='Who&When 루트 (Algorithm-Generated/, Hand-Crafted/ 를 포함)')
    ap.add_argument('--agentrx', help='AgentRx root (tau_retail/, ground_truth/)')
    ap.add_argument('--aeb', help='AgentErrorBench 루트 (Label/, Original_Failure_Trajectory/ 를 포함)')
    ap.add_argument('--trail-gaia')
    ap.add_argument('--trail-swe')
    ap.add_argument('-o', '--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'traces.jsonl'))
    a = ap.parse_args()
    # No source flags: use the self-contained current experiment corpus.
    if not any((a.whowhen, a.aeb, a.agentrx, a.trail_gaia, a.trail_swe)):
        raw = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw')
        a.whowhen = os.path.join(raw, 'WhoWhen')
        a.aeb = os.path.join(raw, 'AgentErrorBench')
        a.agentrx = os.path.join(raw, 'AgentRx')
    for source in (a.whowhen, a.aeb, a.agentrx):
        if source and not os.path.isdir(source):
            ap.error(f'Data directory does not exist: {source}')
    traces = []
    if a.whowhen:
        traces += norm_whowhen(a.whowhen)
    if a.aeb:
        traces += norm_aeb(a.aeb)
    if a.agentrx:
        traces += norm_agentrx(a.agentrx)
    pq_list = [(s, p) for s, p in (('GAIA', a.trail_gaia), ('SWE', a.trail_swe)) if p]
    if pq_list:
        traces += norm_trail(pq_list)
    if not traces:
        ap.error('No traces found; check source paths')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')
    print(f'{len(traces)} traces -> {a.out}')
    groups = Counter((t['benchmark'], t['subset']) for t in traces)
    for (bm, sub), n in sorted(groups.items()):
        ts = [t for t in traces if t['benchmark'] == bm and t['subset'] == sub]
        with_gold = sum(1 for t in ts if t['gold']['step'] is not None)
        steps = [s for t in ts for s in t['steps']]
        obs_cov = sum(1 for s in steps if s['obs']) / len(steps) if steps else 0
        extra = f' | obs 채움 {obs_cov:.0%}' if bm == 'AEB' else ''
        print(f'[{bm}/{sub}] n={n} gold_step 보유 {with_gold}/{n} | 평균 단계 {len(steps) / n:.1f}{extra}')
