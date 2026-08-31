#!/usr/bin/env python3
"""e1_coverage.py — E1 커버리지(진단가능성) 계산기 → Table 7.

규칙 (paper_final IV장, 2026-08-18 개정):
- 항목별 요구 필드 전부 가용 → 해당 trace에서 '판정가능'.
- 요구 수준: 판정 방식 R(규칙) → explicit(E) 필드 필요.
             L→H(LLM 판정) → embedded(B)도 허용 (LLM은 자유텍스트 판독 가능).
             R+L → R 성분 요구 필드는 E, L 성분은 B 허용.
             R(파생) → 입력 항목이 판정가능하고 step_index가 E이면 판정가능.
- 벤치마크 수준 판정가능: trace의 p≥0.9 에서 판정가능. N/A는 분모 제외.
- L3-06 삭제됨(2026-08-18) → 26항목.
"""
import json, argparse, sys
from collections import defaultdict

# (항목ID, 판정방식, [(필드, 요구수준 'E'|'EB')], 파생입력항목 or None)
ITEMS = [
    ('L0-01', 'R',   [('task_spec', 'E')], None),
    ('L0-02', 'R',   [('step_index', 'E')], None),
    ('L0-03', 'R',   [('module_tags', 'E')], None),
    ('L0-04', 'R',   [('observation', 'E')], None),
    ('L0-05', 'R',   [('outcome_label', 'E'), ('outcome_rationale', 'E')], None),
    ('L0-06', 'R',   [('raw_tool_response', 'E')], None),
    ('L0-07', 'R',   [('sys_metadata', 'E')], None),
    ('L1-01', 'R',   [('step_index', 'E'), ('step_limit_cfg', 'E')], None),
    ('L1-02', 'R',   [('action_field', 'E'), ('error_msg', 'E')], None),
    ('L1-03', 'R+L', [('observation', 'E'), ('error_msg', 'EB')], None),
    ('L1-04', 'R',   [('action_field', 'E')], None),
    ('L2-M1', 'L',   [('memory_field', 'EB'), ('step_index', 'EB')], None),
    ('L2-M2', 'L',   [('memory_field', 'EB'), ('step_index', 'EB')], None),
    ('L2-M3', 'L',   [('memory_field', 'EB'), ('observation', 'EB')], None),
    ('L2-R1', 'L',   [('reflection_field', 'EB'), ('outcome_label', 'EB')], None),
    ('L2-R2', 'L',   [('reflection_field', 'EB'), ('observation', 'EB')], None),
    ('L2-P1', 'L',   [('plan_field', 'EB'), ('task_spec', 'EB')], None),
    ('L2-P2', 'L',   [('plan_field', 'EB'), ('observation', 'EB')], None),
    ('L2-P3', 'L',   [('plan_field', 'EB')], None),
    ('L2-A1', 'L',   [('plan_field', 'EB'), ('action_field', 'EB')], None),
    ('L2-A2', 'R+L', [('action_field', 'E'), ('observation', 'EB')], None),
    ('L3-01', 'L',   [], 'ANY_L2'),
    ('L3-02', 'Rd',  [('step_index', 'E')], 'L3-01'),
    ('L3-03', 'Rd',  [('step_index', 'E')], 'L3-01'),
    ('L3-04', 'L',   [('reflection_field', 'EB'), ('plan_field', 'EB')], None),
    ('L3-05', 'Rd',  [], 'ANY_L2'),
]
L2_IDS = [i for i, *_ in ITEMS if i.startswith('L2')]


def ok(avail, field, need):
    v = avail.get(field)
    return v == 'E' if need == 'E' else v in ('E', 'B')


def judge_trace(avail, na=frozenset()):
    """trace 1건 → {항목: True(판정가능)|False(판정불가)|None(N/A)}"""
    res = {}
    for iid, _, reqs, dep in ITEMS:
        if iid in na:
            res[iid] = None; continue
        d = all(ok(avail, f, n) for f, n in reqs)
        if dep == 'ANY_L2':
            d = d and any(res.get(l2) for l2 in L2_IDS)
        elif dep:
            d = d and bool(res.get(dep))
        res[iid] = d
    return res


def table7(traces, p=0.9, na=frozenset()):
    groups = defaultdict(list)
    for t in traces:
        groups[(t['benchmark'], t['subset'])].append(t)
        groups[(t['benchmark'], 'ALL')].append(t)
    out = {}
    for key, ts in sorted(groups.items()):
        per = defaultdict(int)
        for t in ts:
            for iid, v in judge_trace(t['avail'], na).items():
                if v: per[iid] += 1
        n = len(ts)
        det = {iid: per[iid] / n >= p for iid, *_ in ITEMS if iid not in na}
        denom = len(det)
        cov = sum(det.values()) / denom
        out['/'.join(key)] = {'n': n, 'denom': denom,
                              'determinable': sorted(k for k, v in det.items() if v),
                              'indeterminable': sorted(k for k, v in det.items() if not v),
                              'coverage': round(cov, 4)}
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('traces')
    ap.add_argument('--p', type=float, default=0.9)
    ap.add_argument('-o', '--out', default='table7.json')
    a = ap.parse_args()
    traces = [json.loads(l) for l in open(a.traces)]
    t7 = table7(traces, a.p)
    json.dump(t7, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print(f'26항목 기준 (L3-06 삭제 반영) | p>={a.p} | N/A=0')
    for k, v in t7.items():
        print(f"\n[{k}] n={v['n']}  커버리지 = {len(v['determinable'])}/{v['denom']} = {v['coverage']:.1%}")
        print('  판정가능:', ' '.join(v['determinable']))
        print('  판정불가:', ' '.join(v['indeterminable']))
