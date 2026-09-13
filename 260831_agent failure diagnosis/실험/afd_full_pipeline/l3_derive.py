#!/usr/bin/env python3
"""l3_derive.py — 골드 주석 기반 L3 파생 지표 → Table 8 절반 (골드 앵커 통계).

정의 (paper_final IV장):
- t0 = 최초(결정적) 오류 단계 (0-based)
- 위치 = t0/(T-1) 3분위: Left [0,1/3) / Mid [1/3,2/3) / Right [2/3,1]  (T=1이면 Left)
- 전파 길이 = (T-1) - t0  (오류 단계부터 종결까지 남은 단계 수)
"""
import json, argparse, statistics as st
from collections import defaultdict, Counter


def derive(traces):
    groups = defaultdict(list)
    for t in traces:
        if t['gold']['step'] is None:
            continue
        T, t0 = t['n_steps'], t['gold']['step']
        frac = t0 / (T - 1) if T > 1 else 0.0
        pos = 'Left' if frac < 1/3 else ('Mid' if frac < 2/3 else 'Right')
        rec = {'t0': t0, 'T': T, 'frac': frac, 'pos': pos, 'prop': (T - 1) - t0}
        groups[(t['benchmark'], t['subset'])].append(rec)
        groups[(t['benchmark'], 'ALL')].append(rec)
    out = {}
    for key, rs in sorted(groups.items()):
        pc = Counter(r['pos'] for r in rs)
        n = len(rs)
        out['/'.join(key)] = {
            'n': n,
            't0':   {'mean': round(st.mean(r['t0'] for r in rs), 2), 'median': st.median(r['t0'] for r in rs)},
            'T':    {'mean': round(st.mean(r['T'] for r in rs), 2), 'median': st.median(r['T'] for r in rs)},
            'pos':  {k: f'{pc[k]} ({pc[k]/n:.1%})' for k in ('Left', 'Mid', 'Right')},
            'prop': {'mean': round(st.mean(r['prop'] for r in rs), 2), 'median': st.median(r['prop'] for r in rs),
                     'max': max(r['prop'] for r in rs), 'zero': sum(1 for r in rs if r['prop'] == 0)},
        }
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('traces'); ap.add_argument('-o', '--out', default='table8_gold.json')
    a = ap.parse_args()
    traces = [json.loads(l) for l in open(a.traces, encoding='utf-8')]
    r = derive(traces)
    json.dump(r, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    for k, v in r.items():
        print(f"\n[{k}] n={v['n']}")
        print(f"  t0 평균 {v['t0']['mean']} / 중앙값 {v['t0']['median']}   |  T 평균 {v['T']['mean']} / 중앙값 {v['T']['median']}")
        print(f"  위치 3분위: Left {v['pos']['Left']}  Mid {v['pos']['Mid']}  Right {v['pos']['Right']}")
        print(f"  전파 길이: 평균 {v['prop']['mean']} / 중앙값 {v['prop']['median']} / 최대 {v['prop']['max']} / 0(종결단계 오류) {v['prop']['zero']}건")
