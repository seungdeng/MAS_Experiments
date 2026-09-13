#!/usr/bin/env python3
"""table11_breakdown.py — LLM 판정 발생률을 벤치마크×항목으로 분해 (Table 11).

aggregate()는 4모델 전수 합산만 내므로, trace_id -> benchmark 매핑을 traces.jsonl에서
가져와 judg_*.jsonl을 벤치마크별로 쪼갠다. 판정불가(v=null)는 분모에서 제외.

사용법: python table11_breakdown.py traces.jsonl judg_gemini.jsonl judg_haiku.jsonl judg_gpt5mini.jsonl judg_deepseek.jsonl -o table11.json
"""
import json, argparse, sys
from collections import defaultdict, Counter

VALID_ITEMS = ['L2-M1', 'L2-M2', 'L2-M3', 'L2-R1', 'L2-R2', 'L2-P1', 'L2-P2', 'L2-P3', 'L2-A1', 'L2-A2', 'L3-01', 'L3-04']
LAYER = {i: ('L2' if i.startswith('L2') else 'L3') for i in VALID_ITEMS}


def load(traces_path, judg_paths):
    bm = {}
    for l in open(traces_path, encoding='utf-8'):
        t = json.loads(l)
        bm[t['trace_id']] = t['benchmark']
    # (model) -> (benchmark) -> item -> [cnt_true, cnt_judged]
    agg = defaultdict(lambda: defaultdict(lambda: Counter()))
    for jf in judg_paths:
        for l in open(jf, encoding='utf-8'):
            row = json.loads(l)
            items = (row.get('result') or {}).get('items', {})
            if not isinstance(items, dict): continue
            b = bm.get(row['trace_id'], '?')
            for iid, v in items.items():
                if iid not in VALID_ITEMS or not isinstance(v, dict): continue
                if v.get('v') is None: continue
                agg[row['model']][b][iid + ':n'] += 1
                agg[row['model']][b][iid + ':t'] += bool(v['v'])
    return agg


def build_table(agg):
    out = {}
    models = sorted(agg)
    benches = sorted({b for m in agg for b in agg[m]})
    for b in benches:
        out[b] = {}
        for iid in VALID_ITEMS:
            rates = []
            for m in models:
                n = agg[m][b].get(iid + ':n', 0)
                t = agg[m][b].get(iid + ':t', 0)
                if n: rates.append(t / n)
            out[b][iid] = {
                'per_model': {m: (agg[m][b].get(iid+':t',0), agg[m][b].get(iid+':n',0)) for m in models if agg[m][b].get(iid+':n',0)},
                'mean_rate_across_models': round(sum(rates) / len(rates), 4) if rates else None,
            }
    return out, models, benches


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('traces')
    ap.add_argument('judg', nargs='+')
    ap.add_argument('-o', '--out', default='table11.json')
    a = ap.parse_args()
    agg = load(a.traces, a.judg)
    table, models, benches = build_table(agg)
    json.dump(table, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    print(f'모델: {models}')
    for b in benches:
        print(f'\n[{b}]')
        for layer in ('L2', 'L3'):
            cells = []
            for iid in VALID_ITEMS:
                if LAYER[iid] != layer: continue
                r = table[b][iid]['mean_rate_across_models']
                n_total = sum(n for _, n in table[b][iid]['per_model'].values())
                cells.append(f'{iid} {r:.0%}(n={n_total})' if r is not None else f'{iid} N/A')
            print(f'  {layer}: ' + '  '.join(cells))
