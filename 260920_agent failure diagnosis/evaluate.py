#!/usr/bin/env python3
"""Exact accuracy, per-trace chance-adjusted accuracy, and checklist coverage.
No LLM calls. --detail adds subsets and individual coverage items.
"""
import argparse
import math
import os
from collections import defaultdict

from checklist import EN, ERROR, NO_ERROR, L0, UPPER
from common import expand_results, find_traces, latest, read_jsonl


def norm(value):
    return str(value).strip().lower() if value else None


def agent_name(value):
    name = norm(value)
    return 'orchestrator' if name and name.startswith('orchestrator (') else name


def corrected(hit, count):
    return (float(hit) - 1 / count) / (1 - 1 / count) if count > 1 else None


def mcnemar_exact(b, c):
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def keys_of(trace):
    return [(trace['benchmark'], trace['subset']), (trace['benchmark'], 'ALL')]


def trace_scores(g, r):
    pred = r.get('pred') or {}
    valid = not r.get('errors')
    scores = {}
    if g['gold'].get('step') is not None:
        hit = bool(valid and pred.get('step') == g['gold']['step'])
        scores['step'] = (hit, corrected(hit, len(g['steps'])))
    if g['gold'].get('agent'):
        candidates = {agent_name(s.get('agent')) for s in g['steps']} - {None}
        target = agent_name(g['gold']['agent'])
        hit = bool(valid and agent_name(pred.get('agent')) == target)
        scores['agent'] = (hit, corrected(hit, len(candidates)) if target in candidates else None)
    if g['gold'].get('module'):
        hit = bool(valid and norm(pred.get('module')) == norm(g['gold']['module']))
        scores['module'] = (hit, None)
    return scores


def metric():
    return {'n': 0, 'hit': 0, 'adjusted_sum': 0.0, 'adjusted_n': 0}


def eval_attr(traces, files):
    gold = {t['trace_id']: t for t in traces}
    groups = defaultdict(dict)
    for f in files:
        for r in latest(read_jsonl(f), key=lambda r: (r['trace_id'], r['method'], r['model'])):
            groups[(r['method'], r['model'])][r['trace_id']] = r
    out = {}
    for key, rows in sorted(groups.items()):
        per = defaultdict(lambda: defaultdict(metric))
        for tid, r in rows.items():
            if tid not in gold:
                continue
            for level, (hit, adj) in trace_scores(gold[tid], r).items():
                for bench in keys_of(gold[tid]):
                    m = per[bench][level]
                    m['n'] += 1
                    m['hit'] += hit
                    if adj is not None:
                        m['adjusted_sum'] += adj
                        m['adjusted_n'] += 1
        out[key] = per
    return out, groups, gold


def print_attr(res, detail=False):
    print('Accuracy: exact match (%). Adjusted: mean((hit - 1/N)/(1 - 1/N)); N=T (step), K (agent).')
    print('Adjusted: chance=0, perfect=1; negatives retained. N<=1 / missing gold candidate excluded from adj_n.')
    print('Agent names normalized; Orchestrator activity suffixes merged. Module: raw only.')
    print(f'{"method":8s} {"model":32s} {"bench":20s} {"level":7s} {"n":>5s} {"accuracy":>9s} {"adjusted":>10s} {"adj_n":>6s}')
    for (method, model), per in res.items():
        for bench, levels in sorted(per.items()):
            if not detail and bench[1] != 'ALL':
                continue
            for level in ('step', 'agent', 'module'):
                if level not in levels:
                    continue
                m = levels[level]
                adj = f"{m['adjusted_sum'] / m['adjusted_n']:.4f}" if m['adjusted_n'] else '-'
                print(f'{method:8s} {model:32s} {"/".join(bench):20s} {level:7s} {m["n"]:5d} '
                      f'{100 * m["hit"] / m["n"]:8.1f}% {adj:>10s} {m["adjusted_n"]:6d}')
    print('Missing predictions / call or parse errors count as incorrect. Missing result rows are not evaluated.')
    print('For methods with different sample sizes, use the common-trace comparison below.')


def print_paired(groups, gold, base='aao', detail=False):
    by_model = defaultdict(dict)
    for (method, model), rows in groups.items():
        by_model[model][method] = rows
    for model, methods in sorted(by_model.items()):
        if base not in methods:
            continue
        for method, rows in sorted(methods.items()):
            if method == base:
                continue
            stats = defaultdict(lambda: {'n': 0, 'b': 0, 'c': 0, 'delta': 0.0, 'adj_n': 0})
            for tid in sorted(set(rows) & set(methods[base]) & set(gold)):
                new = trace_scores(gold[tid], rows[tid])
                old = trace_scores(gold[tid], methods[base][tid])
                for level, (hit, adj) in new.items():
                    hit0, adj0 = old[level]
                    for bench in keys_of(gold[tid]):
                        m = stats[(bench, level)]
                        m['n'] += 1
                        m['b'] += hit and not hit0
                        m['c'] += hit0 and not hit
                        if adj is not None and adj0 is not None:
                            m['delta'] += adj - adj0
                            m['adj_n'] += 1
            print(f'\nPaired {method} - {base} [{model}] (common traces; p: raw exact McNemar)')
            for (bench, level), m in sorted(stats.items()):
                if not detail and bench[1] != 'ALL':
                    continue
                adj = f"{m['delta'] / m['adj_n']:+.4f}" if m['adj_n'] else '-'
                print(f'  {"/".join(bench):20s} {level:7s} n={m["n"]:3d} '
                      f'accuracy {100 * (m["b"] - m["c"]) / m["n"]:+.1f}pp '
                      f'adjusted {adj} (n={m["adj_n"]}) p={mcnemar_exact(m["b"], m["c"]):.3f}')


def coverage_stats(rows):
    items = list(L0) + list(UPPER['L1']) + list(UPPER['L2'])
    rates, hits_by_item = {}, {}
    for iid in items:
        if iid in L0:
            hits = sum((r.get('l0') or {}).get(iid, {}).get('v') is True for r in rows)
        else:
            hits = sum((r.get(iid[:2].lower()) or {}).get(iid, {}).get('status') in (ERROR, NO_ERROR) for r in rows)
        hits_by_item[iid] = hits
        rates[iid] = hits / len(rows) if rows else None
    counts = {label: (sum(hits_by_item[i] for i in ids), len(rows) * len(ids))
              for label, ids in [('L0', list(L0)), ('L1+L2', list(UPPER['L1']) + list(UPPER['L2'])), ('ALL', items)]}
    return rates, counts


def eval_coverage(files, detail=False):
    for f in files:
        rows = latest(read_jsonl(f))
        by = defaultdict(list)
        for r in rows:
            by[(r['benchmark'], r['subset'])].append(r)
        print(f'\nCoverage: {os.path.basename(f)} (n={len(rows)})')
        print('Coverage = assessable (trace, item) pairs / all pairs; L0 recorded / L1+L2 assessable. No threshold.')
        print(f'{"subset":20s} {"n":>5s} | {"L0":>15s} | {"L1+L2":>15s} | {"ALL":>15s}')
        for bench, rs in sorted(by.items()):
            rates, counts = coverage_stats(rs)
            cells = [f'{a:2d}/{n:<2d} ({100*a/n:5.1f}%)' for a, n in counts.values()]
            print(f'{"/".join(bench):20s} {len(rs):5d} | ' + ' | '.join(cells))
            if detail:
                for iid, rate in rates.items():
                    print(f'  {iid:6s} {rate:6.1%}  {EN[iid]}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('traces', nargs='?', default=None)
    ap.add_argument('--attr', nargs='*', help='attribution files (default: all if --ckl not given)')
    ap.add_argument('--ckl', nargs='*', help='checklist files for coverage')
    ap.add_argument('--base', default='aao')
    ap.add_argument('--detail', action='store_true', help='include subsets and coverage items')
    ap.add_argument('--coverage', action='store_true', help='include item coverage; requires --ckl')
    a = ap.parse_args()
    if a.coverage and not a.ckl:
        ap.error('--coverage requires --ckl')
    if a.ckl:
        files = expand_results('ckl', a.ckl)
        if not files:
            ap.error('no checklist files found')
        eval_coverage(files, a.detail or a.coverage)
    if a.attr is not None or not a.ckl:
        files = expand_results('attribution', a.attr or ['attr_*.jsonl'])
        if not files:
            ap.error('no attribution files found')
        res, groups, gold = eval_attr(read_jsonl(find_traces(a.traces)), files)
        for (method, model), rows in sorted(groups.items()):
            unmatched = len(set(rows) - set(gold))
            if unmatched:
                print(f'{method} [{model}]: {unmatched} result rows excluded (trace absent from current gold data).')
        print_attr(res, a.detail)
        print_paired(groups, gold, a.base, a.detail)


if __name__ == '__main__':
    main()
