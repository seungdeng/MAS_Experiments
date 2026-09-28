#!/usr/bin/env python3
"""evaluate.py — 결과 집계. 기본 출력은 한 화면에 들어오는 요약이고, --detail 을 주면 전체 표를 낸다.

  귀인 성능 (베이스라인 vs 체크리스트 포함, 모델별):
    python evaluate.py                       (results/attribution/attr_*.jsonl 전부)
    python evaluate.py --attr attr_aao_*.jsonl attr_ckl_*.jsonl
  체크리스트 판정 통계:
    python evaluate.py --ckl ckl_*.jsonl     (results/ckl/ 에서 찾는다)
  전체 표(서브셋별·항목별): 위 명령에 --detail
  항목별 커버리지(판정할 수 있었던 비율): python evaluate.py --ckl ckl_google_gemini-3.7-flash.jsonl --coverage

귀인 지표: 골드 단계가 있는 궤적 기준 단계 정확일치(exact)/±k, Who&When 은 에이전트 일치율, AEB 는 모듈 일치율.
호출 실패·파싱 실패는 오답으로 센다(분모 유지) — fail 열에 별도 표기.
쌍대 비교: 같은 모델의 ckl vs aao 를 공통 궤적에서 비교 (b=ckl만 정답, c=aao만 정답, McNemar 정확검정).
"""
import argparse
import math
import os
import sys
from collections import Counter, defaultdict

from checklist import EN, ERROR, INDET, L0, NO_ERROR, UPPER
from common import expand_results, find_traces, latest, read_jsonl

KS = (0, 1, 3, 5)
BENCH = ('AEB', 'AgentRx', 'WhoWhen')
SUBSETS = [('AEB', 'ALFWorld'), ('AEB', 'WebShop'), ('AEB', 'GAIA'), ('AgentRx', 'tau_retail'),
           ('WhoWhen', 'AG'), ('WhoWhen', 'HC')]


def norm(x):
    return str(x).strip().lower() if x else None


def keys_of(t):
    return [(t['benchmark'], t['subset']), (t['benchmark'], 'ALL')]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def pct(x, n, w=5):
    return f'{100 * x / n:{w}.1f}' if n else '  -'.rjust(w)


# ---------------------------------------------------------------- attribution
def eval_attr(traces, files):
    gold = {t['trace_id']: t for t in traces}
    groups = defaultdict(dict)                       # (method, model) -> {trace_id: row}
    for f in files:
        for r in latest(read_jsonl(f), key=lambda r: (r['trace_id'], r['method'], r['model'])):
            groups[(r['method'], r['model'])][r['trace_id']] = r
    out = {}
    for (method, model), rows in sorted(groups.items()):
        per = defaultdict(lambda: {'n': 0, 'fail': 0, 'hit': Counter(), 'agent': [0, 0], 'module': [0, 0]})
        for tid, r in rows.items():
            g = gold.get(tid)
            if not g or g['gold']['step'] is None:
                continue
            p = r.get('pred') or {}
            ps = p.get('step')
            for k in keys_of(g):
                m = per[k]
                m['n'] += 1
                if ps is None or r.get('errors'):
                    m['fail'] += 1
                else:
                    for w in KS:
                        m['hit'][w] += abs(ps - g['gold']['step']) <= w
                if g['gold'].get('agent'):
                    m['agent'][1] += 1
                    m['agent'][0] += norm(p.get('agent')) == norm(g['gold']['agent'])
                if g['gold'].get('module'):
                    m['module'][1] += 1
                    m['module'][0] += norm(p.get('module')) == norm(g['gold']['module'])
        out[(method, model)] = per
    return out, groups, gold


def print_attr_compact(res):
    """한 행 = (방식, 모델). 벤치마크 3묶음(전체)만 exact / ±3 (+ 모듈·에이전트) 로 요약."""
    ns = {b: max(per.get((b, 'ALL'), {}).get('n', 0) for per in res.values()) for b in BENCH}
    print(f'귀인 성능 (%).  n: AEB {ns["AEB"]} / AgentRx {ns["AgentRx"]} / Who&When {ns["WhoWhen"]}   (exact=단계 정확일치, ±3=3단계 이내)')
    print(f'{"":6s} {"":34s} | {"AEB":^20s} | {"AgentRx":^11s} | {"Who&When":^20s} |')
    print(f'{"method":6s} {"model":34s} | {"exact":>5s} {"±3":>5s} {"module":>7s} | {"exact":>5s} {"±3":>5s} | {"exact":>5s} {"±3":>5s} {"agent":>7s} | {"fail":>4s} | {"평가한 n (A/R/W)":>17s}')
    for (method, model), per in res.items():
        cells, fail = [], 0
        for b in BENCH:
            m = per.get((b, 'ALL'))
            if not m:
                cells.append(None)
                continue
            fail += m['fail']
            cells.append(m)
        def c(m, w):
            return pct(m['hit'][w], m['n']) if m else '    -'
        def x(m, key):
            return pct(*m[key], w=7) if m else '      -'
        a, t, w_ = cells
        print(f'{method:6s} {model:34s} | {c(a, 0)} {c(a, 3)} {x(a, "module")} | {c(t, 0)} {c(t, 3)} | '
              f'{c(w_, 0)} {c(w_, 3)} {x(w_, "agent")} | {fail:4d} | ' + '/'.join(f'{m["n"] if m else 0:>3d}' for m in cells).rjust(17)
              + ('  <- 일부만 실행됨' if any((m["n"] if m else 0) < ns[b] for m, b in zip(cells, BENCH)) else ''))
    print('fail = 호출·파싱 실패(오답으로 계산됨). n이 다른 행끼리는 직접 비교하지 말 것(쌍대 비교는 공통 궤적만 사용). 서브셋별 표는 --detail')


def print_attr_detail(res):
    print(f'{"method":6s} {"model":34s} {"bench":18s} {"n":>4s} {"fail":>4s} {"exact":>7s} {"±1":>7s} {"±3":>7s} {"±5":>7s} {"agent":>7s} {"module":>7s}')
    for (method, model), per in res.items():
        for k in sorted(per):
            m = per[k]
            p_ = lambda x, n: f'{x / n:6.1%}' if n else '     -'
            print(f'{method:6s} {model:34s} {"/".join(k):18s} {m["n"]:4d} {m["fail"]:4d} '
                  + ' '.join(p_(m['hit'][w], m['n']) for w in KS)
                  + f' {p_(*m["agent"])} {p_(*m["module"])}')


def paired_counts(groups, gold, base):
    """-> {(model, method): {(w, bench_key): [b, c, n]}}"""
    by_model = defaultdict(dict)
    for (method, model), rows in groups.items():
        by_model[model][method] = rows
    out = {}
    for model, ms in sorted(by_model.items()):
        if base not in ms:
            continue
        for method, rows in ms.items():
            if method == base:
                continue
            cnt = defaultdict(lambda: [0, 0, 0])
            for tid in set(rows) & set(ms[base]):
                g = gold.get(tid)
                if not g or g['gold']['step'] is None:
                    continue
                for w in (0, 3):
                    ok = lambda r: (r['pred'].get('step') is not None and not r.get('errors')
                                    and abs(r['pred']['step'] - g['gold']['step']) <= w)
                    a, b0 = ok(rows[tid]), ok(ms[base][tid])
                    for k in keys_of(g):
                        cnt[(w, k)][0] += a and not b0
                        cnt[(w, k)][1] += b0 and not a
                        cnt[(w, k)][2] += 1
            out[(model, method)] = cnt
    return out


def print_paired(groups, gold, base='aao', detail=False):
    res = paired_counts(groups, gold, base)
    if not res:
        print(f'\n(쌍대 비교 없음: 같은 모델에서 "{base}" 와 다른 방식의 결과가 둘 다 있어야 한다)')
        return
    print(f'\n쌍대 비교: 각 방식 − {base}  (같은 모델, 공통 궤적).  Δ=정답률 차(%p), p=McNemar 정확검정, +면 {base}보다 좋음')
    for (model, method), cnt in res.items():
        print(f'  {method} vs {base}  [{model}]')
        for k in (sorted(k for (_, k) in cnt) if detail else [(b, 'ALL') for b in BENCH]):
            parts = []
            for w in (0, 3):
                if (w, k) in cnt:
                    b, c, n = cnt[(w, k)]
                    parts.append(f'{"exact" if w == 0 else "±3":>5s} {100 * (b - c) / n:+5.1f}%p (p={mcnemar_exact(b, c):.2f}, +{b}/-{c})')
                    nn = n
            if parts:
                print(f'    {"/".join(k):16s} n={nn:3d}  ' + '   '.join(parts))
        # b/c 가 궁금하면: +b = 이 방식만 맞춤, -c = 기준만 맞춤


# ---------------------------------------------------------------- checklist stats
def _layer_stats(rs, layer):
    tot = a = e = 0
    for r in rs:
        for x in r[layer.lower()].values():
            tot += 1
            a += x['status'] != INDET
            e += x['status'] == ERROR
    return tot, a, e


def _flag_hit(rs, gold, w=3):
    hit = m = 0
    for r in rs:
        g = gold.get(r['trace_id'])
        steps = [x['step'] for layer in ('l1', 'l2') for x in r[layer].values()
                 if x['status'] == ERROR and x.get('step') is not None]
        if g and g['gold']['step'] is not None and steps:
            m += 1
            hit += abs(min(steps) - g['gold']['step']) <= w
    return hit, m


def eval_ckl_compact(traces, files):
    gold = {t['trace_id']: t for t in traces}
    for f in files:
        rows = latest(read_jsonl(f))
        nerr = sum(1 for r in rows if r.get('errors'))
        print(f'\n===== {os.path.basename(f)}  (model={rows[0]["model"] if rows else "?"}, {len(rows)} traces, 호출 오류 {nerr}건) =====')
        by = defaultdict(list)
        for r in rows:
            by[(r['benchmark'], r['subset'])].append(r)
        print('\n[1] L0: 로그에 기록돼 있다고 판정된 비율 (%)')
        print(f'{"subset":16s} {"n":>4s} | ' + ' '.join(f'{i:>5s}' for i in L0))
        for k in SUBSETS:
            rs = by.get(k)
            if rs:
                print(f'{"/".join(k):16s} {len(rs):4d} | ' + ' '.join(pct(sum(1 for r in rs if r['l0'][i]['v'] is True), len(rs)) for i in L0))
        print('\n[2] L1/L2: 판단 가능 비율(assess)과, 판단 가능했던 것 중 오류 비율(err) (%)   /  체크리스트로 찍은 오류 step')
        print(f'{"subset":16s} {"n":>4s} | {"L1 assess":>9s} {"L1 err":>7s} | {"L2 assess":>9s} {"L2 err":>7s} | {"step±3 적중":>14s}')
        for k in SUBSETS:
            rs = by.get(k)
            if not rs:
                continue
            cells = []
            for layer in ('L1', 'L2'):
                tot, a, e = _layer_stats(rs, layer)
                cells += [pct(a, tot, 9), pct(e, a, 7)]
            hit, m = _flag_hit(rs, gold)
            hs = f'{pct(hit, m)} (n={m})' if m else '-'
            print(f'{"/".join(k):16s} {len(rs):4d} | {cells[0]} {cells[1]} | {cells[2]} {cells[3]} | {hs:>14s}')
        print('\nassess = 요구 L0 가 충족돼 판단 가능했던 비율(낮으면 L0 미충족으로 "판단 불가"가 많다는 뜻).')
        print('err    = 판단 가능했던 항목 중 오류로 판정된 비율. 판단 가능 표본이 작으면(assess 낮음) 신뢰하기 어렵다.')
        print('step±3 = 체크리스트가 찍은 가장 이른 오류 step 이 골드와 3단계 이내인 비율(n=찍은 궤적 수). 항목별 표는 --detail')


def eval_ckl_detail(traces, files):
    gold = {t['trace_id']: t for t in traces}
    for f in files:
        rows = latest(read_jsonl(f))
        by = defaultdict(list)
        for r in rows:
            for k in [(r['benchmark'], r['subset']), (r['benchmark'], 'ALL')]:
                by[k].append(r)
        print(f'\n===== {os.path.basename(f)}  (model={rows[0]["model"] if rows else "?"}, {len(rows)} traces, '
              f'{sum(1 for r in rows if r.get("errors"))} with call errors) =====')
        for k in sorted(by):
            rs = by[k]
            n = len(rs)
            print(f'\n[{"/".join(k)}] n={n}')
            print('  L0 recorded: ' + '  '.join(f'{i}:{sum(1 for r in rs if r["l0"][i]["v"] is True) / n:.0%}' for i in L0))
            print(f'  {"item":6s} {"ERROR":>7s} {"NO_ERR":>7s} {"INDET:dep":>10s} {"INDET:llm":>10s} {"assessable":>11s}')
            for layer in ('L1', 'L2'):
                for iid in UPPER[layer]:
                    c = Counter()
                    for r in rs:
                        x = r[layer.lower()][iid]
                        c[x['status'] if x['status'] != INDET else 'dep' if (x['reason'] or '').startswith('dep') else 'llm'] += 1
                    print(f'  {iid:6s} {c[ERROR] / n:7.1%} {c[NO_ERROR] / n:7.1%} {c["dep"] / n:10.1%} {c["llm"] / n:10.1%} '
                          f'{(c[ERROR] + c[NO_ERROR]) / n:11.1%}')
            hit, m = Counter(), 0
            for r in rs:
                g = gold.get(r['trace_id'])
                steps = [x['step'] for layer in ('l1', 'l2') for x in r[layer].values()
                         if x['status'] == ERROR and x.get('step') is not None]
                if g and g['gold']['step'] is not None and steps:
                    m += 1
                    for w in KS:
                        hit[w] += abs(min(steps) - g['gold']['step']) <= w
            if m:
                print(f'  earliest flagged ERROR step vs gold (n={m}): ' + ' '.join(f'±{w}:{hit[w] / m:.1%}' for w in KS))


# ---------------------------------------------------------------- coverage per item
def eval_coverage(files, p=0.9):
    """항목별 커버리지 = 그 항목을 판정할 수 있었던 궤적 비율.
    L0 = 로그에 기록됨(v=true) 비율 / L1·L2 = 판단 가능(요구 L0 충족 + LLM 이 null 아님) 비율.
    서브셋에서 비율 >= p 이면 그 항목은 '판정가능'(*). 서브셋 커버리지 = 판정가능 항목 수 / 18."""
    items = list(L0) + list(UPPER['L1']) + list(UPPER['L2'])
    for f in files:
        rows = latest(read_jsonl(f))
        by = defaultdict(list)
        for r in rows:
            by[(r['benchmark'], r['subset'])].append(r)
        cols = [k for k in SUBSETS if k in by]
        print(f'\n===== 항목별 커버리지: {rows[0]["model"] if rows else "?"}  ({os.path.basename(f)}) =====')
        print(f'커버리지 = 판정할 수 있었던 궤적 비율(%). *는 p>={p:.0%} 라서 그 서브셋에서 "판정가능".')
        print('L0 = 로그에 기록됨 / L1·L2 = 판단 가능(요구 L0 충족 + LLM 답 있음)\n')
        print(f'{"item":6s} | ' + ' '.join(f'{"/".join(k)[:12]:>12s}' for k in cols) + f' | {"질문(요약)":s}')
        print(f'{"n":6s} | ' + ' '.join(f'{len(by[k]):>12d}' for k in cols) + ' |')
        ok = {k: Counter() for k in cols}
        for iid in items:
            cells = []
            for k in cols:
                rs = by[k]
                if iid in L0:
                    x = sum(1 for r in rs if r['l0'][iid]['v'] is True)
                else:
                    layer = 'l1' if iid.startswith('L1') else 'l2'
                    x = sum(1 for r in rs if r[layer][iid]['status'] != INDET)
                rate = x / len(rs)
                good = rate >= p
                ok[k][iid[:2]] += good
                cells.append(f'{100 * rate:11.0f}{"*" if good else " "}')
            print(f'{iid:6s} | ' + ' '.join(cells) + f' | {EN[iid][:52]}')
        print('-' * 8)
        for lay, total in (('L0', len(L0)), ('L1', len(UPPER['L1'])), ('L2', len(UPPER['L2']))):
            print(f'{lay + " 판정가능":6s} | ' + ' '.join(f'{ok[k][lay]:>8d}/{total:<3d}' for k in cols))
        print(f'{"전체":6s} | ' + ' '.join(f'{sum(ok[k].values()):>8d}/{len(items):<3d}' for k in cols)
              + '   <- 커버리지(판정가능 항목 수 / 18)')
        print(f'{"":6s} | ' + ' '.join(f'{100 * sum(ok[k].values()) / len(items):>11.1f}%' for k in cols))


# ---------------------------------------------------------------- agreement between judge models
def kappa_ac1(pairs):
    """[(a, b)] -> (n, Po, Cohen kappa, Gwet AC1). Same definitions as the 260831 kappa.py."""
    n = len(pairs)
    if not n:
        return 0, None, None, None
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    po = sum(1 for a, b in pairs if a == b) / n
    c1, c2 = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((c1[l] / n) * (c2[l] / n) for l in labels)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    q = len(labels)
    if q < 2:
        ac1 = 1.0
    else:
        pi = {l: (c1[l] + c2[l]) / (2 * n) for l in labels}
        pe_g = sum(p * (1 - p) for p in pi.values()) / (q - 1)
        ac1 = (po - pe_g) / (1 - pe_g) if pe_g < 1 else 1.0
    return n, po, kappa, ac1


def _row(name, st, extra=''):
    n, po, k, g = st
    if not n:
        return f'  {name:24s} {"-":>6s}'
    return f'  {name:24s} {n:6d} {100 * po:6.1f} {k:6.2f} {g:6.2f}{extra}'


def eval_agreement(files):
    """Two or more ckl files: how much do the judge models agree? (the L0 gate and the L1/L2 verdicts)"""
    data = []
    for f in files:
        rows = {r['trace_id']: r for r in latest(read_jsonl(f))}
        data.append((os.path.basename(f), rows))
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            (fa, ra), (fb, rb) = data[i], data[j]
            common_ids = sorted(set(ra) & set(rb))
            ma, mb = next(iter(ra.values()))['model'], next(iter(rb.values()))['model']
            print(f'\n===== 판정 모델 간 일치: {ma}  vs  {mb}   (공통 궤적 {len(common_ids)}) =====')
            print(f'  {"":24s} {"n":>6s} {"일치%":>6s} {"kappa":>6s} {"AC1":>6s}   (AC1 = 쏠림에 강한 지표)')
            print('  [L0] 로그에 기록됨 true/false')
            for iid in L0:
                pairs = [(ra[t]['l0'][iid]['v'] is True, rb[t]['l0'][iid]['v'] is True) for t in common_ids]
                ta, tb = sum(a for a, _ in pairs), sum(b for _, b in pairs)
                print(_row(iid, kappa_ac1(pairs), f'   true {100 * ta / len(pairs):3.0f}% vs {100 * tb / len(pairs):3.0f}%'))
            print('  [L1/L2] 판단 가능 여부, 그리고 둘 다 판단 가능한 것의 오류 여부')
            for layer in ('l1', 'l2'):
                asm, err = [], []
                for t in common_ids:
                    for iid, xa in ra[t][layer].items():
                        xb = rb[t][layer][iid]
                        aa, ab = xa['status'] != INDET, xb['status'] != INDET
                        asm.append((aa, ab))
                        if aa and ab:
                            err.append((xa['status'] == ERROR, xb['status'] == ERROR))
                print(_row(layer.upper() + ' 판단 가능 여부', kappa_ac1(asm)))
                print(_row(layer.upper() + ' 오류 여부(양쪽 가능)', kappa_ac1(err)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('traces', nargs='?', default=None, help='default: data/traces.jsonl')
    ap.add_argument('--attr', nargs='*', help='attribution result files (default: attr_*.jsonl)')
    ap.add_argument('--ckl', nargs='*', help='checklist judgement files (ckl_*.jsonl)')
    ap.add_argument('--base', default='aao', help='baseline method for the paired comparison')
    ap.add_argument('--detail', action='store_true', help='full tables (per subset / per item)')
    ap.add_argument('--coverage', action='store_true', help='with --ckl: per-item coverage table (share of traces the item could be judged)')
    ap.add_argument('--p', type=float, default=0.9, help='threshold for "determinable" in --coverage (default 0.9)')
    a = ap.parse_args()
    traces = read_jsonl(find_traces(a.traces))
    if a.ckl:
        files = expand_results('ckl', a.ckl)
        if not files:
            sys.exit('no checklist files found (results/ckl/ckl_*.jsonl)')
        if a.coverage:
            eval_coverage(files, a.p)
            sys.exit(0)
        (eval_ckl_detail if a.detail else eval_ckl_compact)(traces, files)
        if len(files) >= 2:
            eval_agreement(files)
    if a.attr is not None or not a.ckl:
        files = expand_results('attribution', a.attr or ['attr_*.jsonl'])
        if not files:
            sys.exit('no attribution files found (results/attribution/attr_*.jsonl)')
        res, groups, gold = eval_attr(traces, files)
        (print_attr_detail if a.detail else print_attr_compact)(res)
        print_paired(groups, gold, a.base, a.detail)
