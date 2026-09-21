#!/usr/bin/env python3
"""recount_260831.py — 구판(260831) 결과 중 '사후 보정으로 고칠 수 있는 것'만 다시 센다. LLM 호출 없음.

  Table 8  L3-01 정답 정합: AEB 단계 번호가 1-based 로 표시됐는데 골드는 0-based 였다.
           판정 LLM 은 화면에 보인 [step k] 를 그대로 답했으므로, AEB 는 (예측 − 1) 과 0-based 골드를 비교해야 한다.
           (Who&When·TRAIL 은 0-based 라벨이라 그대로.)  → 재실행 불필요.
  Table 9  E1 커버리지: AEB 모듈 태그(<plan> 등) 검출이 content+obs 전체에서 이뤄져, obs(=프롬프트 템플릿)에 들어 있는
           안내문까지 '기록됨'으로 셌다.  → 에이전트 출력 + 실제 환경 관찰(신판 normalize 의 추출본)로 같은 규칙을 다시 적용.

보정 불가(재실행 필요): Table 7(모델 간 일치)·Table 11(발생률) — 판정 LLM 이 받은 입력 자체가 열화돼 있었다.
  (이번 실험은 새 체크리스트·새 파이프라인으로 다시 돌리므로 이 표들은 대체된다.)

사용법: python recount_260831.py [--old-dir <구판 afd_full_pipeline 경로>] [-o recount_260831_report.txt]
  원본 폴더는 읽기만 한다. 산출: results/recount_260831/recount_260831_report.txt, recount_260831.json
"""
import argparse
import ast
import importlib.util
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OLD = os.path.join(HERE, '..', '260831_agent failure diagnosis', '실험', 'afd_full_pipeline')
KS = (0, 1, 3, 5)
# 실험결과_정리.txt(2026-09-03)의 Table 8 — 이 스크립트의 'as-run' 재현이 맞는지 검증하는 기준값
EXPECTED_AS_RUN = {'claude-haiku': (331, 15.4, 39.3, 58.9, 70.4), 'deepseek-flash': (159, 8.2, 39.6, 60.4, 69.8),
                   'gemini-flash': (294, 28.9, 46.9, 67.0, 75.9), 'gpt5-mini': (179, 8.9, 39.1, 60.9, 72.6)}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- Table 8
def parse_result(res):
    if isinstance(res, str):
        try:
            res = ast.literal_eval(res)
        except Exception:
            return None
    return res if isinstance(res, dict) and isinstance(res.get('items'), dict) else None


def l3_01_pairs(judg_path, gold):
    """구판 aggregate 와 같은 규칙: (trace, item) 마지막 응답 우선, L3-01 의 step 이 있고 골드 단계가 있는 것.
    -> {model: [(benchmark, pred_as_shown, gold0)]}"""
    last = defaultdict(dict)                       # model -> trace_id -> item dict
    for r in read_jsonl(judg_path):
        if r.get('variant', 'a') != 'a':
            continue
        res = parse_result(r.get('result'))
        if res and 'L3-01' in res['items']:
            last[r['model']][r['trace_id']] = res['items']['L3-01']
    out = defaultdict(list)
    for model, d in last.items():
        for tid, it in d.items():
            g = gold.get(tid)
            if not g or g['gold']['step'] is None or not isinstance(it, dict) or it.get('step') is None:
                continue
            try:
                out[model].append((g['benchmark'], int(it['step']), g['gold']['step']))
            except (TypeError, ValueError):
                pass
    return out


def rates(diffs):
    n = len(diffs)
    return n, [100 * sum(1 for d in diffs if d <= k) / n if n else 0.0 for k in KS]


def table8(old_dir, gold_by_id):
    res = {}
    for f in sorted(os.listdir(old_dir)):
        if f.startswith('judg_') and f.endswith('.jsonl'):
            for model, pairs in l3_01_pairs(os.path.join(old_dir, f), gold_by_id).items():
                as_run = [abs(p - g) for b, p, g in pairs]
                fixed = [abs(p - (1 if b == 'AEB' else 0) - g) for b, p, g in pairs]     # AEB labels were 1-based
                entry = {'all': {'as_run': rates(as_run), 'corrected': rates(fixed)}}
                for bm in ('AEB', 'WhoWhen', 'TRAIL'):
                    sel = [(b, p, g) for b, p, g in pairs if b == bm]
                    if sel:
                        entry[bm] = {'as_run': rates([abs(p - g) for b, p, g in sel]),
                                     'corrected': rates([abs(p - (1 if b == 'AEB' else 0) - g) for b, p, g in sel])}
                res[model] = entry
    return res


# ---------------------------------------------------------------- Table 9
LAYER_TOTALS = {'L0': 'L0', 'L1': 'L1', 'L2': 'L2', 'L3': 'L3'}


def layer_counts(t7row):
    c = Counter(i[:2] for i in t7row['determinable'])
    return {k: c.get(k, 0) for k in LAYER_TOTALS}


def table9(old_dir, new_traces_path):
    e1 = load_module('old_e1', os.path.join(old_dir, 'e1_coverage.py'))
    oldn = load_module('old_normalize', os.path.join(old_dir, 'normalize.py'))
    old = read_jsonl(os.path.join(old_dir, 'traces.jsonl'))
    new_steps = {t['trace_id']: t for t in read_jsonl(new_traces_path) if t['benchmark'] == 'AEB'}

    changed = defaultdict(Counter)                  # subset -> flag -> #traces whose flag changed
    fixed = []
    for t in old:
        t2 = json.loads(json.dumps(t))
        if t['benchmark'] == 'AEB':
            nt = new_steps[t['trace_id']]
            steps = nt['steps']
            blob = '\n'.join(s['content'] + s['obs'] for s in steps)         # agent output + REAL observation
            sub = t['subset']
            marker = oldn.AEB_ERR[sub]
            a = t2['avail']
            a['module_tags'] = 'E' if oldn.MODULE_TAG.search(blob) else None
            a['observation'] = 'E' if all(s['obs'] for s in steps[:-1]) else ('B' if any(s['obs'] for s in steps) else None)
            a['step_limit_cfg'] = 'B' if oldn.AEB_LIMIT.search(blob) else None
            a['error_msg'] = 'E' if (marker and marker in blob) else ('B' if oldn.ERR_PAT.search(blob) else None)
            for k, tag in (('action_field', 'action'), ('plan_field', 'plan'), ('memory_field', 'memory'),
                           ('reflection_field', 'reflection')):
                a[k] = 'E' if f'<{tag}>' in blob else None
            for k in a:
                if a[k] != t['avail'].get(k):
                    changed[sub][k] += 1
        fixed.append(t2)
    o7, c7 = e1.table7(old), e1.table7(fixed)
    return o7, c7, changed


# ---------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old-dir', default=DEFAULT_OLD)
    ap.add_argument('--traces', default=os.path.join(HERE, 'data', 'traces.jsonl'), help='신판 traces.jsonl (AEB 관찰 추출본)')
    ap.add_argument('-o', '--out', default=os.path.join(HERE, 'results', 'recount_260831', 'recount_260831_report.txt'))
    a = ap.parse_args()
    sys.stdout.reconfigure(errors='replace')
    old_dir = os.path.abspath(a.old_dir)
    gold = {t['trace_id']: t for t in read_jsonl(os.path.join(old_dir, 'traces.jsonl'))}
    L, js = [], {}

    def p(s=''):
        L.append(s)
        print(s)

    # ---- Table 8
    t8 = table8(old_dir, gold)
    p('=' * 88)
    p('Table 8 (보정) 최초 오류 단계(L3-01) 정답 정합 — AEB 단계 라벨이 1-based 였으므로 AEB 예측을 -1 하여 0-based 골드와 비교')
    p('=' * 88)
    p(f'{"model":15s} {"n":>4s} | {"as-run (구판 표)":^33s} | {"corrected (보정)":^33s}')
    p(f'{"":15s} {"":>4s} | {"exact":>7s} {"±1":>7s} {"±3":>7s} {"±5":>7s} | {"exact":>7s} {"±1":>7s} {"±3":>7s} {"±5":>7s}')
    mism = []
    for model in ('claude-haiku', 'deepseek-flash', 'gemini-flash', 'gpt5-mini'):
        if model not in t8:
            continue
        n, ar = t8[model]['all']['as_run']
        _, cr = t8[model]['all']['corrected']
        p(f'{model:15s} {n:4d} | ' + ' '.join(f'{x:6.1f}%' for x in ar) + ' | ' + ' '.join(f'{x:6.1f}%' for x in cr))
        exp = EXPECTED_AS_RUN.get(model)
        if exp and (n != exp[0] or any(abs(x - e) > 0.05 for x, e in zip(ar, exp[1:]))):
            mism.append(model)
    p('\n벤치마크별 (n / exact ±1 ±3 ±5)   as-run → corrected')
    for model in t8:
        for bm in ('AEB', 'WhoWhen', 'TRAIL'):
            if bm in t8[model]:
                n, ar = t8[model][bm]['as_run']
                _, cr = t8[model][bm]['corrected']
                tag = '' if bm != 'TRAIL' else '  (n 매우 작음)'
                p(f'  {model:15s} {bm:8s} n={n:3d}  ' + '/'.join(f'{x:5.1f}' for x in ar) + '  →  ' + '/'.join(f'{x:5.1f}' for x in cr) + tag)
    p('\n[검증] as-run 재현이 실험결과_정리.txt 의 Table 8 과 ' + ('일치' if not mism else f'불일치: {mism}  ← 확인 필요'))
    p('※ 가정: 판정 LLM 이 AEB 로그에 표시된 1-based [step k] 라벨을 그대로 답했다. Who&When·TRAIL 은 0-based 라벨이라 보정 없음.')
    js['table8'] = t8

    # ---- Table 9
    o7, c7, changed = table9(old_dir, a.traces)
    p('\n' + '=' * 88)
    p('Table 9 (보정) 층위별 진단가능성 커버리지 (24항목, p>=0.9) — AEB 모듈 태그를 에이전트 출력 + 실제 관찰에서만 검출')
    p('=' * 88)
    p(f'{"benchmark(n)":22s} | {"L0/7":>5s} {"L1/4":>5s} {"L2/10":>6s} {"L3/3":>5s} {"전체/24":>9s} | 구분')
    rows = [('AEB/ALFWorld', 'AEB ALFWorld'), ('AEB/WebShop', 'AEB WebShop'), ('AEB/GAIA', 'AEB GAIA'),
            ('AEB/ALL', 'AEB ALL'), ('TRAIL/ALL', 'TRAIL ALL'), ('WhoWhen/AG', 'Who&When AG'), ('WhoWhen/HC', 'Who&When HC')]
    for key, label in rows:
        for tag, t7 in (('구판', o7), ('보정', c7)):
            r = t7[key]
            lc = layer_counts(r)
            mark = ''
            if tag == '보정' and r['determinable'] != o7[key]['determinable']:
                lost = sorted(set(o7[key]['determinable']) - set(r['determinable']))
                gain = sorted(set(r['determinable']) - set(o7[key]['determinable']))
                mark = '  변경: ' + (f'-{" -".join(lost)}' if lost else '') + (f' +{" +".join(gain)}' if gain else '')
            name = '%s (%d)' % (label, r['n'])
            p(f'{name:22s} | {lc["L0"]:5d} {lc["L1"]:5d} {lc["L2"]:6d} {lc["L3"]:5d} '
              f'{r["coverage"]:8.1%} | {tag}{mark}')
    p('\n보정으로 바뀐 필드 수 (AEB, 서브셋별 trace 수):')
    for sub in ('ALFWorld', 'WebShop', 'GAIA'):
        p(f'  {sub:9s} ' + (', '.join(f'{k}:{v}' for k, v in sorted(changed[sub].items())) or '변경 없음'))
    js['table9'] = {'old': {k: v['coverage'] for k, v in o7.items()}, 'corrected': {k: v['coverage'] for k, v in c7.items()},
                    'changed_flags': {k: dict(v) for k, v in changed.items()}}
    p('\n※ TRAIL·Who&When 은 이 오류의 영향을 받지 않아 구판과 동일하다. ALL(AEB 전체) 행은 위 서브셋 합산 기준.')
    p('※ Table 7(모델 간 일치)·Table 11(발생률)은 판정 입력이 열화돼 있어 사후 보정이 불가능하다 → 신판 파이프라인으로 재실행.')

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    with open(os.path.splitext(a.out)[0].replace('_report', '') + '.json', 'w', encoding='utf-8') as f:
        json.dump(js, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
