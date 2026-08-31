#!/usr/bin/env python3
"""e3_judge.py — E3 LLM 판정 파이프라인 (Table 6·8·9) + 비용 견적.

사용법:
  python3 e3_judge.py traces.jsonl --estimate                    # 비용 견적만 (API 불필요)
  ANTHROPIC_API_KEY=sk-... python3 e3_judge.py traces.jsonl \
      --run --model sonnet --out judg_sonnet.jsonl               # 본 실행 (중단 후 재실행 시 이어서)
  python3 e3_judge.py traces.jsonl --run --model haiku --out judg_haiku.jsonl
  python3 e3_judge.py traces.jsonl --run --model sonnet --variant b --out judg_sonnet_vb.jsonl
  python3 e3_judge.py traces.jsonl --aggregate judg_sonnet.jsonl judg_haiku.jsonl  # 집계·κ·골드정합

설계 (paper_final IV장, 2026-08-18 개정 반영):
- 모듈군 4분할 프롬프트: M(L2-M1~M3) / R(L2-R1~R2) / P(L2-P1~P3) / A(L2-A1~A2) + T(L3-01·04)
  = trace당 5콜. JSON 강제, 파싱 실패 시 1회 재시도.
- 판정 모델: 동일 계열 상·하위 (sonnet=claude-sonnet-4-6, haiku=claude-haiku-4-5-20251001).
- 프롬프트 변형 b: 문항 순서 역순 + 지시문 재서술 (자기 일관성용).
- 절단 정책: 스텝당 1,200자 + 트레이스 총 100,000자 (초과 시 앞 60%/뒤 40% 보존, 중간 생략 표기).
- 시드성 재현: temperature=0. 골드 라벨은 프롬프트에 절대 미포함.
"""
import json, os, sys, argparse, time, re, urllib.request

MODELS = {'sonnet': 'claude-sonnet-4-6', 'haiku': 'claude-haiku-4-5-20251001'}
PRICE = {'sonnet': (3.0, 15.0), 'haiku': (1.0, 5.0)}   # $/MTok (in, out) 2026-08 표준가; 배치 API는 50% 할인
STEP_CAP, TRACE_CAP, OUT_TOK = 1200, 100_000, 700

GROUPS = {
 'M': [('L2-M1','이전 기록에 없는 사실을 회상하는가'),('L2-M2','기록에 있는 정보를 회상하지 못하는가'),('L2-M3','요약이 과제 관련 핵심 세부를 누락하는가')],
 'R': [('L2-R1','진행 상황을 과대/과소평가하는가'),('L2-R2','직전 관찰 결과를 잘못 해석하는가')],
 'P': [('L2-P1','과제 제약과 모순되는 계획을 세우는가'),('L2-P2','전제조건상 불가능한 행동을 계획하는가'),('L2-P3','동일 전략 고수·탐색 낭비가 지속되는가')],
 'A': [('L2-A1','행동이 직전 계획과 불일치하는가'),('L2-A2','행동 인자·대상 지정이 잘못되는가')],
 'T': [('L3-01','최초의 결정적 오류 단계는 어디인가 (step_idx로 답하라, 없으면 null)'),('L3-04','최초 오류 이후 자기 교정 시도가 있는가')],
}

SYS_A = ('당신은 LLM 에이전트 실행 로그의 오류 진단 평가자다. 아래 실행 로그를 읽고 각 문항에 대해 '
         '증거가 명확할 때만 true로 판정하라. 반드시 JSON만 출력하라. 형식: '
         '{"items": {"항목ID": {"v": true|false|null, "step": int|null, "why": "20자 내"}}}. '
         'null은 로그 정보 부족으로 판정불가인 경우다.')
SYS_B = ('출력은 JSON 하나만 허용한다: {"items": {...}}. 각 항목 {"v":..., "step":..., "why":...}. '
         '당신의 임무는 에이전트 로그를 정밀 감사하여 문항별 오류 발생 여부를 보수적으로(확실할 때만 true) 판정하는 것이다. '
         '판정불가는 null로 표기한다.')


def render(trace):
    parts = [f"[TASK] {trace['task_spec'][:1500]}"]
    for s in trace['steps']:
        c = s['content'][:STEP_CAP]
        o = (s.get('obs') or '')[:400]
        parts.append(f"[step {s['i']}] ({s.get('agent','')}) {c}" + (f"\n  <obs> {o}" if o else ''))
    body = '\n'.join(parts)
    if len(body) > TRACE_CAP:
        h = int(TRACE_CAP*0.6); t = TRACE_CAP - h
        body = body[:h] + '\n...[중간 생략]...\n' + body[-t:]
    return body


def build_prompt(trace, gkey, variant='a'):
    qs = GROUPS[gkey]
    if variant == 'b': qs = list(reversed(qs))
    qtxt = '\n'.join(f'- {i}: {q}' for i, q in qs)
    return (SYS_B if variant == 'b' else SYS_A), f"### 문항\n{qtxt}\n\n### 실행 로그\n{render(trace)}"


def call_api(model_id, system, user, key, max_retry=1):
    body = json.dumps({'model': model_id, 'max_tokens': OUT_TOK, 'temperature': 0,
                       'system': system, 'messages': [{'role': 'user', 'content': user}]}).encode()
    for attempt in range(max_retry + 1):
        try:
            req = urllib.request.Request('https://api.anthropic.com/v1/messages', data=body,
                headers={'content-type': 'application/json', 'x-api-key': key, 'anthropic-version': '2023-06-01'})
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r)
            txt = ''.join(b.get('text', '') for b in d.get('content', []))
            m = re.search(r'\{.*\}', txt, re.S)
            return json.loads(m.group(0)), d.get('usage', {})
        except Exception as e:
            if attempt >= max_retry: return {'_error': str(e)}, {}
            time.sleep(3)


def estimate(traces, n_models=2, variants=1):
    tot_in = sum(sum(len(build_prompt(t, g)[1]) for g in GROUPS) for t in traces)
    in_tok = tot_in / 4  # ≈4자/토큰 (영문 위주; 보수적으로 ×1.2 병기)
    calls = len(traces) * len(GROUPS)
    print(f'trace {len(traces)}건 × 5콜 = {calls} 콜/모델·변형')
    print(f'입력 ≈ {in_tok/1e6:.2f}M tok (여유율 1.2배 시 {in_tok*1.2/1e6:.2f}M) | 출력 ≈ {calls*OUT_TOK/1e6:.2f}M tok')
    grand = 0
    for m, (pi, po) in PRICE.items():
        c = (in_tok*1.2/1e6)*pi + (calls*OUT_TOK/1e6)*po
        grand += c
        print(f'  {m:7s}: ${c:7.2f}  (배치 API 사용 시 ${c/2:6.2f})')
    extra = grand/2 * 0.2 * (1 if variants else 0)  # 변형 b는 sonnet만·표본 20% 가정
    print(f'2모델 전수 합계: ${grand:.2f} (배치 ${grand/2:.2f}) + 변형b 표본분 ≈ ${extra:.2f}')
    print(f'총 예상: 표준 ${grand+extra*2:.2f} / 배치 ${grand/2+extra:.2f}')


def run(traces, model, out, variant, key):
    done = set()
    if os.path.exists(out):
        done = {(j['trace_id'], j['group']) for j in map(json.loads, open(out))}
    f = open(out, 'a')
    for n, t in enumerate(traces):
        for g in GROUPS:
            if (t['trace_id'], g) in done: continue
            sysp, user = build_prompt(t, g, variant)
            res, usage = call_api(MODELS[model], sysp, user, key)
            f.write(json.dumps({'trace_id': t['trace_id'], 'benchmark': t['benchmark'], 'subset': t['subset'],
                                'model': model, 'variant': variant, 'group': g,
                                'result': res, 'usage': usage}, ensure_ascii=False) + '\n')
            f.flush()
        if (n+1) % 10 == 0: print(f'{n+1}/{len(traces)}', file=sys.stderr)


def aggregate(traces, judg_files):
    from collections import defaultdict, Counter
    gold = {t['trace_id']: t['gold'] for t in traces}
    bm = {t['trace_id']: (t['benchmark'], t['subset']) for t in traces}
    J = defaultdict(dict)   # (model)-> trace -> item -> v/step
    for jf in judg_files:
        for row in map(json.loads, open(jf)):
            items = (row['result'] or {}).get('items', {})
            for iid, v in items.items():
                J[(row['model'], row['variant'])].setdefault(row['trace_id'], {})[iid] = v
    keys = sorted(J)
    # (1) 발생률 (Table 8 절반)
    for k in keys:
        cnt = Counter(); n = Counter()
        for tid, items in J[k].items():
            for iid, v in items.items():
                if isinstance(v, dict) and v.get('v') is not None:
                    n[iid] += 1; cnt[iid] += bool(v['v'])
        print(f'\n[발생률 {k}] ' + '  '.join(f'{i}:{cnt[i]}/{n[i]}' for i in sorted(n)))
    # (2) 모델 간 κ (Table 6)
    if len(keys) >= 2:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from kappa import run as krun
        a, b = keys[0], keys[1]
        pairs = []
        for tid in set(J[a]) & set(J[b]):
            for iid in set(J[a][tid]) & set(J[b][tid]):
                va, vb = J[a][tid][iid], J[b][tid][iid]
                if isinstance(va, dict) and isinstance(vb, dict):
                    pairs.append((str(va.get('v')), str(vb.get('v'))))
        print(f'\n[모델 간 일치 {a} vs {b}] ', end=''); krun(pairs)
    # (3) 골드 앵커 정합 (L3-01 step, ±k)
    for k in keys:
        hit = {0: 0, 1: 0, 3: 0, 5: 0}; n = 0
        for tid, items in J[k].items():
            g = gold.get(tid, {})
            if g.get('step') is None: continue
            v = items.get('L3-01')
            if not isinstance(v, dict) or v.get('step') is None: continue
            n += 1; d = abs(int(v['step']) - g['step'])
            for w in hit:
                if d <= w: hit[w] += 1
        if n:
            print(f'\n[골드 앵커 L3-01 {k}] n={n} exact {hit[0]/n:.1%} | ±1 {hit[1]/n:.1%} | ±3 {hit[3]/n:.1%} | ±5 {hit[5]/n:.1%}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('traces')
    ap.add_argument('--estimate', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--model', choices=list(MODELS), default='sonnet')
    ap.add_argument('--variant', choices=['a', 'b'], default='a')
    ap.add_argument('--out', default='judgments.jsonl')
    ap.add_argument('--limit', type=int, help='앞 N건만 (파일럿·표본용)')
    ap.add_argument('--aggregate', nargs='+', help='판정 jsonl 목록 → 집계')
    a = ap.parse_args()
    traces = [json.loads(l) for l in open(a.traces)]
    if a.limit: traces = traces[:a.limit]
    if a.estimate: estimate(traces)
    elif a.run:
        key = os.environ.get('ANTHROPIC_API_KEY') or sys.exit('ANTHROPIC_API_KEY 환경변수 필요')
        run(traces, a.model, a.out, a.variant, key)
    elif a.aggregate: aggregate(traces, a.aggregate)
    else: ap.print_help()
