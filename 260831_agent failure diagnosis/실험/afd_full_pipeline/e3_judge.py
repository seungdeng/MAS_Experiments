#!/usr/bin/env python3
"""e3_judge.py — E3 LLM 판정 파이프라인 (Table 7·9·11) + 비용 견적.

사용법:
  python3 e3_judge.py traces.jsonl --estimate                    # 비용 견적만 (API 불필요)
  OPENROUTER_API_KEY=sk-or-... python3 e3_judge.py traces.jsonl \
      --run --model gemini-flash --out judg_gemini.jsonl         # 본 실행 (중단 후 재실행 시 이어서)
  python3 e3_judge.py traces.jsonl --run --model claude-haiku --out judg_haiku.jsonl
  python3 e3_judge.py traces.jsonl --run --model gpt5-mini --out judg_gpt5mini.jsonl
  python3 e3_judge.py traces.jsonl --run --model deepseek-flash --out judg_deepseek.jsonl
  python3 e3_judge.py traces.jsonl --run --model gemini-flash --variant b --out judg_gemini_vb.jsonl
  python3 e3_judge.py traces.jsonl --aggregate judg_*.jsonl      # 집계·모델간 κ(전체 쌍)

설계 (paper_final IV장, 2026-08-31 개정 반영):
- 모듈군 4분할 프롬프트: M(L2-M1~M3) / R(L2-R1~R2) / P(L2-P1~P3) / A(L2-A1~A2) + T(L3-01·04)
  = trace당 5콜. JSON 강제, 파싱 실패 시 1회 재시도.
- 판정 모델: OpenRouter 경유 4종 독립 모델(gemini-flash/claude-haiku/gpt5-mini/deepseek-flash).
  <표 12> 판정 모델 민감도(동일 계열 상·하위 P/R 비교) 실험은 취소됨 — 이 스크립트는
  이제 층위별 판정 안정성(Table 7, 모델 간 일치·κ)과 발생률(Table 11) 산출용으로만 쓰인다.
- 프롬프트 변형 b: 문항 순서 역순 + 지시문 재서술 (자기 일관성용).
- 절단 정책: 스텝당 1,200자 + 트레이스 총 100,000자 (초과 시 앞 60%/뒤 40% 보존, 중간 생략 표기).
- 시드성 재현: temperature=0. 골드 라벨은 프롬프트에 절대 미포함.
- 모든 체크리스트 항목은 O/X(참/거짓/판정불가) 이진판정이다. step 필드는 채점 대상이 아닌
  참고용 근거 정보로만 응답에 남기며, 정답 step과 대조하는 골드 앵커 정합(Table 8)은
  이진판정 원칙에 맞지 않아 취소됨(2026-09-03).
"""
import json, os, sys, argparse, time, re, urllib.request


def _load_dotenv():
    """의존성 없이 .env를 읽어 os.environ에 채운다 (이미 설정된 값은 덮어쓰지 않음)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if not os.path.exists(path): return
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

MODELS = {
    'gemini-flash':   'google/gemini-3.7-flash',
    'claude-haiku':   'anthropic/claude-haiku-4.5',
    'gpt5-mini':      'openai/gpt-5-mini',
    'deepseek-flash': 'deepseek/deepseek-v4-flash-0731',
}
PRICE = {   # $/MTok (in, out), OpenRouter passthrough 표준가 2026-08
    'gemini-flash':   (0.75, 3.75),
    'claude-haiku':   (1.00, 5.00),
    'gpt5-mini':      (0.25, 2.00),
    'deepseek-flash': (0.03, 0.16),
}
OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
STEP_CAP, TRACE_CAP, OUT_TOK = 1200, 100_000, 2200   # 700→1500→2200: 추론모델(gpt-5-mini,
                                                      # deepseek-flash)이 reasoning 토큰으로
                                                      # 예산을 소진해 content가 잘리는 문제
                                                      # (2026-09-01/02 관측) 방지 여유분

GROUPS = {
 'M': [('L2-M1','이전 기록에 없는 사실을 회상하는가'),('L2-M2','기록에 있는 정보를 회상하지 못하는가'),('L2-M3','요약이 과제 관련 핵심 세부를 누락하는가')],
 'R': [('L2-R1','진행 상황을 과대/과소평가하는가'),('L2-R2','직전 관찰 결과를 잘못 해석하는가')],
 'P': [('L2-P1','과제 제약과 모순되는 계획을 세우는가'),('L2-P2','전제조건상 불가능한 행동을 계획하는가'),('L2-P3','동일 전략 고수·탐색 낭비가 지속되는가')],
 'A': [('L2-A1','행동이 직전 계획과 불일치하는가'),('L2-A2','행동 인자·대상 지정이 잘못되는가')],
 'T': [('L3-01','최초의 결정적 오류 단계가 식별되는가'),('L3-04','자기 교정 시도가 있는가')],
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
    # reasoning.effort는 provider별로 해석이 달라 deepseek 등에서 무시/과다소모될 수 있어
    # 토큰 상한(max_tokens) 형태로 명시 — reasoning에 최대 400, 나머지는 실제 JSON 답변용으로 남긴다.
    body = json.dumps({'model': model_id, 'max_tokens': OUT_TOK, 'temperature': 0,
                       'reasoning': {'max_tokens': 400},
                       'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]}).encode()
    for attempt in range(max_retry + 1):
        try:
            req = urllib.request.Request(OPENROUTER_URL, data=body,
                headers={'content-type': 'application/json', 'authorization': f'Bearer {key}'})
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r)
            txt = d['choices'][0]['message']['content']
            if not txt:
                raise ValueError(f'empty content (finish_reason={d["choices"][0].get("finish_reason")})')
            m = re.search(r'\{.*\}', txt, re.S)
            return json.loads(m.group(0)), d.get('usage', {})
        except Exception as e:
            if attempt >= max_retry: return {'_error': str(e)}, {}
            time.sleep(3)


def estimate(traces, variants=1):
    tot_in = sum(sum(len(build_prompt(t, g)[1]) for g in GROUPS) for t in traces)
    in_tok = tot_in / 4  # ≈4자/토큰 (영문 위주; 보수적으로 ×1.2 병기)
    calls = len(traces) * len(GROUPS)
    print(f'trace {len(traces)}건 × 5콜 = {calls} 콜/모델·변형')
    print(f'입력 ≈ {in_tok/1e6:.2f}M tok (여유율 1.2배 시 {in_tok*1.2/1e6:.2f}M) | 출력 ≈ {calls*OUT_TOK/1e6:.2f}M tok')
    grand = 0
    for m, (pi, po) in PRICE.items():
        c = (in_tok*1.2/1e6)*pi + (calls*OUT_TOK/1e6)*po
        grand += c
        print(f'  {m:14s} ({MODELS[m]:32s}): ${c:8.2f}')
    extra = grand * 0.2 * (1 if variants else 0)  # 변형 b는 표본 20% 가정, 4모델 전체 적용
    print(f'4모델 전수 합계: ${grand:.2f} + 변형b 표본분(20%, 4모델) ≈ ${extra:.2f}')
    print(f'총 예상: ${grand+extra:.2f}')


def run(traces, model, out, variant, key):
    done = set()
    if os.path.exists(out):
        # 실패(_error) 기록은 완료로 치지 않음 — 재실행 시 자동 재시도되도록.
        done = {(j['trace_id'], j['group']) for j in map(json.loads, open(out, encoding='utf-8'))
                if '_error' not in (j.get('result') or {})}
    f = open(out, 'a', encoding='utf-8')
    for n, t in enumerate(traces):
        for g in GROUPS:
            if (t['trace_id'], g) in done: continue
            sysp, user = build_prompt(t, g, variant)
            res, usage = call_api(MODELS[model], sysp, user, key)
            f.write(json.dumps({'trace_id': t['trace_id'], 'benchmark': t['benchmark'], 'subset': t['subset'],
                                'model': model, 'variant': variant, 'group': g,
                                'result': res, 'usage': usage}, ensure_ascii=False) + '\n')
            f.flush()
            if '_error' in res:
                print(f'  ! {t["trace_id"]}/{g}: {res["_error"]}', file=sys.stderr, flush=True)
        print(f'{n+1}/{len(traces)} ({t["trace_id"]})', file=sys.stderr, flush=True)


VALID_ITEMS = {iid for qs in GROUPS.values() for iid, _ in qs}


def aggregate(judg_files):
    from collections import defaultdict, Counter
    J = defaultdict(dict)   # (model)-> trace -> item -> v/step
    skipped = 0
    bad_ids = Counter()
    for jf in judg_files:
        for row in map(json.loads, open(jf, encoding='utf-8')):
            items = (row['result'] or {}).get('items', {})
            if isinstance(items, list):
                # 드물게 모델이 {items:{id:{...}}} 대신 [{id/itemId, v/value, step, why}] 형태로 응답 —
                # 딕셔너리로 정규화해 살린다 (형식만 다를 뿐 판정 자체는 유효한 데이터).
                norm = {}
                for e in items:
                    if not isinstance(e, dict): continue
                    iid = e.get('id') or e.get('itemId')
                    if not iid: continue
                    norm[iid] = {'v': e.get('v', e.get('value')), 'step': e.get('step'), 'why': e.get('why')}
                items = norm
            if not isinstance(items, dict):
                skipped += 1; continue
            for iid, v in items.items():
                if iid not in VALID_ITEMS:   # 모델이 존재하지 않는/삭제된 항목ID를 환각한 경우 제외
                    bad_ids[iid] += 1; continue
                J[(row['model'], row['variant'])].setdefault(row['trace_id'], {})[iid] = v
    if skipped:
        print(f'[경고] items 형식 불량으로 건너뜀: {skipped}건', file=sys.stderr)
    if bad_ids:
        print(f'[경고] 존재하지 않는 항목ID 환각으로 제외: {dict(bad_ids)}', file=sys.stderr)
    keys = sorted(J)
    # (1) 발생률 (Table 11)
    for k in keys:
        cnt = Counter(); n = Counter()
        for tid, items in J[k].items():
            for iid, v in items.items():
                if isinstance(v, dict) and v.get('v') is not None:
                    n[iid] += 1; cnt[iid] += bool(v['v'])
        print(f'\n[발생률 {k}] ' + '  '.join(f'{i}:{cnt[i]}/{n[i]}' for i in sorted(n)))
    # (2) 모델 간 κ — 전체 쌍 (Table 7; 4모델이면 6쌍)
    if len(keys) >= 2:
        import itertools
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from kappa import run as krun
        for a, b in itertools.combinations(keys, 2):
            pairs = []
            for tid in set(J[a]) & set(J[b]):
                for iid in set(J[a][tid]) & set(J[b][tid]):
                    va, vb = J[a][tid][iid], J[b][tid][iid]
                    if isinstance(va, dict) and isinstance(vb, dict):
                        pairs.append((str(va.get('v')), str(vb.get('v'))))
            if pairs:
                print(f'\n[모델 간 일치 {a} vs {b}] ', end=''); krun(pairs)
    # Table 8(골드 앵커 정합)은 취소됨 — 전 체크리스트 이진판정 원칙상 LLM에게 step 번호를
    # 채점 대상으로 요구하지 않기로 함(2026-09-03). step 필드는 여전히 참고용 보조정보로만 응답에 남음.


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('traces')
    ap.add_argument('--estimate', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--model', choices=list(MODELS), default='gemini-flash')
    ap.add_argument('--variant', choices=['a', 'b'], default='a')
    ap.add_argument('--out', default='judgments.jsonl')
    ap.add_argument('--limit', type=int, help='앞 N건만 (파일럿·표본용)')
    ap.add_argument('--aggregate', nargs='+', help='판정 jsonl 목록 → 집계')
    a = ap.parse_args()
    traces = [json.loads(l) for l in open(a.traces, encoding='utf-8')]
    if a.limit: traces = traces[:a.limit]
    if a.estimate: estimate(traces)
    elif a.run:
        key = os.environ.get('OPENROUTER_API_KEY') or sys.exit('OPENROUTER_API_KEY 환경변수 필요')
        run(traces, a.model, a.out, a.variant, key)
    elif a.aggregate: aggregate(a.aggregate)
    else: ap.print_help()
