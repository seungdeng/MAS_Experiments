#!/usr/bin/env python3
"""checklist.py — 3계층 18항목 체크리스트 정의, 종속(상속) 관계, 판정 상태 로직.

계층
- L0 (6항목)  로그 자체의 기록 여부(관측가능성). 판정값: v=true(기록됨) / false(없음) / null(판정불가)
- L1 (3항목)  시스템·실행 오류. 요구 L0 항목에 종속
- L2 (9항목)  인지 모듈(M/R/P/A) 오류. 요구 L0 항목에 종속

종속 규칙 (핵심)
  L1/L2 항목의 '요구 L0'가 하나라도 true가 아니면(=상속받지 못하면) 그 항목은
  LLM 답변과 무관하게 '판단 불가(INDETERMINATE)'이다. '오류 없음(NO_ERROR)'으로 취급하지 않는다.
  → 근거가 로그에 없는 상태에서 "오류 없음"이 되는 것을 구조적으로 차단한다.
  → 요구 L0가 모두 충족된 항목만 LLM에 묻는다 (토큰 절감 + 환각 판정 방지).

최종 상태 (L1/L2)
  ERROR          오류 발생 (step 포함)
  NO_ERROR       오류 없음 (요구 L0 충족 + LLM이 false)
  INDETERMINATE  판단 불가 — reason: 'dep:L0-03,L0-04' (요구 L0 미충족) | 'llm' (LLM이 null/응답 실패)
"""
import argparse

ERROR, NO_ERROR, INDET = 'ERROR', 'NO_ERROR', 'INDETERMINATE'

L0 = {
    'L0-01': '과제 명세(목표·제약)가 로그에 포함되는가',
    'L0-02': '각 단계의 경계가 명확히 식별되는가',
    'L0-03': '단계마다 모듈별 출력이 나뉘어 기록되는가',
    'L0-04': '행동의 결과가 기록되는가',
    'L0-05': '도구/API 응답이 가공 이전 상태로 보존되는가',
    'L0-06': '모델·토큰·지연·한도 등 시스템 정보가 기록되는가',
}

# id -> (판정 질문, 요구 L0 목록)
L1 = {
    'L1-01': ('목표를 달성하지 못한 채 step 한도에 도달하여 종료되는가 (C-S1)', ['L0-02', 'L0-06']),
    'L1-02': ('행동이 형식 오류로 실행되지 못하는가 (C-S2)',                   ['L0-03', 'L0-05']),
    'L1-03': ('행동이 환경·도구의 문제로 실패하는가 (C-S3)',                   ['L0-03', 'L0-05']),
}

L2 = {
    'L2-M1': ('이전 기록과 어긋나는 내용을 사실로 진술하는가 (C-M1)',           ['L0-03', 'L0-04']),
    'L2-M2': ('이전 기록에 있는 정보를 활용하지 못하는가 (C-M2)',               ['L0-03', 'L0-04']),
    'L2-R1': ('누적된 진행 정도를 실제와 다르게 판단하는가 (C-R1)',             ['L0-03', 'L0-04']),
    'L2-R2': ('직전 행동의 결과를 잘못 해석하는가 (C-R2)',                     ['L0-03', 'L0-04']),
    'L2-P1': ('과제 명세를 위반하는 계획을 수립하는가 (C-P1)',                 ['L0-01', 'L0-03']),
    'L2-P2': ('현재 상태에서 실행 불가능한 행동을 계획하는가 (C-P2)',           ['L0-03', 'L0-04']),
    'L2-P3': ('실패를 확인했음에도 다른 전략으로 전환하지 않는가 (C-P3)',       ['L0-02', 'L0-03', 'L0-04']),
    'L2-A1': ('계획한 것과 다른 행동을 실행하는가 (C-A1)',                     ['L0-03']),
    'L2-A2': ('행동의 대상이나 값이 잘못 지정되는가 (C-A2)',                   ['L0-03', 'L0-04']),
}

UPPER = {'L1': L1, 'L2': L2}
LAYER_IDS = ('L0', 'L1', 'L2')

# English wording used in the LLM prompts (the benchmarks and logs are English). The Korean text above is the
# canonical item definition for the paper; keep the two in sync.
EN = {
    'L0-01': 'Does the log contain the task specification (goal and constraints)?',
    'L0-02': 'Are the boundaries of each step clearly identifiable?',
    'L0-03': 'Are the outputs of each module (e.g. memory, reflection, plan, action) recorded separately at each step?',
    'L0-04': 'Is the result of each action recorded?',
    'L0-05': 'Are tool/API responses preserved in their raw form, before any processing?',
    'L0-06': 'Is system information such as the model, tokens, latency, or limits (e.g. step limit) recorded?',
    'L1-01': 'Does the run end by reaching the step limit without achieving the goal? (C-S1)',
    'L1-02': 'Does an action fail to execute because of a format error? (C-S2)',
    'L1-03': 'Does an action fail because of a problem in the environment or a tool? (C-S3)',
    'L2-M1': 'Does the agent state something as fact that contradicts earlier records? (C-M1)',
    'L2-M2': 'Does the agent fail to use information that is available in earlier records? (C-M2)',
    'L2-R1': 'Does the agent misjudge its cumulative progress compared with the actual state? (C-R1)',
    'L2-R2': 'Does the agent misinterpret the result of its previous action? (C-R2)',
    'L2-P1': 'Does the agent make a plan that violates the task specification? (C-P1)',
    'L2-P2': 'Does the agent plan an action that is infeasible in the current state? (C-P2)',
    'L2-P3': 'Does the agent fail to switch to a different strategy after recognizing a failure? (C-P3)',
    'L2-A1': 'Does the agent execute an action different from what it planned? (C-A1)',
    'L2-A2': 'Is the target or value of an action specified incorrectly? (C-A2)',
}


def deps_of(item_id):
    for layer in UPPER.values():
        if item_id in layer:
            return layer[item_id][1]
    raise KeyError(item_id)


def l0_present(l0_results):
    """L0 판정 결과 {id: {'v': bool|None, ...}} → 기록이 확인된(v is True) L0 ID 집합.
    false / null / 누락은 모두 '상속 불가'로 취급한다."""
    return {i for i, r in (l0_results or {}).items() if isinstance(r, dict) and r.get('v') is True}


def unmet_deps(item_id, present):
    return [d for d in deps_of(item_id) if d not in present]


def askable(layer, present):
    """해당 계층에서 LLM에 물어볼 항목 ID(요구 L0 전부 충족). 비면 그 계층 호출을 생략한다."""
    return [i for i in UPPER[layer] if not unmet_deps(i, present)]


def resolve(layer, present, llm_items):
    """L1/L2 최종 상태 산출.
    llm_items: LLM 응답의 {id: {'v','step','why'}} (호출을 생략했으면 {} 또는 None)."""
    out = {}
    llm_items = llm_items or {}
    for iid, (question, _) in UPPER[layer].items():
        unmet = unmet_deps(iid, present)
        if unmet:  # 종속 미충족 → LLM 응답이 있어도 무시 (판단 불가가 우선)
            out[iid] = {'status': INDET, 'reason': 'dep:' + ','.join(unmet), 'step': None, 'why': ''}
            continue
        r = llm_items.get(iid)
        v = r.get('v') if isinstance(r, dict) else None
        if v is True:
            out[iid] = {'status': ERROR, 'reason': None, 'step': _as_int(r.get('step')), 'why': r.get('why', '')}
        elif v is False:
            out[iid] = {'status': NO_ERROR, 'reason': None, 'step': None, 'why': r.get('why', '')}
        else:
            out[iid] = {'status': INDET, 'reason': 'llm', 'step': None, 'why': (r or {}).get('why', '') if isinstance(r, dict) else ''}
    return out


def _as_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def selftest():
    # 1) 종속 관계가 표(L0의 '종속' 열: 1,2,11,7,2,1)와 일치
    cnt = {i: 0 for i in L0}
    for layer in UPPER.values():
        for _, deps in layer.values():
            for d in deps:
                assert d in L0, d
                cnt[d] += 1
    assert list(cnt.values()) == [1, 2, 11, 7, 2, 1], cnt
    assert (len(L0), len(L1), len(L2)) == (6, 3, 9)
    assert set(EN) == set(L0) | set(L1) | set(L2)          # every item has an English prompt wording

    all_ok = {i: {'v': True} for i in L0}

    # 2) 요구 L0 전부 충족 + LLM false → 오류 없음
    r = resolve('L2', l0_present(all_ok), {i: {'v': False} for i in L2})
    assert all(x['status'] == NO_ERROR for x in r.values())

    # 3) 핵심: L0-03 없음 → L0-03에 종속된 항목은 LLM이 false라 해도 '판단 불가'
    no03 = dict(all_ok, **{'L0-03': {'v': False}})
    r = resolve('L2', l0_present(no03), {i: {'v': False} for i in L2})
    assert all(x['status'] == INDET and x['reason'].startswith('dep:L0-03') for x in r.values()), r
    r = resolve('L1', l0_present(no03), {i: {'v': False} for i in L1})
    assert r['L1-01']['status'] == NO_ERROR          # L1-01은 L0-03에 종속되지 않음
    assert r['L1-02']['status'] == INDET and r['L1-03']['status'] == INDET

    # 4) 종속 미충족 항목은 LLM이 true라 해도 오류로 세지 않는다
    r = resolve('L2', l0_present(no03), {'L2-M1': {'v': True, 'step': 3}})
    assert r['L2-M1']['status'] == INDET

    # 5) L0 null / 누락도 상속 불가로 취급
    partial = {'L0-01': {'v': True}, 'L0-03': {'v': None}}
    assert l0_present(partial) == {'L0-01'}
    assert askable('L2', l0_present(partial)) == []  # P1은 L0-03 필요 → 전부 불가
    assert askable('L2', l0_present(all_ok)) == list(L2)

    # 6) 요구 충족 + LLM null/누락 → 판단 불가(reason=llm), 오류 true → step 파싱
    r = resolve('L2', l0_present(all_ok), {'L2-M1': {'v': None}, 'L2-M2': {'v': True, 'step': '4', 'why': 'x'}})
    assert r['L2-M1']['status'] == INDET and r['L2-M1']['reason'] == 'llm'
    assert r['L2-R1']['status'] == INDET and r['L2-R1']['reason'] == 'llm'  # LLM 응답 누락
    assert r['L2-M2']['status'] == ERROR and r['L2-M2']['step'] == 4

    # 7) 빈 L0(호출 실패) → L1/L2 전부 판단 불가, 호출 대상 없음
    assert askable('L1', l0_present({})) == [] and askable('L2', l0_present(None)) == []
    print('SELFTEST OK  (18항목, 종속 열 [1,2,11,7,2,1] 일치, 종속 미충족 → 판단 불가 검증)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--show', action='store_true', help='항목표와 종속 관계 출력')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.show:
        for k, q in L0.items():
            print(k, q)
        for layer in UPPER.values():
            for k, (q, d) in layer.items():
                print(k, q, '<-', ', '.join(d))
    else:
        ap.print_help()
