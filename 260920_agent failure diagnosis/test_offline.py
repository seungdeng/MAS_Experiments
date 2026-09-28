#!/usr/bin/env python3
"""test_offline.py — API 없이 파이프라인 배선 검증 (가짜 LLM). `python test_offline.py`

검증 항목
 1) L0 에서 L0-03 이 false → L0-03 에 종속된 L1/L2 항목은 '판단 불가', L2 호출은 생략, L1 호출엔 L1-01 만 포함
 2) 모두 충족 → 3콜, L2 오류 step 이 결과에 반영
 3) 귀인 프롬프트: 체크리스트 블록에 판단 불가 항목이 없음, view=errors 는 오류 항목만
 4) LLM 클래스: 캐시 마커/지원 파라미터 처리, 응답 파싱·usage.cost 합산, finish_reason=length 처리
 5) evaluate: 정확일치·±k·McNemar
"""
import json
import os
import tempfile

os.environ['OPENROUTER_API_KEY'] = 'test-key'
import attribute
import ckl_judge
import common
import evaluate
from checklist import ERROR, INDET, NO_ERROR

TRACE = {'trace_id': 'T1', 'benchmark': 'AEB', 'subset': 'GAIA', 'task_spec': 'find X', 'n_steps': 3, 'meta': {'model': 'm'},
         'gold': {'step': 2, 'agent': None, 'module': 'plan', 'type': None},
         'steps': [{'i': i, 'agent': 'm', 'content': f'<plan>p{i}</plan><action>a{i}</action>', 'obs': f'o{i}'} for i in range(3)]}


class FakeLLM:
    model = 'fake/model'

    def __init__(self, l0_false=()):
        self.l0_false, self.calls = set(l0_false), []

    def ask(self, log, instruction):
        self.calls.append(instruction)
        if '[Instruction] L0' in instruction:
            items = {i: {'v': i not in self.l0_false, 'why': 'w'} for i in ('L0-01', 'L0-02', 'L0-03', 'L0-04', 'L0-05', 'L0-06')}
        elif '[Instruction] L1' in instruction or '[Instruction] L2' in instruction:
            ids = [l.split(':')[0][2:] for l in instruction.splitlines() if l.startswith('- L')]
            items = {i: {'v': i == 'L2-P2', 'step': 1 if i == 'L2-P2' else None, 'why': 'bad plan'} for i in ids}
        else:
            return {'step': 2, 'module': 'plan', 'reason': 'r'}, {'cost': 0.01}
        return {'items': items}, {'cost': 0.01, 'prompt_tokens': 10}


def test_dependency():
    llm = FakeLLM(l0_false={'L0-03'})
    row = ckl_judge.judge_trace(llm, TRACE)
    assert len(llm.calls) == 2, len(llm.calls)                       # L0 + L1 (L2 skipped: every L2 item needs L0-03)
    l1_ids = [l.split(':')[0][2:] for l in llm.calls[1].splitlines() if l.startswith('- L1')]
    assert l1_ids == ['L1-01'], l1_ids                               # L1-02/03 need L0-03 -> not asked
    assert row['asked'] == {'L1': ['L1-01'], 'L2': []}
    assert row['l1']['L1-01']['status'] == NO_ERROR
    assert row['l1']['L1-02']['status'] == INDET and row['l1']['L1-02']['reason'] == 'dep:L0-03'
    assert all(x['status'] == INDET for x in row['l2'].values())    # never NO_ERROR without evidence
    print('ok 1: L0-03 missing -> dependent items INDETERMINATE, L2 call skipped')


def test_full():
    llm = FakeLLM()
    row = ckl_judge.judge_trace(llm, TRACE)
    assert len(llm.calls) == 3
    assert row['l2']['L2-P2']['status'] == ERROR and row['l2']['L2-P2']['step'] == 1
    assert row['l2']['L2-M1']['status'] == NO_ERROR
    assert 'errors' not in row and row['usage']['cost'] == 0.03
    print('ok 2: all prerequisites met -> 3 calls, error step recorded')
    return row


def test_attribute_prompt(full_row):
    part = ckl_judge.judge_trace(FakeLLM(l0_false={'L0-03'}), TRACE)          # L0-03 missing -> most items not assessable
    bi = attribute.build_instruction
    txt = bi(TRACE, 'ckl', part)                                             # default: layers L0,L1,L2 / view applicable
    assert 'L0-03: ' in txt and '-> NO.' in txt and 'L0-01: ' in txt and '-> YES.' in txt      # L0 is included
    assert 'L1-01: ' in txt and '-> no error found' in txt
    assert 'L1-02' not in txt and 'L2-M1' not in txt                          # INDETERMINATE items omitted by default
    txt_i = bi(TRACE, 'ckl', part, {'indet': True})
    assert 'L1-02' in txt_i and 'not assessable (missing L0-03)' in txt_i    # --ckl-indet
    assert 'Log properties' not in bi(TRACE, 'ckl', part, {'layers': ('L1', 'L2')})          # --ckl-layers
    assert 'Error findings' not in bi(TRACE, 'ckl', part, {'layers': ('L0',)})
    txt_err = bi(TRACE, 'ckl', full_row, {'view': 'errors', 'layers': ('L1', 'L2')})           # --ckl-view errors
    assert 'L2-P2: ' in txt_err and 'ERROR at step 1' in txt_err and 'no error found' not in txt_err
    base = bi(TRACE, 'aao')
    assert 'Checklist findings' not in base and bi(TRACE, 'aao', full_row) == base            # baseline ignores the checklist
    assert 'earliest error that led to the failure' in base                                   # agreed definition
    assert '"module"' in base and '"agent"' in bi(dict(TRACE, benchmark='WhoWhen'), 'aao')
    try:
        bi(TRACE, 'ckl')
        raise AssertionError('ckl without a checklist row must fail')
    except ValueError:
        pass
    row = attribute.attribute_trace(FakeLLM(), TRACE, 'ckl', full_row, {}, label='ckl_x')
    assert row['pred']['step'] == 2 and row['pred']['module'] == 'plan' and row['method'] == 'ckl_x'
    assert row['ckl_model'] == 'fake/model' and row['ckl_opts']['layers'] == ['L0', 'L1', 'L2']
    print('ok 3: attribution prompts (L0 included; layers / view / indet options; baseline; label) and parsing')


def test_llm_class():
    table = {}
    common.model_table = lambda refresh=False: table      # no network: control the OpenRouter model list
    l = common.LLM('anthropic/claude-sonnet-5')            # list unknown -> nothing optional is sent
    body = l._body('LOG', 'INSTR')
    assert 'temperature' not in body and 'response_format' not in body
    assert body['messages'][1]['content'][0]['cache_control'] == {'type': 'ephemeral'} and l.use_cache
    assert body['messages'][1]['content'][1]['text'] == 'INSTR' and body['messages'][0]['role'] == 'system'
    assert not common.LLM('openai/x').use_cache
    table['a/b'] = {'id': 'a/b', 'supported_parameters': ['temperature', 'response_format', 'reasoning']}
    l2 = common.LLM('a/b', reasoning='low')
    assert l2.params == {'temperature': 0.0, 'response_format': {'type': 'json_object'}, 'reasoning': {'effort': 'low'}}
    table['c/d'] = {'id': 'c/d', 'supported_parameters': ['max_tokens']}
    l3 = common.LLM('c/d', reasoning='low')
    assert l3.params == {} and set(l3.skipped) == {'temperature', 'reasoning'}
    l3._post = lambda body: {'choices': [{'message': {'content': 'ok {"a": 1} done'}, 'finish_reason': 'stop'}],
                             'usage': {'prompt_tokens': 100, 'completion_tokens': 5, 'cost': 0.002,
                                       'prompt_tokens_details': {'cached_tokens': 80, 'cache_write_tokens': 0}}}
    res, u = l3.ask('L', 'I')
    assert res == {'a': 1} and u['cost'] == 0.002 and u['cached_tokens'] == 80
    l3._post = lambda body: {'choices': [{'message': {'content': ''}, 'finish_reason': 'length'}], 'usage': {}}
    assert '_error' in l3.ask('L', 'I')[0]
    l3._post = lambda body: {'choices': [{'message': {'content': 'no json here'}, 'finish_reason': 'stop'}], 'usage': {}}
    assert l3.ask('L', 'I')[0]['_error'] == 'json_parse_failed'
    table['e/f'] = {'id': 'e/f', 'context_length': 400000, 'supported_parameters': [],
                    'top_provider': {'context_length': 400000, 'max_completion_tokens': 128000}}
    m = common.LLM('e/f')
    assert m._body('x' * 1000, 'i')['max_tokens'] == 128000                      # default = model max output
    assert common.LLM('e/f', max_tokens=500)._body('x', 'i')['max_tokens'] == 500   # explicit override
    big = m._body('x' * 300_000, 'i')['max_tokens']                              # long prompt: reduced to fit the context
    assert 1024 <= big < 128000 and big + 300_000 <= 400000, big
    sent = []
    def success(body):
        sent.append(body)
        return {'choices': [{'message': {'content': '{"ok": true}'}, 'finish_reason': 'stop'}]}
    m._post = success
    result, usage = m.ask('x' * 400_000, 'i')
    assert result == {'ok': True} and usage['input_truncated_calls'] == 1
    assert usage['input_omitted_chars'] > 0 and 'omitted to fit context' in sent[-1]['messages'][1]['content'][0]['text']
    fixed = common.LLM('e/f', max_tokens=128000)
    fixed._post = success
    assert fixed.ask('x' * 300_000, 'i')[1]['input_truncated_calls'] == 1
    assert sent[-1]['max_tokens'] == 128000
    responses = iter([{'_error': 'input_too_long: provider limit'}, success({})])
    m._post = lambda body: next(responses)
    result, usage = m.ask('x' * 10000, 'i')
    assert result == {'ok': True} and usage['context_retries'] == 1 and usage['input_truncated_calls'] == 1
    m._post = success
    assert m.ask('complete log', 'i')[1]['input_truncated_calls'] == 0
    unicode_log = '\uac00' * 10000
    assert len(common.shorten_log(unicode_log, 1000).encode('utf-8')) <= 1000
    assert common.LLM('c/d')._body('x', 'i')['max_tokens'] == 32768             # model without limit metadata
    print('ok 4: request body / supported-parameter gating / max_tokens=max / usage.cost / error paths')


def test_render_and_evaluate():
    log = common.render_log(TRACE)
    assert log.startswith('[TASK] find X\n[META] model=m\n[step 0] (m)') and '<obs> o2' in log
    big = dict(TRACE, steps=[{'i': i, 'agent': 'm', 'content': 'x' * 900, 'obs': ''} for i in range(400)])
    big['task_spec'] = 'TASK' * 2000
    big['steps'][200]['content'] = 'MIDDLE' * 2000
    big['steps'][200]['obs'] = 'OBS' * 2000
    r = common.render_log(big)
    assert len(r) > 300_000 and 'omitted' not in r
    assert big['task_spec'] in r and big['steps'][200]['content'] in r and big['steps'][200]['obs'] in r
    assert all(f'[step {i}]' in r for i in range(400))

    gold = [dict(TRACE, trace_id=f'T{i}', gold={'step': 5, 'agent': None, 'module': 'plan', 'type': None}) for i in range(10)]
    tmp = tempfile.mkdtemp()
    def write(name, method, steps):
        p = os.path.join(tmp, name)
        with open(p, 'w', encoding='utf-8') as f:
            for i, s in enumerate(steps):
                f.write(json.dumps({'trace_id': f'T{i}', 'benchmark': 'AEB', 'subset': 'GAIA', 'method': method, 'model': 'm',
                                    'pred': {'step': s, 'module': 'plan'}}) + '\n')
        return p
    base = write('a.jsonl', 'aao', [5, 4, 9, 9, 9, 9, 9, 9, 9, 9])       # exact 1, ±1 2
    new = write('b.jsonl', 'ckl', [5, 5, 5, 5, 9, 9, 9, 9, 9, 9])        # exact 4
    res, groups, g = evaluate.eval_attr(gold, [base, new])
    m = res[('aao', 'm')][('AEB', 'GAIA')]
    assert m['n'] == 10 and m['hit'][0] == 1 and m['hit'][1] == 2 and m['module'] == [10, 10]
    assert res[('ckl', 'm')][('AEB', 'ALL')]['hit'][0] == 4
    assert abs(evaluate.mcnemar_exact(3, 0) - 0.25) < 1e-12 and evaluate.mcnemar_exact(0, 0) == 1.0
    print('ok 5: rendering (full task, steps, observations preserved) and evaluation metrics')


def test_paths():
    root = tempfile.mkdtemp()
    common.RESULTS_DIR, common.DATA_DIR = os.path.join(root, 'results'), os.path.join(root, 'data')
    out = common.result_path('ckl', 'ckl_x.jsonl')                                    # bare name -> results/ckl/
    assert out == os.path.join(root, 'results', 'ckl', 'ckl_x.jsonl') and os.path.isdir(os.path.dirname(out))
    assert common.result_path('ckl', os.path.join(root, 'elsewhere', 'a.jsonl')).endswith('a.jsonl')   # explicit dir kept
    open(out, 'w').close()
    assert common.find_result('ckl', 'ckl_x.jsonl') == out                            # read by bare name
    assert common.expand_results('ckl', ['ckl_*.jsonl']) == [out]
    os.makedirs(common.DATA_DIR)
    open(os.path.join(common.DATA_DIR, 'traces.jsonl'), 'w').close()
    assert common.find_traces('traces.jsonl') == os.path.join(common.DATA_DIR, 'traces.jsonl')  # old-style argument still works
    print('ok 6: data/ and results/ path helpers')


def test_agentrx():
    import normalize
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, 'tau_retail'))
        os.makedirs(os.path.join(tmp, 'ground_truth'))
        messages = [
            {'index': 1, 'role': 'system', 'content': 'Retail policy'},
            {'index': 2, 'role': 'user', 'content': 'Return my order'},
            {'index': 3, 'role': 'assistant', 'content': None,
             'tool_calls': [{'id': 'call1', 'function': {'name': 'return_order', 'arguments': '{}'}}]},
            {'index': 4, 'role': 'tool', 'content': 'done', 'tool_call_id': 'call1'},
        ]
        row = {'task_id': 2, 'traj': messages, 'reward': 0,
               'info': {'task': {'actions': ['SECRET_REFERENCE'], 'outputs': ['SECRET_OUTPUT']}}}
        label = {'trajectory_id': 2, 'failures': [
            {'failure_id': 1, 'step_number': 1, 'failure_category': 'Earlier error'},
            {'failure_id': 2, 'step_number': 3, 'failure_category': 'SECRET_CATEGORY',
             'step_reason': 'SECRET_EXPLANATION', 'failed_agent': 'Assistant'}],
            'root_cause': {'failure_id': 2}}
        def write(name, data):
            with open(os.path.join(tmp, name), 'w', encoding='utf-8') as fh:
                json.dump(data, fh)
        write('tau_retail/tau_dataset_failed.json', [row])
        write('ground_truth/tau_ground_truth.json', [label])
        trace, = normalize.norm_agentrx(tmp)
        assert trace['gold']['step'] == 2 and trace['gold']['source_step'] == 3
        assert len(trace['steps']) == 4 and trace['steps'][3]['agent'] == 'tool'
        prompt = common.render_log(trace)
        assert 'SECRET_' not in prompt and 'return_order' in prompt and 'call1' in prompt
        instruction = attribute.build_instruction(trace, 'aao')
        assert 'first unrecoverable' in instruction and 'OpenTelemetry' not in instruction
        label['failures'][1]['step_number'] = 99
        write('ground_truth/tau_ground_truth.json', [label])
        try:
            normalize.norm_agentrx(tmp)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid root cause step accepted')
    print('ok 7: AgentRx root cause mapping, tool messages, label isolation, invalid gold rejection')


if __name__ == '__main__':
    test_dependency()
    row = test_full()
    test_attribute_prompt(row)
    test_llm_class()
    test_render_and_evaluate()
    test_paths()
    test_agentrx()
    print('ALL OFFLINE TESTS PASSED')
