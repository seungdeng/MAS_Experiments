"""common.py — log rendering, OpenRouter client, .env loading, jsonl I/O, trace selection.
Shared by ckl_judge.py / attribute.py / evaluate.py.  Standard library only.

OpenRouter notes (verified against openrouter.ai docs and GET /api/v1/models, 2026-09)
- Endpoint: POST {OPENROUTER_BASE_URL}/chat/completions, header `Authorization: Bearer <key>`.
- The response `usage` always carries `cost` (USD actually billed), `prompt_tokens`, `completion_tokens`,
  `prompt_tokens_details.cached_tokens` / `cache_write_tokens`.  We sum `usage.cost` directly.
- GET /api/v1/models (public) lists per-model `pricing` (USD per token, as strings) and `supported_parameters`.
  We use it (a) for cost estimates and (b) to send optional parameters (temperature, response_format,
  reasoning) ONLY to models that support them — e.g. Claude Sonnet 5 does not accept `temperature`.
- Prompt caching is a prefix match.  Every call for one trace shares the same system prompt and the same
  log block placed first (with `cache_control` for providers that need explicit breakpoints); only the
  instruction that follows changes.  Providers with automatic caching (OpenAI, DeepSeek, Gemini 2.5, ...)
  benefit from the identical prefix without any marker.
"""
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------- .env
def load_env(path=None):
    """Minimal .env loader (KEY=VALUE, '#' comments, optional quotes). Never overrides real env vars."""
    here = os.path.dirname(os.path.abspath(__file__))
    for p in ([path] if path else [os.path.join(here, '.env'), os.path.join(os.getcwd(), '.env')]):
        if p and os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    k, v = line.split('=', 1)
                    v = v.strip()
                    if len(v) >= 2 and v[0] == v[-1] and v[0] in '"\'':
                        v = v[1:-1]
                    os.environ.setdefault(k.strip(), v)


load_env()
BASE_URL = os.environ.get('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1').rstrip('/')

# Folder layout: data/ (traces.jsonl), results/ckl, results/attribution, results/recount_260831.
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
RESULTS_DIR = os.path.join(ROOT, 'results')
DEFAULT_TRACES = os.path.join(DATA_DIR, 'traces.jsonl')


def result_path(kind, name):
    """Where to WRITE an output. A bare file name goes to results/<kind>/ (created); a name with a directory is used as given."""
    if os.path.dirname(name):
        return name
    d = os.path.join(RESULTS_DIR, kind)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def find_result(kind, name):
    """Where to READ a result file: as given if it exists, else results/<kind>/<name>."""
    if os.path.exists(name):
        return name
    alt = os.path.join(RESULTS_DIR, kind, name)
    return alt if os.path.exists(alt) else name


def expand_results(kind, patterns):
    """Glob patterns (as given, else inside results/<kind>/) -> sorted unique file list."""
    import glob
    files = []
    for p in patterns:
        files += glob.glob(p) or glob.glob(os.path.join(RESULTS_DIR, kind, p))
    return sorted(set(files))


def find_traces(path=None):
    path = path or DEFAULT_TRACES
    if os.path.exists(path):
        return path
    alt = os.path.join(DATA_DIR, path)
    return alt if os.path.exists(alt) else path


SYSTEM = ('You are an evaluator that analyzes execution logs of LLM agents. Read the [Execution Log] in the user '
          'message and follow the [Instruction] that comes after it. Output a single JSON object and nothing else. '
          'Do not guess or fill in information that is not in the log, and give a positive verdict only when '
          'the evidence is clear.')


# ---------------------------------------------------------------- log rendering
def render_step(s):
    txt = f"[step {s['i']}] ({s.get('agent', '')}) " + (s.get('content') or '')
    if s.get('obs'):
        txt += '\n  <obs> ' + s['obs']
    return txt


def render_log(trace):
    """Render every normalized field and step without truncation."""
    head = []
    if trace.get('task_spec'):
        head.append(f"[TASK] {trace['task_spec']}")
    if trace.get('meta'):
        head.append('[META] ' + ', '.join(f'{k}={v}' for k, v in trace['meta'].items()))
    return '\n'.join(head + [render_step(s) for s in trace['steps']])


def shorten_log(text, byte_budget):
    """Keep head/tail (70/30) and visibly mark missing text; preserve UTF-8."""
    raw = text.encode('utf-8')
    if len(raw) <= byte_budget:
        return text
    marker = '\n...[log content omitted to fit context]...\n'
    remaining = byte_budget - len(marker.encode('utf-8'))
    if remaining < 2:
        raise InputTooLongError('input_too_long: no room for log after instructions and output budget')
    head = max(1, int(remaining * 0.7))
    tail = remaining - head
    return raw[:head].decode('utf-8', errors='ignore') + marker + raw[-tail:].decode('utf-8', errors='ignore')


class InputTooLongError(ValueError):
    """Even the fixed instructions and minimum output cannot fit."""


# ---------------------------------------------------------------- OpenRouter
class FatalAPIError(RuntimeError):
    """Configuration problem (bad key, unknown model, no credits, unsupported parameter). Stop the whole run."""


_models_cache = {}


def model_table(refresh=False):
    """{slug: model dict} from the public models endpoint. {} if the fetch fails."""
    if not _models_cache or refresh:
        try:
            req = urllib.request.Request(f'{BASE_URL}/models')
            with urllib.request.urlopen(req, timeout=40) as r:
                _models_cache.clear()
                _models_cache.update({m['id']: m for m in json.load(r)['data']})
        except Exception:
            pass
    return _models_cache


def model_prices(slug):
    """USD per token: dict(inp, out, cache_read, cache_write). Cache prices fall back to the input price."""
    info = model_table().get(slug)
    if not info:
        return None
    p = info.get('pricing', {})
    f = lambda k: float(p[k]) if p.get(k) not in (None, '') else None
    inp, out = f('prompt') or 0.0, f('completion') or 0.0
    return {'inp': inp, 'out': out, 'cache_read': f('input_cache_read') if f('input_cache_read') is not None else inp,
            'cache_write': f('input_cache_write') if f('input_cache_write') is not None else inp}


def safe_name(slug):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', slug)


def extract_json(text):
    m = re.search(r'\{.*\}', text or '', re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def add_usage(total, u):
    for k, v in (u or {}).items():
        if isinstance(v, (int, float)):
            total[k] = total.get(k, 0) + v
    return total


CACHE_PREFIXES = ('anthropic/', 'google/', 'qwen/')   # providers documented to need explicit cache_control


class LLM:
    def __init__(self, model=None, temperature=0.0, max_tokens='max', reasoning=None, json_mode=True,
                 cache='auto', timeout=600.0, max_retries=5):
        self.model = model or os.environ.get('DEFAULT_MODEL')
        if not self.model:
            raise FatalAPIError('No model given: pass --model <openrouter-slug> or set DEFAULT_MODEL in .env')
        self.key = os.environ.get('OPENROUTER_API_KEY')
        if not self.key:
            raise FatalAPIError('OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in.')
        self.max_tokens, self.timeout, self.max_retries = max_tokens, timeout, max_retries
        info = model_table().get(self.model)
        top = (info or {}).get('top_provider') or {}
        self.ctx = top.get('context_length') or (info or {}).get('context_length')          # context window
        self.max_out = top.get('max_completion_tokens') or 32768                              # model's max output
        if model_table() and info is None:
            raise FatalAPIError(f'Model "{self.model}" is not in the OpenRouter model list (check the slug).')
        supported = set(info.get('supported_parameters', [])) if info else None   # None = unknown, send nothing optional
        ok = lambda p: supported is not None and p in supported
        self.params = {}                                  # optional parameters actually sent (recorded in outputs)
        if temperature is not None and ok('temperature'):
            self.params['temperature'] = temperature
        if json_mode and ok('response_format'):
            self.params['response_format'] = {'type': 'json_object'}
        if reasoning and ok('reasoning'):
            self.params['reasoning'] = {'effort': reasoning}
        self.use_cache = cache == 'on' or (cache == 'auto' and self.model.startswith(CACHE_PREFIXES))
        self.skipped = [p for p, v in (('temperature', temperature), ('reasoning', reasoning)) if v is not None and p not in self.params]

    def _max_tokens(self, log_text, instruction):
        """Budget full input using UTF-8 bytes + framing allowance, not exact tokens.

        The request fitter trims the log only when this conservative budget requires it.
        """
        requested = self.max_tokens if isinstance(self.max_tokens, int) else self.max_out
        if not self.ctx:
            return requested
        estimate = len((SYSTEM + '[Execution Log]\n' + log_text + instruction).encode('utf-8')) + 2048
        available = self.ctx - estimate
        minimum = requested if isinstance(self.max_tokens, int) else min(1024, requested)
        if available < minimum:
            raise InputTooLongError(
                f'input_too_long: conservative input budget={estimate}, '
                f'output budget={minimum}, context={self.ctx}; fixed request budget cannot fit')
        return min(requested, available)

    def _body(self, log_text, instruction):
        log_part = {'type': 'text', 'text': '[Execution Log]\n' + log_text}
        if self.use_cache:
            log_part['cache_control'] = {'type': 'ephemeral'}
        body = {'model': self.model, 'max_tokens': self._max_tokens(log_text, instruction),
                'messages': [{'role': 'system', 'content': SYSTEM},
                             {'role': 'user', 'content': [log_part, {'type': 'text', 'text': instruction}]}]}
        body.update(self.params)
        return body

    def _post(self, body):
        headers = {'Authorization': f'Bearer {self.key}', 'Content-Type': 'application/json'}
        if os.environ.get('OPENROUTER_HTTP_REFERER'):
            headers['HTTP-Referer'] = os.environ['OPENROUTER_HTTP_REFERER']
        if os.environ.get('OPENROUTER_APP_TITLE'):
            headers['X-Title'] = os.environ['OPENROUTER_APP_TITLE']
        data = json.dumps(body).encode('utf-8')
        err = 'unknown'
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(f'{BASE_URL}/chat/completions', data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.load(r)
                e = resp.get('error')
                if not e:
                    return resp
                code, msg = e.get('code'), e.get('message', '')          # HTTP 200 with an error object
            except urllib.error.HTTPError as ex:
                code = ex.code
                try:
                    msg = json.load(ex).get('error', {}).get('message', '')
                except Exception:
                    msg = ex.reason
                retry_after = ex.headers.get('retry-after')
            except (urllib.error.URLError, TimeoutError, ConnectionError) as ex:
                code, msg, retry_after = 'network', str(ex), None
            else:
                retry_after = None
            err = f'{code}: {msg}'
            if code in (401, 402, 403, 404):
                hint = ' (402: OpenRouter reserves credit for max_tokens; top up or pass a lower --max-tokens N)' if code == 402 else ''
                raise FatalAPIError(f'HTTP {err}{hint}')                    # bad key / no credits / bad model
            low = str(msg).lower()
            if code == 413:
                return {'_error': f'input_too_long: {msg}'}
            if code == 400:
                if 'context' in low and ('length' in low or 'maximum' in low or 'exceed' in low) or 'too long' in low:
                    return {'_error': f'input_too_long: {msg}'}
                raise FatalAPIError(f'HTTP 400 (unsupported parameter?): {msg}')
            if attempt < self.max_retries:                                   # 408/429/5xx/network/provider errors
                time.sleep(min(float(retry_after) if retry_after else 2 ** attempt + random.random(), 60))
        return {'_error': err}

    def ask(self, log_text, instruction, attempts=2):
        """-> (parsed JSON dict or {'_error': ...}, usage dict). Config errors raise FatalAPIError."""
        usage, last = {}, 'unknown'
        original = log_text
        try:
            if self.ctx:
                requested = self.max_tokens if isinstance(self.max_tokens, int) else min(1024, self.max_out)
                fixed = len((SYSTEM + '[Execution Log]\n' + instruction).encode('utf-8')) + 2048
                log_text = shorten_log(original, self.ctx - fixed - requested)
            body = self._body(log_text, instruction)
        except InputTooLongError as exc:
            return {'_error': str(exc)}, usage
        reductions = 0
        for _ in range(attempts):
            resp = self._post(body)
            while str(resp.get('_error', '')).startswith('input_too_long:') and reductions < 8:
                reductions += 1
                try:
                    log_text = shorten_log(original, len(log_text.encode('utf-8')) // 2)
                    body = self._body(log_text, instruction)
                except InputTooLongError:
                    break
                resp = self._post(body)
            # Numeric fields survive per-layer and per-run usage aggregation.
            omitted = max(0, len(original) - len(log_text.replace('\n...[log content omitted to fit context]...\n', '')))
            usage.update({'input_truncated_calls': int(log_text != original),
                          'input_original_chars': len(original),
                          'input_omitted_chars': omitted,
                          'context_retries': reductions})
            if '_error' in resp:
                return {'_error': resp['_error']}, usage
            u = resp.get('usage') or {}
            det = u.get('prompt_tokens_details') or {}
            add_usage(usage, {'prompt_tokens': u.get('prompt_tokens'), 'completion_tokens': u.get('completion_tokens'),
                              'cached_tokens': det.get('cached_tokens'), 'cache_write_tokens': det.get('cache_write_tokens'),
                              'cost': u.get('cost')})
            ch = (resp.get('choices') or [{}])[0]
            text = (ch.get('message') or {}).get('content') or ''
            if ch.get('finish_reason') == 'length' and not extract_json(text):
                return {'_error': 'finish_reason=length (hit the output limit; reasoning tokens count toward it)'}, usage
            parsed = extract_json(text)
            if parsed is not None:
                return parsed, usage
            last = 'json_parse_failed'
        return {'_error': last}, usage


# ---------------------------------------------------------------- I/O / selection
def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


class Appender:
    """Thread-safe jsonl appender (flush per row so an interrupted run can resume)."""
    def __init__(self, path):
        self.f, self.lock = open(path, 'a', encoding='utf-8'), threading.Lock()

    def write(self, row):
        with self.lock:
            self.f.write(json.dumps(row, ensure_ascii=False) + '\n')
            self.f.flush()


def latest(rows, key=lambda r: r['trace_id']):
    """When a key was written several times by re-runs, keep the last row."""
    return list({key(r): r for r in rows}.values())


def done_ids(path, key=lambda r: r['trace_id']):
    """Keys already finished. Rows that still carry `errors` are re-run."""
    if not os.path.exists(path):
        return set()
    return {key(r) for r in latest(read_jsonl(path), key) if not r.get('errors')}


def add_llm_args(ap):
    ap.add_argument('--model', help='OpenRouter slug, e.g. anthropic/claude-sonnet-5 (default: DEFAULT_MODEL in .env)')
    ap.add_argument('--temperature', type=float, default=0.0, help='sent only if the model supports it (default 0)')
    ap.add_argument('--reasoning', choices=['minimal', 'low', 'medium', 'high', 'xhigh'],
                    help='reasoning effort, sent only if the model supports `reasoning`')
    ap.add_argument('--max-tokens', type=lambda x: x if x == 'max' else int(x), default='max',
                    help="output limit; 'max' (default) = the model's maximum output. Reasoning tokens count toward it")
    ap.add_argument('--no-json-mode', action='store_true', help='do not send response_format=json_object')
    ap.add_argument('--cache', choices=['auto', 'on', 'off'], default='auto', help='cache_control breakpoints')
    ap.add_argument('--workers', type=int, default=4)


def make_llm(a):
    llm = LLM(a.model, a.temperature, a.max_tokens, a.reasoning, not a.no_json_mode, a.cache)
    mt = f'model max {llm.max_out}' if a.max_tokens == 'max' else a.max_tokens
    print(f'[llm] {llm.model} | max_tokens: {mt} | params sent: {llm.params or "none"} | cache_control: {llm.use_cache}'
          + (f' | NOT supported, skipped: {llm.skipped}' if llm.skipped else ''), file=sys.stderr)
    return llm


def add_select_args(ap):
    ap.add_argument('traces', nargs='?', default=DEFAULT_TRACES, help='default: data/traces.jsonl')
    ap.add_argument('--subset', action='append', help='e.g. WhoWhen/AG, AEB/GAIA, AgentRx/tau_retail (repeatable)')
    ap.add_argument('--limit', type=int, help='keep the first N after selection (pilot runs)')
    ap.add_argument('--sample', type=int, help='random N after selection (fixed seed)')
    ap.add_argument('--seed', type=int, default=20260920)


def select_traces(args):
    ts = read_jsonl(find_traces(args.traces))
    if args.subset:
        keys = [s.split('/') for s in args.subset]
        ts = [t for t in ts if any(t['benchmark'] == k[0] and (len(k) == 1 or t['subset'] == k[1]) for k in keys)]
    if args.sample:
        ts = random.Random(args.seed).sample(ts, min(args.sample, len(ts)))
    if args.limit:
        ts = ts[:args.limit]
    return ts


def run_jobs(todo, fn, out, workers):
    """Run fn(item) -> row over `todo` in a thread pool, appending each row to `out` as it finishes.
    A FatalAPIError (bad key/model/credits/parameter) cancels the queue and propagates. -> summed usage."""
    w, usage, n = Appender(out), {}, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, t) for t in todo]
        for f in as_completed(futs):
            try:
                row = f.result()
            except FatalAPIError:
                for x in futs:
                    x.cancel()
                raise
            w.write(row)
            add_usage(usage, row.get('usage'))
            n += 1
            if n % 10 == 0 or n == len(todo):
                print(f'{n}/{len(todo)}  spent ${usage.get("cost", 0):.4f}', file=sys.stderr)
    return usage


# ---------------------------------------------------------------- environment check
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Check the OpenRouter setup: key, models, prices, supported parameters.')
    ap.add_argument('--model', action='append', help='slug to inspect (repeatable)')
    ap.add_argument('--ping', action='store_true', help='send one tiny request per model (costs a fraction of a cent)')
    ap.add_argument('--search', help='list model slugs containing this text, e.g. --search haiku')
    a = ap.parse_args()
    print(f'.env loaded | OPENROUTER_API_KEY: {"set" if os.environ.get("OPENROUTER_API_KEY") else "MISSING"} | base: {BASE_URL}')
    tab = model_table()
    print(f'model list: {len(tab)} models' if tab else 'model list: fetch FAILED (network?)')
    if a.search:
        print('slugs matching "%s":' % a.search, [i for i in tab if a.search.lower() in i.lower() and ':' not in i][:40])
    for slug in a.model or ([os.environ['DEFAULT_MODEL']] if os.environ.get('DEFAULT_MODEL') else []):
        info, pr = tab.get(slug), model_prices(slug)
        if not info:
            print(f'- {slug}: NOT FOUND'); continue
        sp = set(info.get('supported_parameters', []))
        print(f'- {slug}: ctx={info.get("context_length")} | $/Mtok in={pr["inp"] * 1e6:.3f} out={pr["out"] * 1e6:.3f} '
              f'cache_read={pr["cache_read"] * 1e6:.3f} | temperature={"temperature" in sp} '
              f'json={"response_format" in sp} reasoning={"reasoning" in sp}')
        if a.ping:
            llm = LLM(slug, max_tokens=300)
            res, u = llm.ask('[step 0] (agent) hello', 'Reply with {"ok": true}.')
            print(f'    ping -> {res} | cost=${u.get("cost", 0):.6f}')
