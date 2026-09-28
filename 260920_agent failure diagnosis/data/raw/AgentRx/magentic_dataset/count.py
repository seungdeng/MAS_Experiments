#!/usr/bin/env python3
import os, sys, json

STEP_KEYS = (
    "steps", "raw_steps", "messages", "events", "trace", "trajectory",
    "log", "turns", "actions"
)

def count_steps_in_obj(obj):
    # If the whole file is already a list of steps
    if isinstance(obj, list):
        return len(obj)

    if not isinstance(obj, dict):
        return None

    # Direct common keys
    for k in STEP_KEYS:
        v = obj.get(k)
        if isinstance(v, list):
            return len(v)

    # Common nesting
    for container_key in ("raw", "data", "result", "output"):
        v = obj.get(container_key)
        if isinstance(v, dict):
            for k in STEP_KEYS:
                vv = v.get(k)
                if isinstance(vv, list):
                    return len(vv)

    # Fallback: if exactly one list exists in the dict, assume it's the steps
    lists = [v for v in obj.values() if isinstance(v, list)]
    if len(lists) == 1:
        return len(lists[0])

    return None

def main():
    cwd = os.getcwd()
    files = [f for f in os.listdir(".") if f.lower().endswith((".json", ".jsonl"))]
    files = [f for f in files if os.path.isfile(f) and f != os.path.basename(__file__)]

    results = {}
    processed = 0
    matched = 0

    for fname in sorted(files):
        processed += 1
        stem = os.path.splitext(fname)[0]

        try:
            if fname.lower().endswith(".jsonl"):
                # For JSONL: if each line is a record with steps, store as stem#lineN
                with open(fname, "r", encoding="utf-8") as fh:
                    line_no = 0
                    for line in fh:
                        s = line.strip()
                        if not s:
                            continue
                        line_no += 1
                        try:
                            obj = json.loads(s)
                        except Exception:
                            continue
                        n = count_steps_in_obj(obj)
                        if n is not None:
                            results[f"{stem}#{line_no}"] = n
                            matched += 1
            else:
                with open(fname, "r", encoding="utf-8") as fh:
                    obj = json.load(fh)
                n = count_steps_in_obj(obj)
                if n is not None:
                    results[stem] = n
                    matched += 1
        except Exception:
            # Skip unreadable/bad json files silently
            continue

    out_path = os.path.join(cwd, "steps_by_id.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    # Print the mapping too (so you can redirect if you want)
    print(json.dumps(results, indent=2, sort_keys=True))

    # Minimal status to stderr
    print(f"[ok] scanned={processed} files, wrote={len(results)} ids -> {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
