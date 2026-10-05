"""Client for /v1/systemone (local fast server, Ollama, or any Jev-compatible endpoint).

Python:
    from s1 import decide, decide_many
    r = decide({"ticket": "charged twice"}, {"refund": {"type": "noul", "instructions": "Asks for a refund?"}})
    r["answers"]["refund"]["noul"]                       # -> 0.97
    rs = decide_many([state1, state2, ...], questions)   # concurrent; server micro-batches them

CLI:
    python s1.py --state '{"ticket": "charged twice"}' --questions @questions.json [--model tev1]
    python s1.py --jsonl states.jsonl --questions @questions.json > answers.jsonl   # one state per line

Env: S1_URL (default http://127.0.0.1:11436), S1_MODEL (default clef-flash)
"""
import argparse, json, os, sys, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("S1_URL", "http://127.0.0.1:11436")
MODEL = os.environ.get("S1_MODEL", "clef-flash")


def decide(state, questions, model=None, url=None, timeout=300):
    body = json.dumps({"model": model or MODEL, "state": state, "questions": questions}).encode()
    req = urllib.request.Request((url or URL).rstrip("/") + "/v1/systemone", body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"systemone {e.code}: {e.read().decode(errors='replace')}") from None


def decide_many(states, questions, model=None, url=None, concurrency=16):
    with ThreadPoolExecutor(concurrency) as ex:
        return list(ex.map(lambda s: decide(s, questions, model, url), states))


def _load(v):
    if v.startswith("@"):
        with open(v[1:], encoding="utf-8") as f:
            return json.load(f)
    return json.loads(v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True, help="JSON or @file.json")
    ap.add_argument("--state", help="JSON or @file.json")
    ap.add_argument("--jsonl", help="file with one state JSON per line")
    ap.add_argument("--model")
    ap.add_argument("--url")
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()
    qs = _load(a.questions)
    if a.jsonl:
        with open(a.jsonl, encoding="utf-8") as f:
            states = [json.loads(line) for line in f if line.strip()]
        for r in decide_many(states, qs, a.model, a.url, a.concurrency):
            print(json.dumps(r))
    elif a.state:
        print(json.dumps(decide(_load(a.state), qs, a.model, a.url), indent=2))
    else:
        sys.exit("pass --state or --jsonl")
