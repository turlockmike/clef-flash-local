"""Compare models behind a /v1/systemone endpoint on the 12-ticket sanity set: accuracy, latency, throughput, VRAM.

    python bench/bench_models.py [--url http://127.0.0.1:11436] [--concurrency 16] [models...]

Default models: tev1:0.8b tev1 nimble clef-flash. Each model is unloaded from Ollama (`ollama stop`) after its run.
"""
import argparse, json, os, statistics, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from sanity_data import DATA, QUESTIONS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--url", default=os.environ.get("S1_URL", "http://127.0.0.1:11436"))
ap.add_argument("--concurrency", type=int, default=16)
ap.add_argument("--rounds", type=int, default=4, help="sequential passes over the set for latency")
ap.add_argument("models", nargs="*", default=["tev1:0.8b", "tev1", "nimble", "clef-flash"])
a = ap.parse_args()
URL = a.url.rstrip("/") + "/v1/systemone"


def call(model, msg):
    body = json.dumps({"model": model, "state": {"message": msg}, "questions": QUESTIONS}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    return out, (time.perf_counter() - t) * 1000


def vram():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        return round(int(q.stdout.strip()) / 1024, 1)
    except Exception:
        return None


for model in a.models:
    t = time.perf_counter(); first, _ = call(model, DATA[0][0]); cold = time.perf_counter() - t
    for _ in range(3):
        call(model, DATA[1][0])
    acc = {"intent": 0, "spam": 0, "urgency": 0}; lat = []; misses = []
    for _ in range(a.rounds):
        for msg, *_g in DATA:
            lat.append(call(model, msg)[1])
    for msg, gi, gs, gu in DATA:
        ans = call(model, msg)[0]["answers"]
        pi, ps, pu = ans["intent"]["choice"], ans["spam"]["noul"] > 0.5, round(ans["urgency"]["score"])
        acc["intent"] += pi == gi; acc["spam"] += ps == gs; acc["urgency"] += pu == gu
        if (pi, ps, pu) != (gi, gs, gu):
            misses.append(f"{msg[:40]}... got {pi}/{ans['spam']['noul']:.2f}/{ans['urgency']['score']:.2f} "
                          f"want {gi}/{gs}/{gu}")
    with ThreadPoolExecutor(a.concurrency) as ex:
        t = time.perf_counter()
        list(ex.map(lambda m: call(model, m), [d[0] for d in DATA] * a.concurrency))
        thr = len(DATA) * a.concurrency / (time.perf_counter() - t)
    lat.sort()
    print("RESULT", json.dumps({
        "model": model, "cold_s": round(cold, 1), "vram_gb": vram(),
        "p50_ms": round(statistics.median(lat), 1), "p95_ms": round(lat[int(.95 * len(lat))], 1),
        f"concurrent{a.concurrency}_req_per_s": round(thr, 1),
        "input_tokens": first.get("usage", {}).get("input_tokens"),
        "acc": {k: f"{v}/{len(DATA)}" for k, v in acc.items()}}), flush=True)
    for m in misses:
        print("   miss:", m, flush=True)
    subprocess.run(["ollama", "stop", model], capture_output=True)
