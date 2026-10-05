"""Concurrency load test for a /v1/systemone endpoint.

    python loadtest.py [--url http://127.0.0.1:11436] [--model clef-flash] [--n 256] [--concurrency 32]
"""
import argparse, json, statistics, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:11436")
ap.add_argument("--model", default="clef-flash")
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--concurrency", type=int, default=32)
a = ap.parse_args()

TICKETS = ["I was charged twice this month, please refund one.",
           "Production is down with 502 errors, customers can't check out!",
           "How do I reset my password? The email never arrives.",
           "What does the enterprise plan cost for 200 seats?"]
Q = {"team": {"type": "choice", "instructions": "Which team should handle this?",
              "criteria": {"billing": "Payments, refunds", "technical": "Bugs, outages", "account": "Login, profile",
                           "sales": "Pricing, plans"}},
     "spam": {"type": "noul", "instructions": "Is this spam or a scam?"},
     "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Routine", "Soon", "Emergency"]}}


def call(i):
    body = json.dumps({"model": a.model, "state": {"ticket": TICKETS[i % len(TICKETS)]}, "questions": Q}).encode()
    t = time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(a.url + "/v1/systemone", body,
                                                       {"Content-Type": "application/json"}), timeout=600) as r:
        out = json.load(r)
    return (time.perf_counter() - t) * 1000, out["usage"]["input_tokens"]


call(0)
with ThreadPoolExecutor(a.concurrency) as ex:
    t = time.perf_counter()
    res = list(ex.map(call, range(a.n)))
    wall = time.perf_counter() - t
lat = sorted(r[0] for r in res)
print(json.dumps({"model": a.model, "n": a.n, "concurrency": a.concurrency, "req_s": round(a.n / wall, 1),
                  "p50_ms": round(statistics.median(lat)), "p95_ms": round(lat[int(.95 * len(lat))]),
                  "avg_input_tokens": round(statistics.mean(r[1] for r in res))}))
