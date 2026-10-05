"""Integration tests against a running server (default http://127.0.0.1:11436, override with S1_URL).

    pytest tests/ -v

Requires the server to be up (scripts/start_server.sh). Ollama pass-through tests are skipped when
Ollama is not reachable through the proxy.
"""
import json, os, sys, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from sanity_data import DATA, QUESTIONS, record  # noqa: E402

URL = os.environ.get("S1_URL", "http://127.0.0.1:11436").rstrip("/")
MODEL = os.environ.get("S1_MODEL", "clef-flash")


def post(path, body, timeout=300):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def decide(state, questions, model=MODEL):
    return post("/v1/systemone", {"model": model, "state": state, "questions": questions})


@pytest.fixture(scope="session", autouse=True)
def server_up():
    try:
        with urllib.request.urlopen(URL + "/s1/health", timeout=5) as r:
            h = json.load(r)
    except Exception as e:
        pytest.skip(f"server not reachable at {URL}: {e}")
    assert h["ok"]
    return h


def test_health_shape(server_up):
    for k in ("local_models", "upstream", "queue", "requests", "batches"):
        assert k in server_up


def test_response_shape():
    r = decide({"ticket": "I was charged twice"}, QUESTIONS)
    a = r["answers"]
    assert r["model"] == MODEL
    assert r["usage"]["input_tokens"] > 0
    assert a["intent"]["choice"] in QUESTIONS["intent"]["criteria"]
    assert abs(sum(a["intent"]["probabilities"].values()) - 1) < 1e-3
    assert 0 <= a["spam"]["noul"] <= 1
    assert 0 <= a["urgency"]["score"] <= 2
    assert len(a["urgency"]["probabilities"]) == 3


def test_state_can_be_a_string():
    r = decide("Checkout is down and orders are blocked.",
               {"outage": {"type": "noul", "instructions": "Is a service down?"}})
    assert r["answers"]["outage"]["noul"] > 0.5


def test_instructions_optional():
    r = decide({"text": "hello"}, {"is_greeting": {"type": "noul"}})
    assert 0 <= r["answers"]["is_greeting"]["noul"] <= 1


def test_sanity_set_accuracy():
    """Measured ~11/12 per question for clef-flash on this set; require >= 10/12 on each."""
    answers = [decide(record(m)["state"], QUESTIONS)["answers"] for m, *_ in DATA]
    acc = {"intent": 0, "spam": 0, "urgency": 0}
    misses = []
    for (msg, gi, gs, gu), a in zip(DATA, answers):
        got = (a["intent"]["choice"], a["spam"]["noul"] > 0.5, round(a["urgency"]["score"]))
        acc["intent"] += got[0] == gi
        acc["spam"] += got[1] == gs
        acc["urgency"] += got[2] == gu
        if got != (gi, gs, gu):
            misses.append((msg[:40], got, (gi, gs, gu)))
    print(acc, misses)
    assert all(v >= 10 for v in acc.values()), (acc, misses)


def test_concurrent_requests_batch():
    states = [record(m)["state"] for m, *_ in DATA] * 2
    with ThreadPoolExecutor(16) as ex:
        rs = list(ex.map(lambda s: decide(s, QUESTIONS), states))
    assert len(rs) == len(states)
    assert all("answers" in r for r in rs)
    # the same input through the batched and single paths must agree
    single = decide(states[0], QUESTIONS)["answers"]["intent"]["choice"]
    assert rs[0]["answers"]["intent"]["choice"] == single


@pytest.mark.parametrize("body,msg", [
    ({"model": MODEL, "questions": {"q": {"type": "noul"}}}, "state is required"),
    ({"model": MODEL, "state": "x", "questions": {}}, "at least one question"),
    ({"model": MODEL, "state": "x", "questions": {"q": {"type": "bogus"}}}, "type must be"),
    ({"model": MODEL, "state": "x", "questions": {"q": {"type": "choice", "criteria": {}}}},
     "criteria must not be empty"),
    ({"model": MODEL, "state": "x", "images": ["x"], "questions": {"q": {"type": "noul"}}}, "images/videos"),
])
def test_validation_errors(body, msg):
    with pytest.raises(urllib.error.HTTPError) as e:
        post("/v1/systemone", body)
    assert e.value.code == 400
    assert msg in e.value.read().decode()


def test_bad_json_is_400():
    req = urllib.request.Request(URL + "/v1/systemone", b"{not json", {"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 400


def _ollama_up():
    try:
        urllib.request.urlopen(URL + "/api/tags", timeout=5)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_up(), reason="Ollama upstream not reachable")
def test_proxy_passthrough_tags():
    with urllib.request.urlopen(URL + "/api/tags", timeout=10) as r:
        assert "models" in json.load(r)


@pytest.mark.skipif(not _ollama_up(), reason="Ollama upstream not reachable")
def test_proxy_other_systemone_model():
    tags = json.load(urllib.request.urlopen(URL + "/api/tags"))
    cand = [m["name"] for m in tags["models"] if m["name"].startswith(("tev1", "nimble"))]
    if not cand:
        pytest.skip("no tev1/nimble model pulled in Ollama")
    r = decide({"ticket": "refund please"},
               {"refund": {"type": "noul", "instructions": "Asks for a refund?"}}, model=cand[0])
    assert 0 <= r["answers"]["refund"]["noul"] <= 1
