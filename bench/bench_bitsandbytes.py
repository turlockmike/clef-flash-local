"""bitsandbytes nf4 / int8 Clef-flash benchmark (in-process, no server). usage: bench_bitsandbytes.py <nf4|int8>"""
import sys, time, statistics, json
import torch
from transformers import BitsAndBytesConfig

MODE = sys.argv[1]  # nf4 | int8
REPO = sys.argv[2] if len(sys.argv) > 2 else "Cloudflare/clef-flash"
from huggingface_hub import snapshot_download
path = snapshot_download(REPO)
sys.path.insert(0, path)
from joint_schema_model import load_release_model, encode_record, collate_records, question_options

qc = {
    "nf4": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
    "int8": BitsAndBytesConfig(load_in_8bit=True),
}[MODE]

t0 = time.time()
model, proc = load_release_model(path, device="cuda", quantization_config=qc)
load_s = time.time() - t0
vram_model = torch.cuda.memory_allocated() / 2**30
tok = proc.tokenizer
dev = torch.device("cuda")

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from sanity_data import DATA, record as rec

encoded = [encode_record(tok, rec(m)) for m, *_ in DATA]

@torch.inference_mode()
def run(batch_encoded):
    batch = collate_records(batch_encoded, tok.pad_token_id, dev)
    out = model(batch)
    torch.cuda.synchronize()
    return out

# accuracy
correct = {"intent": 0, "spam": 0, "urgency": 0}
outs = run(encoded)
for (msg, gi, gs, gu), enc, logits in zip(DATA, encoded, outs):
    ans = {}
    for q, lq in zip(enc.questions, logits):
        p = dict(zip(q.option_ids, lq.float().softmax(-1).tolist()))
        ans[q.question_id] = p
    pi = max(ans["intent"], key=ans["intent"].get)
    ps = ans["spam"]["true"] > 0.5
    pu = max(ans["urgency"], key=ans["urgency"].get)
    correct["intent"] += pi == gi
    correct["spam"] += ps == gs
    correct["urgency"] += int(pu) == gu
    print(f"  {msg[:55]:55} intent={pi:9} spam={ans['spam']['true']:.2f} urg={pu}  (gold {gi},{gs},{gu})")

# latency, batch 1
for _ in range(5): run([encoded[0]])
lat = []
for i in range(60):
    t = time.perf_counter(); run([encoded[i % len(encoded)]]); lat.append((time.perf_counter() - t) * 1000)
lat.sort()
# throughput, batch 16
big = (encoded * 2)[:16]
run(big)
t = time.perf_counter(); N = 10
for _ in range(N): run(big)
thr = 16 * N / (time.perf_counter() - t)

res = {"mode": MODE, "repo": REPO, "load_s": round(load_s, 1), "vram_weights_gb": round(vram_model, 2),
       "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
       "p50_ms": round(statistics.median(lat), 1), "p95_ms": round(lat[int(0.95 * len(lat))], 1),
       "batch16_records_per_s": round(thr, 1), "prompt_tokens_avg": round(sum(len(e.input_ids) if hasattr(e, 'input_ids') else 0 for e in encoded) / len(encoded)),
       "acc": {k: f"{v}/{len(DATA)}" for k, v in correct.items()}}
print("RESULT", json.dumps(res))
