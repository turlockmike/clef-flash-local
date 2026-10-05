"""Clef-flash throughput sweep. usage: bench_fast.py <fp8|nf4> [compile]"""
import sys, time, json
import torch
from huggingface_hub import snapshot_download

MODE = sys.argv[1]
COMPILE = "compile" in sys.argv
path = snapshot_download("Cloudflare/clef-flash")
sys.path.insert(0, path)
from joint_schema_model import load_release_model, encode_record, collate_records
import os, sys as _s; _s.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from sanity_data import DATA, record as rec

torch.backends.cuda.matmul.allow_tf32 = True
if MODE == "nf4":
    from transformers import BitsAndBytesConfig
    model, proc = load_release_model(path, device="cuda", quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16))
else:
    from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig, PerRow, PerTensor
    # load bf16 on CPU, then quantize each Linear while moving it to the GPU
    model, proc = load_release_model(path, device="cpu")
    lm = model.language_model.model.language_model
    quantize_(lm, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()), device="cuda")
    model = model.to("cuda")
torch.cuda.empty_cache()
print("weights GB", round(torch.cuda.memory_allocated() / 2**30, 2), flush=True)

tok = proc.tokenizer
dev = torch.device("cuda")
enc = [encode_record(tok, rec(m)) for m, *_ in DATA]
avg_tokens = sum(len(e.input_ids) for e in enc) / len(enc)

fwd = model
if COMPILE:
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 64
    for layer in model.language_model.model.language_model.layers:
        layer.compile(dynamic=True)

@torch.inference_mode()
def run(batch):
    out = fwd(collate_records(batch, tok.pad_token_id, dev))
    torch.cuda.synchronize()
    return out

# correctness check vs labels
outs = run(enc)
ok = 0
for (msg, gi, gs, gu), e, lg in zip(DATA, enc, outs):
    a = {q.question_id: dict(zip(q.option_ids, l.float().softmax(-1).tolist())) for q, l in zip(e.questions, lg)}
    ok += (max(a["intent"], key=a["intent"].get) == gi) + ((a["spam"]["true"] > .5) == gs) + (int(max(a["urgency"], key=a["urgency"].get)) == gu)
print(f"accuracy {ok}/36", flush=True)

rows = []
for bs in [1, 4, 8, 16, 24, 32]:
    batch = (enc * (bs // len(enc) + 1))[:bs]
    try:
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2): run(batch)
        n = max(3, 64 // bs)
        t = time.perf_counter()
        for _ in range(n): run(batch)
        dt = (time.perf_counter() - t) / n
        r = {"bs": bs, "ms_per_batch": round(dt * 1000, 1), "req_s": round(bs / dt, 1),
             "tok_s": round(bs * avg_tokens / dt), "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
    except torch.OutOfMemoryError:
        r = {"bs": bs, "oom": True}; torch.cuda.empty_cache()
    print("ROW", json.dumps(r), flush=True)
    if r.get("oom"): break
print("avg_tokens", round(avg_tokens))
