"""Ollama-compatible front-end with a fast local Clef-flash.

POST /v1/systemone with model "clef-flash*" runs here (FP8 + compiled layers, micro-batched).
Every other request (any path, any model, streaming included) passes through to Ollama unchanged,
so clients use this exactly like an Ollama server.

    python s1_server.py [--host 0.0.0.0] [--port 11436] [--ollama http://127.0.0.1:11434]

Extra endpoint: GET /s1/health
"""
import argparse, json, queue, sys, threading, time, traceback, http.client, urllib.parse
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve the LAN")
ap.add_argument("--port", type=int, default=11436)
ap.add_argument("--ollama", default="http://127.0.0.1:11434")
ap.add_argument("--max-batch", type=int, default=16)
ap.add_argument("--max-batch-tokens", type=int, default=12000, help="padded tokens per forward; above ~16k Windows spills VRAM")
ap.add_argument("--batch-wait-ms", type=float, default=2.0)
ap.add_argument("--no-compile", action="store_true")
args = ap.parse_args()
UP = urllib.parse.urlsplit(args.ollama)

import torch
from huggingface_hub import snapshot_download

path = snapshot_download("Cloudflare/clef-flash")
sys.path.insert(0, path)
from joint_schema_model import (QUESTION_TYPES, collate_records, encode_record,  # noqa: E402
                                load_release_model, systemone_answer)
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerTensor, quantize_  # noqa: E402

t0 = time.time()
model, proc = load_release_model(path, device="cpu")
# Per-row FP8 scaling is unsupported on sm_120 in current torch builds; per-tensor works (~220 TFLOPS on a 5080).
quantize_(model.language_model.model.language_model,
          Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()), device="cuda")
model = model.to("cuda").eval()
torch.cuda.empty_cache()
if not args.no_compile:
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 64
    # Compile per decoder layer: Clef's forward bypasses wrappers, and this fuses torchao's quant ops (~5x faster).
    for layer in model.language_model.model.language_model.layers:
        layer.compile(dynamic=True)
tok = proc.tokenizer
DEV = torch.device("cuda")
work: "queue.Queue[tuple]" = queue.Queue()
recent = []
stats = {"requests": 0, "batches": 0, "forward_s": 0.0, "max_batch_seen": 0}


def is_clef(name):
    return str(name).lower().split("/")[-1].startswith("clef-flash")


def validate(req):
    qs = req.get("questions")
    if "state" not in req:
        raise ValueError("state is required")
    if not isinstance(qs, dict) or not qs:
        raise ValueError("at least one question is required")
    for qid, q in qs.items():
        if q.get("type") not in QUESTION_TYPES:
            raise ValueError(f"{qid}: type must be noul, choice, or score")
        if q["type"] != "noul" and not q.get("criteria"):
            raise ValueError(f"{qid}: criteria must not be empty")
    if req.get("images") or req.get("videos"):
        raise ValueError("images/videos are not supported by the fast path; send them to Ollama directly")


BUCKETS = [b for b in (1, 2, 4, 8, 16, 32, 64) if b <= args.max_batch] or [1]
SEQ_MULTIPLE = 64


@torch.inference_mode()
def forward(items):
    # Pad to a small fixed set of shapes (batch buckets x 64-token multiples): every new shape costs a
    # multi-second recompile, so warmup() compiles them all up front.
    t = time.perf_counter()
    bucket = next((b for b in BUCKETS if b >= len(items)), len(items))
    encs = [it[0] for it in items] + [items[-1][0]] * (bucket - len(items))
    batch = collate_records(encs, tok.pad_token_id, DEV)
    extra = -batch["input_ids"].shape[1] % SEQ_MULTIPLE
    if extra:
        batch["input_ids"] = torch.nn.functional.pad(batch["input_ids"], (0, extra), value=tok.pad_token_id)
        batch["attention_mask"] = torch.nn.functional.pad(batch["attention_mask"], (0, extra), value=0)
    outs = model(batch)
    torch.cuda.synchronize()
    stats["forward_s"] = round(stats["forward_s"] + time.perf_counter() - t, 3)
    stats["batches"] += 1
    stats["max_batch_seen"] = max(stats["max_batch_seen"], len(items))
    recent.append((len(items), max(len(it[0].input_ids) for it in items), round((time.perf_counter() - t) * 1000)))
    del recent[:-20]
    for (enc, req, fut), logits in zip(items, outs):
        answers = {q.question_id: systemone_answer(req["questions"][q.question_id],
                                                   dict(zip(q.option_ids, lq.float().softmax(-1).tolist())))
                   for q, lq in zip(enc.questions, logits)}
        fut.set_result({"model": req["model"], "answers": answers,
                        "usage": {"input_tokens": len(enc.input_ids), "output_tokens": 0}})


def batcher():
    pending = None
    while True:
        items = [pending or work.get()]
        pending = None
        deadline = time.perf_counter() + args.batch_wait_ms / 1000
        longest = len(items[0][0].input_ids)
        while len(items) < args.max_batch:
            try:
                it = work.get(timeout=max(0.0, deadline - time.perf_counter()))
            except queue.Empty:
                break
            n = max(longest, len(it[0].input_ids))
            if n * (len(items) + 1) > args.max_batch_tokens:
                pending = it
                break
            items.append(it)
            longest = n
        try:
            forward(items)
        except Exception as e:  # fail this batch, keep serving
            traceback.print_exc()
            for it in items:
                if not it[2].done():
                    it[2].set_exception(e)


def warmup():
    for words in (20, 300, 900):
        req = {"model": "clef-flash", "state": {"text": "warm " * words},
               "questions": {"q": {"type": "noul", "instructions": "Is this a warmup?"}}}
        enc = encode_record(tok, req)
        for bs in BUCKETS:
            if bs * len(enc.input_ids) <= args.max_batch_tokens:
                forward([(enc, req, Future()) for _ in range(bs)])


HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-connection", "host", "content-length"}


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def do_GET(self):
        if self.path == "/s1/health":
            return self.send_json(200, {"ok": True, "local_models": ["clef-flash"], "upstream": args.ollama,
                                        "queue": work.qsize(), **stats, "recent_batches_bs_len_ms": recent})
        self.proxy(b"")

    def do_HEAD(self):
        self.proxy(b"")

    def do_DELETE(self):
        self.proxy(self.body())

    def do_POST(self):
        raw = self.body()
        if self.path.split("?")[0] == "/v1/systemone":
            try:
                req = json.loads(raw)
            except json.JSONDecodeError as e:
                return self.send_json(400, {"error": f"bad json: {e}"})
            if is_clef(req.get("model", "")):
                return self.clef(req)
        self.proxy(raw)

    def clef(self, req):
        try:
            validate(req)
            enc = encode_record(tok, req)
        except (ValueError, KeyError) as e:
            return self.send_json(400, {"error": str(e)})
        fut = Future()
        stats["requests"] += 1
        work.put((enc, req, fut))
        try:
            self.send_json(200, fut.result(timeout=300))
        except Exception as e:
            self.send_json(500, {"error": str(e)})

    def proxy(self, raw):
        """Stream any request through to Ollama (works for NDJSON / SSE streaming responses)."""
        try:
            conn = http.client.HTTPConnection(UP.hostname, UP.port or 80, timeout=600)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            conn.request(self.command, self.path, body=raw or None, headers=headers)
            resp = conn.getresponse()
        except Exception as e:
            return self.send_json(502, {"error": f"ollama upstream unreachable: {e}"})
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in HOP:
                self.send_header(k, v)
        length = resp.getheader("Content-Length")
        if length is not None:
            self.send_header("Content-Length", length)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(resp.read())
        else:
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while chunk := resp.read1(65536):
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        conn.close()


if __name__ == "__main__":
    warmup()
    threading.Thread(target=batcher, daemon=True).start()
    print(f"READY http://{args.host}:{args.port} upstream={args.ollama} load+compile={time.time() - t0:.0f}s "
          f"vram={torch.cuda.memory_allocated() / 2**30:.1f}GB", flush=True)
    ThreadingHTTPServer((args.host, args.port), H).serve_forever()
