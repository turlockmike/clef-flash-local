# clef-flash-local

Run [Cloudflare's Clef-Flash](https://huggingface.co/Cloudflare/clef-flash) decision model on a single consumer
GPU as a drop-in Ollama endpoint that speaks the Jev / SystemOne `POST /v1/systemone` API.

Clef-Flash is a 9B model (post-trained from Qwen3.5-9B) that takes a **state** (text or JSON) and a schema of
**typed questions**, and returns a calibrated probability for every option of every question in one forward
pass. No text generation, no output parsing. It is meant for high-volume structured judgments: routing, triage,
classification, moderation, yes/no gates, rubric scores.

This repo is the glue that makes it fast on a 16 GB card:

- FP8 weights via torchao (per-tensor scaling), `torch.compile` on every decoder layer
- micro-batching with fixed shape buckets so compiled kernels are reused instead of recompiled
- everything that is not `clef-flash` (other models, `/api/chat`, `/api/tags`, streaming) is proxied to a local
  Ollama, so one URL serves both

Measured on an RTX 5080 (Windows 11, driver 617, torch 2.14 cu130):

| model | where | latency (1 client) | throughput (16+ concurrent) | 12-ticket sanity set |
|---|---|---|---|---|
| **clef-flash** | this server, FP8 | ~60-75 ms | **~24 req/s** | 11/12 per question |
| tev1 (4B) | Ollama | ~210 ms | ~5 req/s | 12/12 |
| nimble (9B) | Ollama | ~240 ms | ~4.6 req/s | 11-12/12 |
| tev1:0.8b | Ollama | ~125 ms | ~10 req/s | poor on noul |

Throughput is GPU-compute-bound: roughly 9,000 input tokens per second, so req/s is about 9000 / tokens-per-request.
The server holds ~11.5 GB of VRAM.

## Requirements

- NVIDIA GPU with 16 GB VRAM (tested on RTX 5080, sm_120). 12 GB may work with `--max-batch-tokens 6000`.
- Python 3.11, [uv](https://docs.astral.sh/uv/), ~18 GB of disk for the weights
- Windows (Git Bash) or Linux. Windows uses `triton-windows`; Linux uses `triton`.
- Optional: [Ollama](https://ollama.com) on `127.0.0.1:11434` for the pass-through models (`tev1`, `nimble`, ...)

## Setup

```bash
scripts/setup.sh            # creates ~/.venvs/clef, installs torch cu130 + deps, downloads Cloudflare/clef-flash
scripts/start_server.sh     # localhost:11436 ; first start takes 1-3 min (load + compile + warmup)
curl -s localhost:11436/s1/health
```

`start_server.sh` is idempotent (prints health if already running), `--stop` kills it, `--lan` binds `0.0.0.0`.
Logs go to `logs/server.log`. Set `S1_PORT` to change the port.

Linux/WSL: edit `setup.sh` to drop `triton-windows` for `triton`, and `start_server.sh` for the venv path
(`~/.venvs/clef/bin/python` is already tried as a fallback).

## Request format

```bash
curl -s localhost:11436/v1/systemone -d '{
  "model": "clef-flash",
  "state": {"ticket": "I was charged twice, refund please"},
  "questions": {
    "team":    {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "technical": "Bugs"}},
    "refund":  {"type": "noul",   "instructions": "Asks for a refund?"},
    "urgency": {"type": "score",  "instructions": "How urgent?", "criteria": ["Routine", "Soon", "Urgent"]}}}'
```

- `state`: any string or JSON
- `noul`: true/false, returns `noul` = P(true)
- `choice`: named options (`criteria` maps option id to description), returns `choice`, `confidence`, `probabilities`
- `score`: ordered options (`criteria` is a list indexed from 0), returns expected `score` (float), `confidence`, `probabilities`
- `instructions` is optional; the question id is used when omitted

Ask every question about one state in **one** request. They are answered in a single forward pass, so extra
questions are nearly free. Images and video are supported by the model but not by this server's fast path.

### Python client

```python
from scripts.s1 import decide, decide_many
r = decide({"ticket": "charged twice"}, {"refund": {"type": "noul", "instructions": "Asks for a refund?"}})
r["answers"]["refund"]["noul"]                      # 0.97
rs = decide_many(states, questions)                  # concurrent; the server batches them
```

CLI: `python scripts/s1.py --jsonl states.jsonl --questions @q.json > answers.jsonl`.
Env `S1_URL` and `S1_MODEL` override the defaults. Any Ollama SDK works by pointing its host at `:11436`.

## Tests and benchmarks

```bash
~/.venvs/clef/Scripts/python.exe -m pytest tests/ -v          # needs the server running
python scripts/loadtest.py --n 256 --concurrency 32           # throughput / p50 / p95
python bench/bench_models.py clef-flash tev1 nimble           # accuracy + latency per model via the endpoint
python bench/bench_quant_sweep.py fp8 compile                 # in-process: FP8 vs nf4, batch-size sweep
python bench/bench_bitsandbytes.py nf4                        # in-process: bitsandbytes nf4 / int8
```

`tests/sanity_data.py` holds the 12-ticket sanity set (intent, spam, urgency) that all of these share.
The test suite checks response shape, validation errors, batched-vs-single agreement, and requires at least
10/12 on each question. Ollama pass-through tests are skipped when Ollama is not running.

## Server flags

| flag | default | notes |
|---|---|---|
| `--host` | 127.0.0.1 | `0.0.0.0` for LAN |
| `--port` | 11436 | |
| `--ollama` | http://127.0.0.1:11434 | upstream for everything that is not clef-flash |
| `--max-batch` | 16 | batch buckets are 1, 2, 4, 8, 16 (32, 64 if raised) |
| `--max-batch-tokens` | 12000 | padded tokens per forward; above ~16k Windows spills VRAM into shared memory |
| `--batch-wait-ms` | 2.0 | how long to wait for more requests before running a batch |
| `--no-compile` | off | skip torch.compile (slower, faster startup) |

## Exposing on a LAN

Start with `--lan`, then on the other machine set `OLLAMA_HOST=http://<server-ip>:11436` or call that URL.
Windows Firewall needs an inbound rule once (admin PowerShell):

```powershell
New-NetFirewallRule -DisplayName "systemone 11436" -Direction Inbound -Protocol TCP -LocalPort 11436 -RemoteAddress LocalSubnet -Action Allow
```

There is no auth. Keep it on the local subnet and do not port-forward it.

## Things learned the hard way

- Ollama's own `clef-flash` GGUF (q8_0) fails on sm_120 with `Clef: non-finite logit` (Ollama 0.35.1). This server avoids that.
- torchao FP8 **per-row** scaling raises "Rowwise scaling is not currently supported" on sm_120. Per-tensor works.
- `torch.compile` must wrap each decoder layer. Compiling the outer model is silently bypassed by Clef's forward.
- Every new (batch, seq-len) shape recompiles for seconds. The server pads to batch buckets and 64-token
  multiples and warms them all at startup. Removing that halves cold throughput.
- bitsandbytes nf4 fits in 8 GB but is ~2x slower; uncompiled FP8 is slower still.
- A big Ollama chat model loaded next to this server will spill to CPU. Stop the server when you need the GPU for an LLM.
- Possible future speedup: NVFP4 (~2x FP8 matmul) needs Meta's MSLK kernels, likely Linux/WSL only.

## License

MIT for the code in this repo. Clef-Flash itself is Apache-2.0 (see the model card).
