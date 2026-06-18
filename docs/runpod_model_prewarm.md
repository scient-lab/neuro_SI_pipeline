# RunPod: pre-warming QwQ-Med-3 on a network volume

How to deploy QwQ-Med-3 on a RunPod **Serverless** vLLM endpoint without the
cold-start download loop. The short version: **don't download the model during a
serverless cold start — pre-download it onto a network volume with a Pod first**,
and ideally **quantize it** so it's ~18 GB instead of 131 GB.

Related: [inference_deployment.md](inference_deployment.md), [aws_inference.md](aws_inference.md).

---

## The problem this solves

QwQ-Med-3 (`yuvalkansal/QwQ-Med-3`) is **~131 GB on disk** — it's saved in
**fp32** (32B params × 4 bytes). If you point a serverless endpoint at the HF repo
and let it download on first start, you get a **restart loop**:

```
model pending download
initializing model files
model pending download   ← back to start: the worker was recycled mid-download
```

Why it loops:
- **131 GB is too big to finish inside a serverless worker's startup window** —
  the worker is recycled (or the throttled H100 is preempted) before the download
  completes, so it restarts from zero.
- Attaching a **network volume pins the endpoint to one datacenter**, and if that
  DC is short on H100s the worker sits **throttled**, making it worse.

Serverless cold start is for fast warm starts, not multi-tens-of-GB pulls.
**Download once, on a Pod; serve from the volume.**

---

## VRAM / dtype facts (read first)

- 131 GB is the **disk** size (fp32). What lands in **VRAM** depends on load dtype:
  | Load dtype | VRAM (weights) | Fits 80 GB? |
  |---|---|---|
  | fp32 | ~128 GB | ❌ (needs 2×80 GB) |
  | **bf16** | ~64 GB | ✅ (~16 GB left for KV cache) |
  | **AWQ 4-bit** | ~16–18 GB | ✅ huge headroom; also fits 48 GB |
- On an 80 GB GPU you **must** load as bf16 (`DTYPE=bfloat16`) or it OOMs.
- **Recommended:** quantize to AWQ 4-bit — smaller download, faster load, far less
  sensitive to GPU scarcity/throttling.

---

## Step 1 — Network volume

Console → **Storage → Network Volumes → New**:
- **Datacenter:** pick one with good 80 GB GPU availability (H100/A100). ⚠️ the
  endpoint will be **locked to this DC** — choose carefully.
- **Size:** **250 GB** (holds the 131 GB fp32 + HF cache; or 100 GB if you only
  keep an AWQ copy).

---

## Step 2 — Pre-download with a Pod (runs to completion, no timeout)

A Pod has no serverless startup-timeout/preemption, so the download actually
finishes (and resumes on retry).

1. **Deploy a Pod** in the **same datacenter** as the volume (any cheap GPU, or a
   CPU pod is fine for downloading). Attach the **network volume** — on Pods it
   mounts at **`/workspace`**.
2. In the Pod terminal:
   ```bash
   pip install -U "huggingface_hub[cli]"
   export HF_HOME=/workspace/hf          # the network volume
   huggingface-cli download yuvalkansal/QwQ-Med-3
   # writes /workspace/hf/hub/models--yuvalkansal--QwQ-Med-3  (~131 GB, resumable)
   ```
3. **Terminate the Pod** when done — the volume persists.

> Path note: `/workspace/hf` (Pod) and `/runpod-volume/hf` (serverless) are the
> **same physical volume**, so the HF cache populated here is exactly what the
> endpoint reads.

---

## Step 3 (recommended) — Quantize to AWQ 4-bit while the Pod is up

Cuts 131 GB → ~18 GB, loads in seconds, fits any 80 GB (even 48 GB) GPU, and
dodges throttling. On a **GPU** Pod with the volume attached:

```bash
pip install -U autoawq transformers accelerate
python - <<'PY'
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
src = "/workspace/hf/hub/models--yuvalkansal--QwQ-Med-3/snapshots"  # or the repo id
import glob, os
src = glob.glob(os.path.join(src, "*"))[0]                          # resolve snapshot dir
dst = "/workspace/models/QwQ-Med-3-awq"
tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
model = AutoAWQForCausalLM.from_pretrained(src, safetensors=True, device_map="auto")
model.quantize(tok, quant_config={"zero_point": True, "q_group_size": 128,
                                  "w_bit": 4, "version": "GEMM"})
model.save_quantized(dst); tok.save_pretrained(dst)
print("saved AWQ to", dst)
PY
```
Result: `/workspace/models/QwQ-Med-3-awq` (~18 GB) on the volume. (Optionally push
to your own HF repo with `huggingface-cli upload`.)

---

## Step 4 — Configure the serverless endpoint

Manage → Edit Endpoint → **Environment Variables**:

**If serving fp32 weights as bf16 (no quantization):**
```
MODEL_NAME=yuvalkansal/QwQ-Med-3
HF_HOME=/runpod-volume/hf
DTYPE=bfloat16                 # REQUIRED — fp32 OOMs on 80 GB
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.95
HF_HUB_OFFLINE=1              # load from volume, never re-download
```

**If serving the AWQ 4-bit copy (recommended):**
```
MODEL_NAME=/runpod-volume/models/QwQ-Med-3-awq
QUANTIZATION=awq
DTYPE=float16
MAX_MODEL_LEN=16384           # more KV headroom — weights are only ~18 GB
GPU_MEMORY_UTILIZATION=0.95
HF_HUB_OFFLINE=1
```

**GPU selection:** allow **multiple 80 GB types** (H100 SXM, H100 PCIe, A100 80 GB)
in the volume's DC — more acceptable GPUs = far less throttling. For the AWQ copy,
even 48 GB cards work.

**Workers:** Max Workers = 1 for testing. Don't bump it to "fix" throttling — that's
a capacity problem, not a worker-count problem.

---

## Step 5 — Verify

The endpoint key must own the endpoint (team-scoped if Amit created it under the team):
```bash
curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
  https://api.runpod.ai/graphql \
  -d '{"query":"query { myself { endpoints { id name } } }"}'
```
Then a single-question test:
```bash
echo "What is long-term potentiation, and why is it important for memory?" > /tmp/one_q.txt
python3 scripts/run_inference_runpod.py --questions-file /tmp/one_q.txt \
  --out /tmp/one_answer.jsonl --max-tokens 256 --poll-timeout 1200
cat /tmp/one_answer.jsonl
```
Logs should now show the worker **loading from disk** (no "downloading model files")
and reach **Running** → job COMPLETED.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `model pending download` loops | 131 GB download recycled mid-cold-start | Pre-download via Pod (Step 2); set `HF_HUB_OFFLINE=1` |
| Worker **Throttled** | No 80 GB GPU free in the volume's locked DC | Allow more GPU types; or recreate volume in a DC with capacity |
| OOM on load | Loading fp32 on 80 GB | Set `DTYPE=bfloat16` (or use the AWQ copy) |
| `HTTP_403` from the script | API key lacks endpoint permission | Use a key from the team that owns the endpoint (see verify call) |
| Cold start still slow | Re-downloading each time | Confirm `HF_HOME=/runpod-volume/...` and `HF_HUB_OFFLINE=1` |

---

## TL;DR

1. Network volume (250 GB) in a DC with 80 GB GPUs.
2. **Pod** + `huggingface-cli download` → model on the volume (one time, runs to completion).
3. **AWQ 4-bit** it (~18 GB) — strongly recommended; kills the size + throttling problems.
4. Endpoint env: `MODEL_NAME` (volume path), `DTYPE=bfloat16` or `QUANTIZATION=awq`,
   `HF_HOME=/runpod-volume/hf`, `HF_HUB_OFFLINE=1`, allow multiple 80 GB GPUs.
5. Never download 131 GB during a serverless cold start.

*RunPod console/UI specifics are a ~Jan 2026 snapshot — verify env-var names against
the current vLLM worker (worker-vllm) README. Last updated: 2026-06-18.*
