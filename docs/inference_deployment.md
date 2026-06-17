# Inference deployment — serving the specialized SLM

How to serve the pipeline's final model — a fine-tuned **Qwen3-14B** (dense) —
behind a streaming API, e.g. a working prototype.

Two serverless options are compared here:

1. **RunPod Serverless** — your vLLM container on RunPod, per-GPU-second billing.
2. **Amazon Bedrock Custom Model Import (CMI)** — managed, per-CMU billing.

> **TL;DR for the prototype:** use **RunPod Serverless** — you already train on
> RunPod, it runs the *same vLLM stack* with no architecture restrictions and an
> OpenAI-compatible streaming API. Switch to **Bedrock CMI** when the product
> needs to live inside AWS (IAM/VPC governance, AWS-native integration,
> zero-container ops).

> A heavier AWS fallback — a **SageMaker LMI (vLLM) endpoint** for
> sustained/high-QPS or models CMI can't serve — is out of scope here; see
> [aws_inference.md](aws_inference.md).

---

## 1. Decision matrix

| | **RunPod Serverless** | **Bedrock CMI** |
|---|---|---|
| Billing | per **GPU-second** (worker active) | per **CMU** (~5-min warm windows) |
| Idle cost | zero (scale to zero) | zero (scale to zero) |
| Architecture limits | **none** (your vLLM) | gated list; Qwen3 dense OK, **no Converse** |
| Serving stack | **your vLLM** (OpenAI API) | managed; you own templating |
| Streaming | yes (OpenAI SSE) | yes (`InvokeModelWithResponseStream`) |
| Container ops | supply worker (vLLM template = turnkey) | **none** |
| Cold start | container + weight load (you control source) | managed-storage load |
| Ecosystem | same vendor as training | native AWS |
| GPU choice | you pick (L40S/A100/H100) | abstracted |

---

## 1b. Cost & cold-start estimates

> Prices fluctuate — **verify on the vendor pages before quoting**. Figures below
> are mid-2026 ballparks for our model (14B, L40S-class) and round-number
> assumptions: a typical answer ≈ 512 output tokens ≈ **~15 s of GPU compute**
> on an L40S (~30–45 tok/s).

### Billing formulas

- **Bedrock CMI** = `CMUs × $0.0785/min`, billed in **5-min windows**. CMUs are
  set **at import** based on size/context (anchors: Llama 8B-128K = **2 CMUs**,
  70B-128K = **8 CMUs**). Estimates below: **14B ≈ 2–3 CMUs**, **32B ≈ 4–6 CMUs**
  — *confirm at import* (the console shows the exact count + real-time cost).
- **RunPod Serverless** = `per-GPU-second × seconds active`. L40S flex
  ≈ $0.00053/s (~$1.90/hr); A100-80GB ≈ $0.00076/s; H100-80GB ≈ $0.00116/s.

### Worked cost — model "up" for a continuous 10-minute window

| Model | Serving HW | **RunPod (10 min)** | **Bedrock CMI (10 min)** — est. CMUs |
|---|---|---|---|
| **14B** | L40S 48 GB | **~$0.32** | **~$1.6** (2 CMU) – **~$2.4** (3 CMU) |
| **32B** | A100/H100 80 GB | **~$0.46 – $0.70** | **~$3.1** (4 CMU) – **~$4.7** (6 CMU) |

→ **For the same active wall-time, RunPod is ~5–7× cheaper.** CMI's premium buys
scale-to-zero + zero-ops, not per-minute price. (32B bf16 ~64 GB won't fit one
L40S — serve on 80 GB-class, or INT8-quantize to ~32 GB to fit an L40S.)

### Cold start (scale-from-zero → first token)

| Model | **RunPod** | **Bedrock CMI** |
|---|---|---|
| **14B** | <2 s (FlashBoot warm cache) → ~30 s cold | ~15–60 s |
| **32B** | ~10–60 s (larger load) | ~30–90 s+ |

Bedrock loads from its **managed storage** (your S3 is only the one-time import
source), so its only cold-start levers are model size / Provisioned Throughput.
RunPod's source is yours to control (network volume / baked image / FlashBoot),
so it's the most tunable — and best case the fastest.

> **The CMI numbers are estimates.** AWS publishes only "tens of seconds" for
> cold start and decides CMUs at import. To remove the guesswork, do **one
> throwaway import** of the 14B (and 32B): the console shows the exact CMU count
> + cost instantly, and you can time a cold invocation. A few dollars of testing
> beats any table here.

### Cost by traffic shape (why RunPod wins for a prototype)

- **Sparse / bursty** (occasional single queries): **RunPod wins big** —
  per-second billing ≈ ~$0.01 per ~15 s request. CMI's **5-min minimum** makes
  each *isolated* query cost a full window (~$0.8–2.4 for a 14B), so 1,000
  scattered requests ≈ **$10–15 on RunPod vs hundreds–$1,000+ on CMI**.
- **Clustered** (many requests within the same 5-min windows): CMI amortizes and
  becomes comparable.
- **Sustained high QPS**: RunPod **active** workers (always-on, discounted rate)
  become cheapest per request. (An AWS-native alternative at this scale is a
  SageMaker LMI endpoint — see [aws_inference.md](aws_inference.md).)

Net: for an intermittent prototype, **RunPod Serverless is both the cheapest and
fastest to first token**. Quantizing (FP8/AWQ) cuts cold-start and per-request
compute everywhere.

## 2. Prepare the model artifact (shared by both options)

Both serve a **fully merged** Hugging Face model — not a bare LoRA adapter.
The `sft` phase merges LoRA via `merge_lora.py`; if you ran GRPO, merge the `rl`
output to full weights too.

Final artifacts under `$OUTPUT_BASE` (default `outputs/`):

```
outputs/sft_checkpoints/checkpoint-<N>/merged_final_model/   # post-SFT merged
outputs/rl_checkpoints/...                                   # post-GRPO (merge adapter first)
```

The directory must contain:

```
config.json                # "architectures": ["Qwen3ForCausalLM"]
*.safetensors              # merged weights (+ index.json if sharded)
tokenizer.json
tokenizer_config.json
special_tokens_map.json
generation_config.json      # recommended (eos/pad ids, sampling defaults)
```

Sanity-check:

```bash
MODEL_DIR=outputs/sft_checkpoints/checkpoint-<N>/merged_final_model
python3 - <<PY
import json; c=json.load(open("$MODEL_DIR/config.json"))
print("architectures:", c.get("architectures"))   # ["Qwen3ForCausalLM"]
PY
ls "$MODEL_DIR"/*.safetensors >/dev/null && echo "weights present"
```

> **Optional:** quantize to FP8/AWQ before publishing — smaller artifact, faster
> cold-start load, lower per-request cost. Validate accuracy on the eval set
> first (4-bit can dent reasoning-heavy QA).

Publish it where the serving option reads from:
- RunPod → a **HF private repo**, a **RunPod network volume**, or baked into the
  worker image.
- Bedrock → an **S3** prefix.

---

## 3. Option A — RunPod Serverless (recommended for the prototype)

You already operate on RunPod, so this is one vendor and the *same vLLM* used in
training-time eval → no behavior drift, no architecture gate.

### Create the endpoint

Use RunPod's prebuilt **vLLM worker** (`runpod/worker-vllm`) — no custom image
needed for a standard Qwen3 model:

1. RunPod console → **Serverless → New Endpoint → vLLM**.
2. Point it at the model:
   - private HF repo `your-org/neuro-slm-qwen3-14b` + `HF_TOKEN`, **or**
   - a **network volume** holding the merged model dir (avoids re-downloading on
     every cold start — recommended for a 14B), **or**
   - bake weights into a derived image for the fastest cold start.
3. GPU: **L40S (48 GB)** is plenty for a 14B; set `MAX_MODEL_LEN` to your context.
4. Scaling: min workers `0` (scale to zero) or `1` **active worker** to kill cold
   starts if the demo must be instant. Set an idle timeout.

### Invoke (OpenAI-compatible)

The worker exposes an OpenAI-compatible API at
`https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1`, so any OpenAI SDK works and
**vLLM applies the Qwen3 chat template for you**:

```python
from openai import OpenAI
client = OpenAI(
    base_url="https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1",
    api_key="<RUNPOD_API_KEY>",
)
stream = client.chat.completions.create(
    model="neuro-slm-qwen3-14b",
    messages=[
        {"role": "system", "content": "You are a neuroscience expert."},
        {"role": "user", "content": "What is long-term potentiation?"},
    ],
    temperature=0.6, top_p=0.95, max_tokens=512,
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Pipe the deltas straight to your prototype's SSE/WebSocket.

---

## 4. Option B — Bedrock Custom Model Import (when the product is AWS-native)

Serverless, fully managed, per-token, **zero container ops**. Best when the
prototype/consumers live in AWS and you want IAM/VPC/CloudWatch governance.

**Architecture gate:** CMI supports Qwen3 only for `Qwen3ForCausalLM` and
`Qwen3MoeForCausalLM`. Our dense Qwen3-14B → `Qwen3ForCausalLM` ✓. **Converse /
ConverseStream are NOT supported for Qwen3**, so you call `InvokeModel` /
`InvokeModelWithResponseStream` and own the chat templating.

### Prereqs
- CMI region: **us-east-1**, **us-west-2**, or **eu-central-1** (bucket + job +
  runtime calls all same region).
- IAM role Bedrock can assume to read the S3 model; caller perms for
  `bedrock:CreateModelImportJob`, `GetModelImportJob`, `bedrock-runtime:InvokeModel*`.

### Upload + import

```bash
aws s3 sync "$MODEL_DIR" "s3://<bucket>/models/neuro-slm-qwen3-14b/" --region us-west-2

aws bedrock create-model-import-job --region us-west-2 \
  --job-name neuro-slm-import-$(date -u +%Y%m%d-%H%M%S) \
  --imported-model-name neuro-slm-qwen3-14b \
  --role-arn arn:aws:iam::<acct>:role/<bedrock-cmi-role> \
  --model-data-source '{"s3DataSource":{"s3Uri":"s3://<bucket>/models/neuro-slm-qwen3-14b/"}}'

aws bedrock get-model-import-job --region us-west-2 --job-identifier <jobArn> \
  --query '{status:status, model:importedModelArn}'   # poll until COMPLETED
```

Record the **`importedModelArn`** — that's your `modelId`.

### Invoke (you apply the Qwen3 template)

```python
import json, boto3
rt = boto3.client("bedrock-runtime", region_name="us-west-2")
MODEL_ID = "arn:aws:bedrock:us-west-2:<acct>:imported-model/<id>"

def build_prompt(system, user):   # keep identical to 3_si_curriculum eval templating
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")

body = {"prompt": build_prompt("You are a neuroscience expert.",
                               "What is long-term potentiation?"),
        "max_tokens": 512, "temperature": 0.6, "top_p": 0.95}

# streaming
resp = rt.invoke_model_with_response_stream(modelId=MODEL_ID, body=json.dumps(body))
for event in resp["body"]:
    chunk = json.loads(event["chunk"]["bytes"])
    print(chunk.get("generation", ""), end="", flush=True)
```

If the fine-tune emits `<think>…</think>`, decide whether to stream or strip it.

---

## 5. Cross-cutting operational notes

- **Cold start** on scale-from-zero: RunPod → use a network volume or baked
  image, or keep 1 active worker; Bedrock → loads from its **managed storage**
  (your S3 is only the one-time import source, not the hot path), so the only
  levers are model size / Provisioned Throughput.
- **Cost shape**: RunPod = GPU-seconds (cheaper per substantial request and for
  continuous use); Bedrock = per-CMU 5-min windows (cheaper only for sparse,
  clustered traffic where scale-to-zero + zero-ops outweigh the premium).
- **Consistency**: keep the chat template + sampling (`temperature`, `top_p`)
  aligned with `configs/default.yaml` and the eval harness so prototype output
  matches what we measured offline. RunPod (vLLM) applies the template for you;
  Bedrock CMI does not — replicate it exactly.
- **Quantization**: FP8/AWQ shrinks footprint and cold start everywhere; always
  re-measure accuracy on the eval set before shipping.
