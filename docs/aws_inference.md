# Serving the specialized SLM on AWS (inference)

How to deploy the pipeline's final model — a fine-tuned **Qwen3-14B** (dense) —
for inference behind an API, e.g. a working prototype.

> **TL;DR** — Use **Amazon Bedrock Custom Model Import (CMI)**. Our model's
> architecture (`Qwen3ForCausalLM`) is supported, it's serverless + scales to
> zero (you pay per token, nothing while idle), and it supports streaming via
> `InvokeModelWithResponseStream`. Caveat: the **Converse** / **ConverseStream**
> APIs are *not* supported for Qwen3, so you call `InvokeModel` /
> `InvokeModelWithResponseStream` and own the prompt templating yourself.

---

## 1. Why Bedrock CMI (and when not to)

The serving profile for a prototype is bursty and low-QPS, by a small team,
cost-sensitive. That is exactly CMI's sweet spot:

| | Bedrock Custom Model Import | SageMaker LMI (vLLM) endpoint |
|---|---|---|
| Infra to manage | none (serverless) | endpoint + instance |
| Idle cost | **zero** (scales to zero) | pays for the instance 24/7 unless scale-to-zero is configured |
| Billing | per-token, ~5-min active windows | per instance-hour (e.g. `g6e` L40S ≈ a couple $/hr) |
| Cold start | yes — first request after idle loads weights from S3 | warm while the endpoint runs |
| Architecture support | must be on the CMI supported list (Qwen3 dense **is**) | any HF architecture |
| Streaming | `InvokeModelWithResponseStream` (no Converse for Qwen3) | native (vLLM) |

**Pick CMI** for the prototype. **Fall back to SageMaker LMI** (see §6) only if
you need a serving feature CMI doesn't expose, or a much higher sustained QPS
where an always-warm endpoint is cheaper than per-token.

### Why not the other two options

The three options on the table were SageMaker **Script mode**, SageMaker
**Bring-Your-Own-Container (BYOC)**, and Bedrock **CMI**. The first two are
SageMaker endpoint patterns; both were rejected for a prototype:

- **Script mode** — you hand SageMaker an inference entry-point script and it
  runs on a stock framework container (HF/PyTorch DLC). Those containers are
  **not optimized for 14B LLM serving** (no paged-attention / continuous
  batching out of the box), so you'd bolt on vLLM/TGI yourself and effectively
  arrive at LMI/BYOC anyway — while still paying for an always-on endpoint. Most
  effort, weakest serving, no upside here.
- **BYOC (raw)** — you build and maintain the full Docker image (vLLM server,
  model loading, health checks, the SageMaker serving contract). Maximum
  flexibility, maximum maintenance. But the AWS-maintained **LMI** container
  already wraps vLLM, so raw BYOC reinvents it. Only justified if you need a
  serving stack LMI doesn't support. If you must go SageMaker, use **LMI**
  (managed BYOC, §6) — not raw BYOC or script mode.
- **CMI (chosen)** — none of the above ops burden: serverless, scales to zero,
  per-token billing, supports our `Qwen3ForCausalLM` and streaming. The right
  default for a bursty, low-QPS prototype.

In short: script mode and raw BYOC both saddle you with an always-on endpoint
*and* container/serving work, which only pays off at sustained high QPS — not
the prototype profile. CMI removes both burdens; LMI is the escape hatch if a
CMI limitation ever bites.

### Architecture support (the gating fact)

Bedrock CMI supports the Qwen3 architecture for **`Qwen3ForCausalLM`** and
`Qwen3MoeForCausalLM` only. Our model is **dense Qwen3-14B → `Qwen3ForCausalLM`**,
so it imports cleanly. Merging LoRA/GRPO adapters into the base does **not**
change the architecture class.

---

## 2. Prerequisites

- A CMI-enabled region: **us-east-1 (N. Virginia)**, **us-west-2 (Oregon)**, or
  **eu-central-1 (Frankfurt)**. Use the same region for the S3 bucket and the
  import job.
- IAM: a role Bedrock can assume to read the model from S3 (the import wizard
  can create it), plus caller permissions for `bedrock:CreateModelImportJob`,
  `bedrock:GetModelImportJob`, and `bedrock-runtime:InvokeModel*`.
- The merged model artifact in **Hugging Face `safetensors` format** (see §3).

---

## 3. Prepare the model artifact

CMI imports a **fully merged** HF model directory — not a bare LoRA adapter.
The pipeline produces the merged model in the `sft` phase (`merge_lora.py`), and
if you ran GRPO, the `rl` phase output must likewise be merged to full weights
before import.

Final artifact lives under `$OUTPUT_BASE` (default `outputs/`):

```
outputs/sft_checkpoints/checkpoint-<N>/merged_final_model/   # post-SFT merged model
outputs/rl_checkpoints/...                                   # post-GRPO (merge any adapter first)
```

The directory you upload must contain:

```
config.json                # "architectures": ["Qwen3ForCausalLM"]
*.safetensors              # merged weights (+ model.safetensors.index.json if sharded)
tokenizer.json
tokenizer_config.json
special_tokens_map.json
generation_config.json      # optional but recommended (eos/pad ids, defaults)
```

Sanity-check before upload:

```bash
MODEL_DIR=outputs/sft_checkpoints/checkpoint-<N>/merged_final_model
python3 - <<PY
import json; c=json.load(open("$MODEL_DIR/config.json"))
print("architectures:", c.get("architectures"))   # must be ["Qwen3ForCausalLM"]
PY
ls "$MODEL_DIR"/*.safetensors >/dev/null && echo "weights present"
```

Upload to S3 (same region as the import job):

```bash
aws s3 sync "$MODEL_DIR" "s3://<bucket>/models/neuro-slm-qwen3-14b/" --region us-west-2
```

> **Optional but worth it:** quantize the merged weights (FP8 or AWQ) before
> upload. Smaller artifact → faster cold-start load from S3 and lower
> per-invocation cost.

---

## 4. Create the import job

**Console:** Bedrock → *Imported models* → *Import model* → point at the S3 URI,
name it (e.g. `neuro-slm-qwen3-14b`), let it create/select the IAM role, submit.
Import takes minutes-to-tens-of-minutes depending on size.

**CLI:**

```bash
aws bedrock create-model-import-job \
  --region us-west-2 \
  --job-name neuro-slm-import-$(date -u +%Y%m%d-%H%M%S) \
  --imported-model-name neuro-slm-qwen3-14b \
  --role-arn arn:aws:iam::<acct>:role/<bedrock-cmi-role> \
  --model-data-source '{"s3DataSource":{"s3Uri":"s3://<bucket>/models/neuro-slm-qwen3-14b/"}}'

# poll until COMPLETED, then grab the imported model ARN:
aws bedrock get-model-import-job --region us-west-2 --job-identifier <jobArn> \
  --query '{status:status, model:importedModelArn}'
```

Record the returned **`importedModelArn`** — that's the `modelId` you invoke.

---

## 5. Invoke

Qwen3 on CMI does **not** support Converse, so use `InvokeModel`
(non-streaming) or `InvokeModelWithResponseStream` (streaming). You must apply
the **Qwen3 chat template** yourself and parse the raw completion. Reuse the
exact templating the pipeline's eval uses so prototype behavior matches
training/eval.

### Prompt templating

```python
# Qwen3 chat format. Keep this identical to what 3_si_curriculum eval uses.
def build_prompt(system: str, user: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
```

### Non-streaming

```python
import json, boto3

rt = boto3.client("bedrock-runtime", region_name="us-west-2")
MODEL_ID = "arn:aws:bedrock:us-west-2:<acct>:imported-model/<id>"

body = {
    "prompt": build_prompt("You are a neuroscience expert.", "What is long-term potentiation?"),
    "max_tokens": 512,
    "temperature": 0.6,
    "top_p": 0.95,
}
resp = rt.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
out = json.loads(resp["body"].read())
print(out)   # field names follow the imported model's output schema
```

### Streaming (token-by-token)

```python
resp = rt.invoke_model_with_response_stream(modelId=MODEL_ID, body=json.dumps(body))
for event in resp["body"]:
    chunk = json.loads(event["chunk"]["bytes"])
    # accumulate / forward each chunk to your SSE or WebSocket
    print(chunk.get("generation", ""), end="", flush=True)
```

> Wire the stream straight to your prototype's SSE/WebSocket for token-by-token
> UX. The first token after idle is delayed by the cold-start load; subsequent
> tokens stream normally.

### Reasoning traces

If the fine-tune emits `<think>…</think>` blocks, decide in your response
handler whether to suppress them or stream them to the UI.

---

## 6. Fallback: SageMaker LMI (vLLM) endpoint

Only if CMI is insufficient. Use the AWS-maintained **LMI (Large Model
Inference)** container (DJL + vLLM) — *not* a hand-built BYOC image or plain
script mode. It serves any HF architecture (incl. Qwen3) and matches the vLLM
stack the pipeline already uses, so inference behavior is consistent with
training-time eval. Enable **managed scale-to-zero** to avoid idle cost for a
prototype. Trade-off: an always-/often-warm endpoint instead of pure per-token.

---

## Operational notes

- **Cold start**: expect a load delay on the first request after the model
  scales from zero. Acceptable for a prototype; for latency-sensitive demos,
  keep it warm with a periodic ping or use Provisioned Throughput.
- **Cost**: per-token in ~5-minute active windows; nothing while idle. Quantize
  to shrink cold-start and cost.
- **Consistency**: keep the chat template + sampling params (`temperature`,
  `top_p`) aligned with `configs/default.yaml` and the eval harness so prototype
  output matches what we measured offline.
- **Region**: bucket, import job, and runtime calls must be in the same
  CMI-enabled region.
