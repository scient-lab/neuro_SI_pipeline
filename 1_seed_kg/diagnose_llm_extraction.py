#!/usr/bin/env python3
"""Diagnose graphrag's LLM entity+relationship extraction by replaying ONE
chunk against any OpenAI-compatible vLLM endpoint, then applying the SAME
parser logic graphrag_index.py uses — and reporting exactly which records get
KEPT vs DROPPED.

This script exists because graphrag_index.py's silent-success failure mode
(LLM emits something, parser drops it all, you end up with 0 relationships
and no error in the log) is impossible to debug by reading logs. Here we
capture the raw LLM response and trace it through the parser.

Two modes:
  LOCAL  (DEFAULT) — loads vLLM IN-PROCESS on the local GPU using the EXACT
         same code path as graphrag_index.py and add_llm_relations.py. This is
         what the pipeline phases themselves do. Requires GPU + the graphrag
         venv. Default model_id comes from configs/default.yaml's
         models.extract (override with --model).

  HTTP   (OPT-IN via --endpoint) — POSTs to any OpenAI-compatible vLLM
         endpoint. Useful when running the diagnostic from a laptop without
         GPU, or against a known-good external endpoint for A/B comparison.
         Pass --endpoint and --api-key explicitly; the tool does NOT auto-
         pick VLLM_ENDPOINT_URL from env because that variable is reserved
         for a separate vLLM-serving Pod with a different purpose.

Usage:
  # Default text + endpoint from .env.runpod (most common diagnostic)
  python3 1_seed_kg/diagnose_llm_extraction.py

  # Test a real chunk from your corpus
  python3 1_seed_kg/diagnose_llm_extraction.py --file corpus/neuroscience/source_txt/snippet.txt

  # Spot-check inline text
  python3 1_seed_kg/diagnose_llm_extraction.py --text "Dopamine modulates the basal ganglia."

  # Explicit endpoint + model
  python3 1_seed_kg/diagnose_llm_extraction.py \\
      --endpoint https://abc-8000.proxy.runpod.net \\
      --api-key sk-... \\
      --model Qwen/Qwen3-14B

  # Save full report (raw response + record breakdown) for later analysis
  python3 1_seed_kg/diagnose_llm_extraction.py --out diagnose_extract.json

Exit code:
  0  at least one relationship parsed cleanly
  1  zero relationships parsed
  2  config error (no endpoint/api-key)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import urllib.error
import urllib.request

# Make prompts_kg importable when invoked from anywhere.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from prompts_kg import (  # noqa: E402
    PROMPT_TEMPLATE,
    USER_EXAMPLE,
    ASSISTANT_EXAMPLE,
    USER_PROMPT,
    get_relation_types,
)

# Same delimiters graphrag_index.py uses — DO NOT CHANGE without updating
# the indexer in lockstep.
DELIMS = dict(
    completion_delimiter="<|COMPLETE|>",
    tuple_delimiter="<|>",
    record_delimiter="##",
)

# Fallback text for a 30-second smoke test when no --text / --file given.
# Dense with named entities + relations a small biomed-tuned LLM should hit.
DEFAULT_TEXT = textwrap.dedent("""\
    The hippocampus consolidates long-term memory. Dopamine modulates basal
    ganglia activity through D1 and D2 receptors. NMDA receptors mediate
    synaptic plasticity in CA1 pyramidal neurons. Mutations in the SCN1A
    gene encoding the Nav1.1 sodium channel cause Dravet syndrome, a severe
    form of childhood epilepsy.""")


# ---------------------------------------------------------------------------
# Prompt construction (mirrors graphrag_index.py:175-204 exactly)
# ---------------------------------------------------------------------------
def build_messages(text: str) -> list[dict]:
    relation_list_str = json.dumps(get_relation_types())
    return [
        {"role": "system",    "content": PROMPT_TEMPLATE.format(
            relation_list=relation_list_str, **DELIMS)},
        {"role": "user",      "content": USER_EXAMPLE},
        {"role": "assistant", "content": ASSISTANT_EXAMPLE.format(**DELIMS)},
        {"role": "user",      "content": USER_PROMPT.format(input_text=text)},
    ]


# ---------------------------------------------------------------------------
# HTTP mode (opt-in via --endpoint)
# ---------------------------------------------------------------------------
def call_endpoint(messages, endpoint, api_key, model, max_tokens, temperature, top_p, timeout):
    body = {
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    if model:
        body["model"] = model
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# LOCAL mode — in-process vLLM (DEFAULT, mirrors graphrag_index.py)
# ---------------------------------------------------------------------------
def call_local(messages, model, max_tokens, temperature, top_p,
               max_model_len, gpu_memory_utilization):
    """Load the model in-process and run ONE chat completion. Same code path
    graphrag_index.py:158 + .py:214 uses, so any bug reproduced here is the
    same bug the pipeline phases will hit."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise RuntimeError(
            "vllm not importable — are you in the graphrag venv? "
            "Run: source .venvs/graphrag/bin/activate"
        ) from e

    print(f"  loading model {model!r} (max_model_len={max_model_len}, "
          f"gpu_mem_util={gpu_memory_utilization}) — ~30 sec first time…",
          file=sys.stderr)
    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    sampling = SamplingParams(
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(max_tokens),
    )
    outputs = llm.chat([messages], sampling_params=sampling)
    # Same response extraction graphrag_index.py:215 does.
    text = outputs[0].outputs[0].text
    # Shape it like an OpenAI response so downstream parser code is mode-agnostic.
    return {"choices": [{"message": {"role": "assistant", "content": text}}],
            "_mode": "local"}


# ---------------------------------------------------------------------------
# Resolve default model from configs/default.yaml (LOCAL mode)
# ---------------------------------------------------------------------------
def resolve_default_model() -> str | None:
    """Read models.extract from configs/default.yaml. Falls back to None if
    unavailable (caller will then require --model)."""
    cfg = os.path.join(os.path.dirname(HERE), "configs", "default.yaml")
    if not os.path.exists(cfg):
        return None
    try:
        import yaml
        with open(cfg) as f:
            data = yaml.safe_load(f) or {}
        return ((data.get("models") or {}).get("extract")) or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Parser — mirrors _process_results_directed() in graphrag_index.py
# ---------------------------------------------------------------------------
def parse_records(response_text: str) -> list[dict]:
    """Apply the indexer's parser logic and tag each record's fate.

    The indexer keeps a record iff:
      - first attribute == literal '"entity"'        and n_parts >= 4
      - first attribute == literal '"relationship"'  and n_parts >= 5

    We additionally flag "would-KEEP" cases (same content but different
    quoting) — those reveal a parser-strictness bug rather than an LLM bug.
    """
    rows: list[dict] = []
    records = [r.strip() for r in response_text.split(DELIMS["record_delimiter"])]
    for i, rec in enumerate(records):
        if not rec:
            continue
        parts = rec.split(DELIMS["tuple_delimiter"])
        first = parts[0] if parts else ""
        n = len(parts)
        first_unquoted = first.strip("() \"'\t")
        verdict = "DROP"
        if first == '"entity"' and n >= 4:
            verdict = "KEEP-entity"
        elif first == '"relationship"' and n >= 5:
            verdict = "KEEP-relationship"
        elif first_unquoted == "entity" and n >= 4:
            verdict = "DROP-would-keep-entity"
        elif first_unquoted == "relationship" and n >= 5:
            verdict = "DROP-would-keep-relationship"
        rows.append({
            "index": i,
            "first_token": first[:40],
            "n_parts": n,
            "verdict": verdict,
            "raw": rec[:240],
        })
    return rows


# ---------------------------------------------------------------------------
# .env.runpod loader (no python-dotenv dep — stdlib only)
# ---------------------------------------------------------------------------
def load_env_file(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip("'\"")
            os.environ.setdefault(k, v)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--text",        help="inline text chunk to extract from")
    p.add_argument("--file",        help="path to a .txt chunk file")
    p.add_argument("--model",       help="LOCAL: model id (HF repo or path). "
                                         "HTTP: optional, server picks default. "
                                         "LOCAL default = models.extract from configs/default.yaml")
    # HTTP mode (explicit opt-in)
    p.add_argument("--endpoint",    help="HTTP mode: vLLM URL base (e.g. https://abc-8000.proxy.runpod.net)")
    p.add_argument("--api-key",     help="HTTP mode: bearer token")
    # LOCAL mode tuning (mirrors graphrag_index.py knobs)
    p.add_argument("--max-model-len",
                   type=int, default=8192,
                   help="LOCAL: vLLM max_model_len (default 8192; reduce if VRAM tight)")
    p.add_argument("--gpu-memory-utilization",
                   type=float, default=0.85,
                   help="LOCAL: vLLM gpu_memory_utilization (default 0.85)")
    # Sampling (both modes)
    p.add_argument("--max-tokens",  type=int,   default=4096)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p",       type=float, default=0.95)
    p.add_argument("--timeout",     type=int,   default=180,
                   help="HTTP mode only: request timeout in seconds")
    p.add_argument("--out",         help="save full JSON report (raw response + records + summary)")
    args = p.parse_args()

    # --- mode selection -----------------------------------------------------
    # HTTP mode requires BOTH --endpoint and --api-key (explicit opt-in).
    # We deliberately do NOT auto-pick VLLM_ENDPOINT_URL from env — that var
    # is reserved for a separate vLLM-serving Pod (see vllm_smoke.sh).
    if args.endpoint or args.api_key:
        if not (args.endpoint and args.api_key):
            print("ERROR: HTTP mode needs BOTH --endpoint AND --api-key.",
                  file=sys.stderr)
            print("       To use LOCAL mode (default), drop both flags.",
                  file=sys.stderr)
            return 2
        mode = "http"
    else:
        mode = "local"

    # --- model resolution ---------------------------------------------------
    model = args.model
    if mode == "local" and not model:
        model = resolve_default_model()
        if not model:
            print("ERROR: LOCAL mode needs --model (configs/default.yaml has no "
                  "models.extract entry).", file=sys.stderr)
            return 2

    # --- input text ---------------------------------------------------------
    if args.text:
        text = args.text
        src = "<--text inline>"
    elif args.file:
        with open(args.file) as f:
            text = f.read()
        src = args.file
    else:
        text = DEFAULT_TEXT
        src = "<built-in DEFAULT_TEXT>"

    print(f"=== Input ({len(text):,} chars) from {src} ===")
    print(text[:400] + ("…" if len(text) > 400 else ""))
    print()

    # --- call LLM (mode-dispatched) -----------------------------------------
    messages = build_messages(text)
    if mode == "http":
        print(f"=== HTTP mode: POST {args.endpoint.rstrip('/')}/v1/chat/completions"
              f"  (model={model or 'default'}, max_tokens={args.max_tokens}) ===")
        try:
            resp = call_endpoint(
                messages, args.endpoint, args.api_key, model,
                args.max_tokens, args.temperature, args.top_p, args.timeout,
            )
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code} {e.reason}", file=sys.stderr)
            print(e.read().decode("utf-8", errors="replace")[:1000], file=sys.stderr)
            return 1
        except Exception as e:
            print(f"HTTP call failed: {e}", file=sys.stderr)
            return 1
    else:  # mode == "local"
        print(f"=== LOCAL mode: in-process vLLM"
              f"  (model={model}, max_tokens={args.max_tokens}) ===")
        try:
            resp = call_local(
                messages, model, args.max_tokens, args.temperature, args.top_p,
                args.max_model_len, args.gpu_memory_utilization,
            )
        except RuntimeError as e:
            print(f"LOCAL call failed: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"LOCAL call failed: {type(e).__name__}: {e}", file=sys.stderr)
            return 1

    try:
        response_text = resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        print("unexpected response shape:", json.dumps(resp)[:1000], file=sys.stderr)
        return 1
    print()

    # --- raw response -------------------------------------------------------
    print("=== Raw LLM response ===")
    print(response_text)
    print()

    # --- parse + classify ---------------------------------------------------
    print("=== Record-by-record parse (replays graphrag_index.py logic) ===")
    rows = parse_records(response_text)
    kept_ent          = sum(1 for r in rows if r["verdict"] == "KEEP-entity")
    kept_rel          = sum(1 for r in rows if r["verdict"] == "KEEP-relationship")
    would_keep_ent    = sum(1 for r in rows if r["verdict"] == "DROP-would-keep-entity")
    would_keep_rel    = sum(1 for r in rows if r["verdict"] == "DROP-would-keep-relationship")
    dropped           = sum(1 for r in rows if r["verdict"] == "DROP")

    for r in rows:
        marker = {
            "KEEP-entity":                  "  ✓",
            "KEEP-relationship":            "  ✓",
            "DROP-would-keep-entity":       "  ⚠",
            "DROP-would-keep-relationship": "  ⚠",
            "DROP":                         "  -",
        }.get(r["verdict"], "  ?")
        print(f"{marker} [{r['index']:>3d}] first={r['first_token']:<30.30s} "
              f"n_parts={r['n_parts']:<2d}  -> {r['verdict']}")

    print()
    print("=== Summary ===")
    print(f"  KEEP-entity                  : {kept_ent}")
    print(f"  KEEP-relationship            : {kept_rel}")
    print(f"  DROP-would-keep-entity       : {would_keep_ent}  "
          f"(parser strictness — fix at graphrag_index.py:~333)")
    print(f"  DROP-would-keep-relationship : {would_keep_rel}  "
          f"(parser strictness — same fix)")
    print(f"  DROP                         : {dropped}")
    print()

    # --- verdict ------------------------------------------------------------
    print("=== Verdict ===")
    if kept_rel > 0:
        print(f"  ✓ Extraction works on this chunk — {kept_rel} relationship(s) parsed.")
        print("    If your full pipeline still produces 0 relationships, the bug is")
        print("    downstream (graph merge, dedup, or storage), not in extraction.")
        exit_code = 0
    elif would_keep_rel > 0:
        print(f"  ✗ Parser strictness bug. Model emitted {would_keep_rel} valid")
        print("    relationship records but the parser rejected them due to a")
        print("    quoting/format mismatch.")
        print()
        print("    Fix at 1_seed_kg/graphrag_index.py:~333:")
        print("      before:  if record_attributes[0] == '\"relationship\"' and ...")
        print("      after :  if record_attributes[0].strip('\"') == 'relationship' and ...")
        print("    Apply the same fix to the '\"entity\"' check on the same loop.")
        exit_code = 1
    elif kept_ent > 0:
        print(f"  ✗ Model emitted entities ({kept_ent}) but NO relationship records at all.")
        print("    Prompt/model issue. Try:")
        print("      - increase --max-tokens (LLM may be truncating before relationships)")
        print("      - bigger model (Qwen3-32B / QwQ-Med-3) — relationships need stronger structured output")
        print("      - check that relation_list reaches the system prompt correctly")
        print(f"      - relation_list size: {len(get_relation_types())} types")
        exit_code = 1
    else:
        print("  ✗ Model emitted neither entities nor relationships in the expected format.")
        print("    Possible causes: wrong model, chat-template stripping the few-shot,")
        print("    very small max_tokens, or endpoint returning an error inside the body.")
        exit_code = 1

    # --- optional JSON report ----------------------------------------------
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        report = {
            "mode": mode,
            "endpoint": args.endpoint if mode == "http" else None,
            "model": model,
            "input_chars": len(text),
            "input_text": text,
            "raw_response": response_text,
            "records": rows,
            "summary": {
                "keep_entity": kept_ent,
                "keep_relationship": kept_rel,
                "would_keep_entity": would_keep_ent,
                "would_keep_relationship": would_keep_rel,
                "drop": dropped,
            },
            "exit_code": exit_code,
        }
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print()
        print(f"Report saved: {args.out}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
