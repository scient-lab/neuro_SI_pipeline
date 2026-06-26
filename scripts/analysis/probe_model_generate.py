#!/usr/bin/env python3
"""
probe_model_generate.py — Health-check any HF checkpoint by generating on a prompt
and classifying the output as coherent vs garbage.

Built for the RL degeneration diagnosis (2026-06-26): GRPO rollouts were garbage
(multi-script salad + U+FFFD — see scripts/analysis/diagnose_charset.py), and we
needed to know WHERE it breaks. At RL step 0 the LoRA is ~zero, so the policy ≈ the
merged SFT model. So greedy-generating from the merged model directly splits it:

  * garbage  → the checkpoint ITSELF is broken — bug is upstream of RL (SFT / merge)
  * coherent → the checkpoint is fine — GRPO degrades it (bug in the RL loop:
               KL/beta too weak, LR too hot, or a degenerate reward signal)

Greedy (do_sample=False) is deliberate: it removes sampling randomness, so garbage
means the model's argmax path is broken, not unlucky sampling. Use --sample to
mirror the GRPO rollout (temperature 0.6 / top_p 0.9) once greedy is confirmed clean.

Loads model AND tokenizer from the SAME path (matching rl_training.py / test_rl.py),
applies the tokenizer's own chat template, generates, prints a TRUNCATED completion,
then pipes it through diagnose_charset.py for a verdict + exit code.

Runs in the si_curriculum venv (needs torch + transformers). Heavy imports are lazy,
so `--help` and argument errors work without a GPU.

Usage:
  # the key RL question — is the merged SFT model healthy?
  python scripts/analysis/probe_model_generate.py outputs/<RUN_ID>/sft_checkpoints/checkpoint-*/merged_final_model
  # sanity baseline — the untouched base model should always be clean:
  python scripts/analysis/probe_model_generate.py Qwen/Qwen3-8B
  # mirror the GRPO rollout (sampled, not greedy):
  python scripts/analysis/probe_model_generate.py <path> --sample
  # custom prompt / longer output:
  python scripts/analysis/probe_model_generate.py <path> --prompt "Explain what a neuron is." --max-new-tokens 400

Exit code: passes through diagnose_charset (0 clean / 1 foreign-script drift /
2 decode-corruption), 3 = load-or-generation error, 4 = empty completion.
"""
import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CLASSIFIER = os.path.join(_HERE, "diagnose_charset.py")

# Self-contained probe prompt: a trivial MCQ a healthy model answers easily (B),
# mirroring the RL task shape (think, then a single A-D answer). The exact prompt
# does not matter for a health check — a healthy model is coherent on ANY sane
# prompt; a broken one garbles everything. Override with --prompt for a real item.
_SYSTEM = ("You answer a single-choice multiple-choice question. Think step by step "
           "inside <think>...</think>, then give the final answer as one letter "
           "A, B, C, or D inside <answer>...</answer>.")
_MCQ = ("Question: Which part of a neuron typically RECEIVES incoming signals from "
        "other neurons?\nA) Axon terminal\nB) Dendrite\nC) Myelin sheath\n"
        "D) Node of Ranvier")


def classify(text):
    """Pipe `text` through diagnose_charset.py (-, stdin); return (report, code)."""
    p = subprocess.run([sys.executable, _CLASSIFIER, "-", "--context", "60"],
                       input=text.encode("utf-8"), capture_output=True)
    return p.stdout.decode("utf-8", "replace"), p.returncode


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_path", help="HF checkpoint dir or hub id (model + tokenizer)")
    ap.add_argument("--prompt", default=None,
                    help="raw user prompt; default = a sample MCQ")
    ap.add_argument("--prompt-file", default=None,
                    help="read the user prompt from a file (overrides --prompt; "
                         "avoids shell-quoting a long multi-line prompt)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--sample", action="store_true",
                    help="sample instead of greedy — mirrors GRPO")
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="sampling temperature with --sample (GRPO: 0.6)")
    ap.add_argument("--top-p", type=float, default=0.9,
                    help="nucleus top_p with --sample (GRPO: 0.9)")
    ap.add_argument("--repetition-penalty", type=float, default=None,
                    help="HF repetition_penalty; GRPO rollout uses 1.15. The prime "
                         "suspect for the multilingual-garbage drift over long gens.")
    ap.add_argument("--min-new-tokens", type=int, default=None,
                    help="force at least N new tokens (suppresses early EOS) — mirror "
                         "GRPO's clipped_ratio=1.0 / never-terminating rollouts")
    ap.add_argument("--show", type=int, default=1500,
                    help="chars of completion to print (default 1500)")
    a = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:  # noqa: BLE001
        print(f"IMPORT ERROR ({e}). Run in the si_curriculum venv (torch+transformers).",
              file=sys.stderr)
        return 3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True,
                                            use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            a.model_path, torch_dtype=getattr(torch, a.dtype),
            trust_remote_code=True).to(device)
    except Exception as e:  # noqa: BLE001
        print(f"LOAD ERROR: {e}", file=sys.stderr)
        return 3
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # resolve the user prompt: --prompt-file > --prompt > built-in MCQ
    if a.prompt_file:
        with open(a.prompt_file, encoding="utf-8", errors="replace") as f:
            user_prompt = f.read().strip()
    else:
        user_prompt = a.prompt  # may be None

    messages = ([{"role": "user", "content": user_prompt}] if user_prompt
                else [{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": _MCQ}])
    try:
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
    except Exception:  # noqa: BLE001 — model without a chat template
        text = (user_prompt if user_prompt else f"{_SYSTEM}\n\n{_MCQ}")

    inputs = tok(text, return_tensors="pt").to(device)
    gen = dict(max_new_tokens=a.max_new_tokens,
               pad_token_id=tok.pad_token_id or tok.eos_token_id)
    if a.min_new_tokens is not None:
        gen["min_new_tokens"] = a.min_new_tokens
    if a.repetition_penalty is not None:
        gen["repetition_penalty"] = a.repetition_penalty
    if a.sample:
        gen.update(do_sample=True, temperature=a.temperature, top_p=a.top_p)
    else:
        gen.update(do_sample=False)

    print(f"model: {a.model_path}")
    _mode = (f"sampled (T={a.temperature},p={a.top_p})" if a.sample else "greedy")
    _extra = "".join([
        f" | rep_penalty={a.repetition_penalty}" if a.repetition_penalty else "",
        f" | min_new_tokens={a.min_new_tokens}" if a.min_new_tokens else ""])
    print(f"device: {device} | dtype: {a.dtype} | {_mode} | "
          f"max_new_tokens: {a.max_new_tokens}{_extra}")
    try:
        with torch.no_grad():
            out = model.generate(**inputs, **gen)
    except Exception as e:  # noqa: BLE001
        print(f"GENERATION ERROR: {e}", file=sys.stderr)
        return 3

    completion = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=False)
    if not completion.strip():
        print("=" * 60)
        print("EMPTY COMPLETION — model generated nothing (immediate EOS).")
        print("That is itself a failure signal (a healthy model produces text).")
        return 4

    print("=" * 60)
    print(f"COMPLETION (first {a.show} chars):")
    print(completion[:a.show])
    print("=" * 60)
    report, code = classify(completion)
    print(report, end="")
    return code


if __name__ == "__main__":
    sys.exit(main())
