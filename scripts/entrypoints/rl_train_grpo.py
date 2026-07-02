#!/usr/bin/env python3
"""rl.train_grpo entrypoint (Phase B thin wrapper).

Twin of scripts/phases/rl.sh::step_train_grpo. Resolves the SFT-merged base model
(the GRPO base) + rl.* knobs via pipeline_config, applies the single-GPU deepspeed
guard, then runs the UNCHANGED 3_si_curriculum/RL/rl_training.py.

Hard-won fixes preserved verbatim (plan §7.4):
  * bug #8  — rl_training.py reads config.sft_checkpoint_path (NOT model_name);
              pass the merged SFT model via --sft_checkpoint_path.
  * bug #14 — --deepspeed is YAML-gated (rl.use_deepspeed). The default config is
              ZeRO-3 + multi-GPU; single-GPU falls through to mpi4py (not
              installed). Fail LOUD if use_deepspeed=true on <=1 GPU rather than
              silently flipping the regime (paper identity = full-FT ZeRO-3).
  * bug #17 — W&B is governed by the inherited WANDB_MODE (pipeline.sh::
              wandb_autodisable); no per-step guard (kept in sync with sft).

The SFT-merged base is the §5.1 cross-phase handoff (sft.merge_lora → rl base);
resolved here by the newest-by-mtime glob (SFT_MERGED_MODEL env overrides), a
later declared-output candidate — same selector as rl.sh / sft_merge_lora.
"""
import glob
import os
import subprocess
import sys

REPO = os.environ["REPO_ROOT"]
sys.path.insert(0, REPO)
import pipeline_config  # noqa: E402

OUT = os.environ["OUTPUT_BASE"]


def _truthy(v) -> bool:
    """Mirror rl.sh's `== true|True|1` check (YAML may parse a bool, str, or int)."""
    return str(v).strip().lower() in ("true", "1", "yes")


def _gpu_count() -> int:
    """Count GPUs via `nvidia-smi -L` (lines starting 'GPU '); 0 if absent —
    mirrors rl.sh's `nvidia-smi -L | grep -c '^GPU ' || true`."""
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return 0
    return sum(1 for line in out.stdout.splitlines() if line.startswith("GPU "))


# The GRPO base = the SFT-merged model (adapter is trained ON TOP of it).
sft_merged = os.environ.get("SFT_MERGED_MODEL") or ""
if not sft_merged:
    cands = glob.glob(os.path.join(OUT, "sft_checkpoints", "checkpoint-*", "merged_final_model"))
    cands = [c for c in cands if os.path.isdir(c)]
    sft_merged = max(cands, key=os.path.getmtime) if cands else ""
if not sft_merged or not os.path.isdir(sft_merged):
    sys.exit("rl.train_grpo: no merged SFT model found. Run sft phase first or set SFT_MERGED_MODEL.")

rl_dataset = os.path.join(OUT, "rl_dataset")
ckpt_dir = os.path.join(OUT, "rl_checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)

domain = os.environ.get("SI_DOMAIN", "neuroscience")
wandb_project = os.environ.get("WANDB_PROJECT") or f"{domain}_rl_kg"

# bug #14: deepspeed is opt-in via rl.use_deepspeed; ZeRO-3 needs >1 GPU.
ds_args = []
if _truthy(pipeline_config.get_phase_param("rl", "use_deepspeed", True)):
    ngpu = _gpu_count()
    if ngpu <= 1:
        sys.exit(
            f"rl.train_grpo: rl.use_deepspeed=true (ZeRO-3) but only {ngpu} GPU detected.\n"
            "  ZeRO-3 requires multi-GPU. Options:\n"
            "    - multi-GPU: launch with >=2 GPUs (SLURM/Della).\n"
            "    - single-GPU demo: override rl.use_deepspeed=false AND rl.use_lora=true\n"
            "      (full-FT 14B will NOT fit one GPU; LoRA is required)."
        )
    ds_cfg = os.environ.get("DEEPSPEED_CFG") or os.path.join(
        REPO, "3_si_curriculum", "RL", "deepspeed_config.json")
    ds_args = ["--deepspeed", ds_cfg]
    print(f"rl.train_grpo: deepspeed ENABLED ({ngpu} GPUs) — config: {ds_cfg}")
else:
    print("rl.train_grpo: single-GPU mode (rl.use_deepspeed=false)")

sys.exit(subprocess.run(
    [sys.executable, "rl_training.py",
     "--sft_checkpoint_path", sft_merged,   # bug #8: field is sft_checkpoint_path
     "--dataset_path", rl_dataset,
     "--output_dir", ckpt_dir,
     *ds_args,
     "--wandb_project", wandb_project],
    cwd=os.path.join(REPO, "3_si_curriculum", "RL"),
).returncode)
