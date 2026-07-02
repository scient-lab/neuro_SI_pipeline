#!/usr/bin/env python3
"""rl.merge_rl entrypoint (Phase B thin wrapper).

Twin of scripts/phases/rl.sh::step_merge_rl (and of sft_merge_lora). Folds the
GRPO LoRA adapter into the SFT-merged base to produce the final deployable
FULL-weights model, then runs the UNCHANGED 3_si_curriculum/training/merge_lora.py.

Only LoRA-RL (rl.use_lora=true — smoke/pilot single-GPU) yields an adapter that
needs folding; full-FT RL (use_lora=false — the paper) already writes full
safetensors checkpoints, so this is a clean no-op there (exit 0, no merge).

The GRPO adapter was trained ON TOP of the SFT-merged model, so THAT is the merge
base (not the original base). Both the SFT-merged base and the newest RL adapter
are resolved by the newest-by-mtime glob (SFT_MERGED_MODEL / RL_ADAPTER_DIR env
override) — the §5.1 handoff, a later declared-output candidate; kept as-is for
behavioral parity with rl.sh / sft_merge_lora.
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
    return str(v).strip().lower() in ("true", "1", "yes")


# Full-FT RL has no adapter to fold — clean no-op (twin of rl.sh's early return).
if not _truthy(pipeline_config.get_phase_param("rl", "use_lora", False)):
    print("rl.merge_rl: rl.use_lora=false — full-FT RL checkpoints are already "
          "full safetensors; no merge needed.")
    sys.exit(0)

# Merge base = the SFT-merged model the GRPO adapter was trained on.
sft_merged = os.environ.get("SFT_MERGED_MODEL") or ""
if not sft_merged:
    cands = [c for c in glob.glob(os.path.join(OUT, "sft_checkpoints", "checkpoint-*", "merged_final_model"))
             if os.path.isdir(c)]
    sft_merged = max(cands, key=os.path.getmtime) if cands else ""
if not sft_merged or not os.path.isdir(sft_merged):
    sys.exit("rl.merge_rl: SFT-merged base model not found (the GRPO adapter was "
             "trained on it). Run sft first or set SFT_MERGED_MODEL.")

# Newest RL checkpoint (adapter) by mtime — avoids a stale higher-numbered dir.
adapter = os.environ.get("RL_ADAPTER_DIR") or ""
if not adapter:
    cands = [c for c in glob.glob(os.path.join(OUT, "rl_checkpoints", "checkpoint-*"))
             if os.path.isdir(c)]
    adapter = max(cands, key=os.path.getmtime) if cands else ""
if not adapter or not os.path.isdir(adapter):
    sys.exit(f"rl.merge_rl: no checkpoint found in {os.path.join(OUT, 'rl_checkpoints')} "
             "(run train_grpo first)")

rc = subprocess.run(
    [sys.executable, "merge_lora.py",
     "--base_model", sft_merged,
     "--adapter_path", adapter],
    cwd=os.path.join(REPO, "3_si_curriculum", "training"),
).returncode
if rc == 0:
    print(f"Merged RL model (deployable full safetensors): {adapter}/merged_final_model/")
sys.exit(rc)
