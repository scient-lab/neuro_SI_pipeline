#!/usr/bin/env python3
"""sft.merge_lora entrypoint (Phase B thin wrapper).

Twin of scripts/phases/sft.sh::step_merge_lora. Picks the NEWEST sft checkpoint
by mtime (ADAPTER_DIR env overrides), then runs the UNCHANGED merge_lora.py to
fold the LoRA adapter into full-precision base weights.

NOTE: the newest-by-mtime glob is exactly the artifact handoff flagged in
DATA_DRIVEN_PIPELINE_EXECUTOR_PLAN §5.1 — a candidate to replace with a declared
manifest output (train_lora records its checkpoint; this reads it by id) rather
than scanning the filesystem. Kept as-is here for behavioral parity.
"""
import glob
import os
import subprocess
import sys

REPO = os.environ["REPO_ROOT"]
sys.path.insert(0, REPO)
import pipeline_config  # noqa: E402

OUT = os.environ["OUTPUT_BASE"]

base_model = pipeline_config.get_model_id("base_sft", "")
if not base_model:
    sys.exit("sft.merge_lora: models.base_sft not set (configs/default.yaml)")

ckpt_dir = os.path.join(OUT, "sft_checkpoints")
adapter = os.environ.get("ADAPTER_DIR")
if not adapter:
    cands = [c for c in glob.glob(os.path.join(ckpt_dir, "checkpoint-*")) if os.path.isdir(c)]
    adapter = max(cands, key=os.path.getmtime) if cands else ""
if not adapter or not os.path.isdir(adapter):
    sys.exit(f"sft.merge_lora: no checkpoint found in {ckpt_dir}")

rc = subprocess.run(
    [sys.executable, "merge_lora.py",
     "--base_model", base_model,
     "--adapter_path", adapter],
    cwd=os.path.join(REPO, "3_si_curriculum", "training"),
).returncode
if rc == 0:
    print(f"Merged SFT model: {adapter}/merged_final_model/")
sys.exit(rc)
