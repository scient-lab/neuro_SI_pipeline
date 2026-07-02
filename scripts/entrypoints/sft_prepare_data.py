#!/usr/bin/env python3
"""sft.prepare_data entrypoint (Phase B thin wrapper).

Twin of scripts/phases/sft.sh::step_prepare_data. Self-resolving: reads the env
contract (REPO_ROOT, OUTPUT_BASE, HF_HOME/HOME) + pipeline_config
(models.base_sft), resolves paths, then invokes the UNCHANGED
3_si_curriculum/training/data_prep.py under this (si_curriculum) venv.

data_prep.py emits a DatasetDict({"train","test"}) with a `text` column via
apply_chat_template; it does NOT tokenize (TRL SFTTrainer does, inside trainer.py
from sft.block_size) — so there is deliberately no --max_length flag here.
"""
import os
import subprocess
import sys

REPO = os.environ["REPO_ROOT"]
sys.path.insert(0, REPO)
import pipeline_config  # noqa: E402  (repo root on sys.path above)

OUT = os.environ["OUTPUT_BASE"]

base_model = pipeline_config.get_model_id("base_sft", "")
if not base_model:
    sys.exit("sft.prepare_data: models.base_sft not set (configs/default.yaml)")

verified = os.path.join(OUT, "curriculum_verified", "curriculum_verified.json")
# Output path: prefer $STEP_OUTPUT (injected by run_phase.py from the declared
# `output` in pipeline_execution.yaml — single source); fall back to the
# canonical location when run standalone (no runner).
ds_dir = os.environ.get("STEP_OUTPUT") or os.path.join(OUT, "sft_dataset")
os.makedirs(ds_dir, exist_ok=True)
cache = os.environ.get("HF_HOME") or os.path.join(os.environ["HOME"], ".cache", "huggingface")

sys.exit(subprocess.run(
    [sys.executable, "data_prep.py",
     "--input_file", verified,
     "--output_path", ds_dir,
     "--model_name", base_model,
     "--cache_dir", cache],
    cwd=os.path.join(REPO, "3_si_curriculum", "training"),
).returncode)
