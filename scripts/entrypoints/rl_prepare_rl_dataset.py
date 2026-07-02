#!/usr/bin/env python3
"""rl.prepare_rl_dataset entrypoint (Phase B thin wrapper).

Twin of scripts/phases/rl.sh::step_prepare_rl_dataset. Self-resolving: reads the
env contract (REPO_ROOT, OUTPUT_BASE) + the verified curriculum, then invokes the
UNCHANGED 3_si_curriculum/RL/data_prep.py under this (si_curriculum) venv.

RL/data_prep.py is ENV-VAR-DRIVEN (INPUT_PATH / OUTPUT_PATH), not argparse — so
those are set on the child env rather than passed as flags.

Audit bug #10 (preserved): data_prep.py's "rl" mode only SLICES the verified
curriculum into a DatasetDict — it no longer chains into preprocess_grpo_dataset,
leaving `question_and_explanation` intact. rl_training.py:602 preprocesses once,
so the chain runs cleanly (previously both passes preprocessed → KeyError).
"""
import os
import subprocess
import sys

REPO = os.environ["REPO_ROOT"]
OUT = os.environ["OUTPUT_BASE"]

verified = os.path.join(OUT, "curriculum_verified", "curriculum_verified.json")
# Output path: prefer $STEP_OUTPUT (injected by run_phase.py from the declared
# `output` in pipeline_execution.yaml — single source); fall back to the
# canonical location when run standalone (no runner). Twin of sft_prepare_data.
ds_dir = os.environ.get("STEP_OUTPUT") or os.path.join(OUT, "rl_dataset")
os.makedirs(ds_dir, exist_ok=True)

env = os.environ.copy()
env["INPUT_PATH"] = verified
env["OUTPUT_PATH"] = ds_dir

sys.exit(subprocess.run(
    [sys.executable, os.path.join(REPO, "3_si_curriculum", "RL", "data_prep.py")],
    env=env,
).returncode)
