#!/usr/bin/env python3
"""sft.train_lora entrypoint (Phase B thin wrapper).

Twin of scripts/phases/sft.sh::step_train_lora. Resolves the base model +
sft.* knobs via pipeline_config (single-GPU-safe fallbacks mirror
configs/default.yaml::sft), clears stale checkpoints so an old higher-epoch dir
can't shadow this run's in merge_lora, then runs the UNCHANGED trainer.py
(torchrun when NPROC>1). W&B is governed by the inherited WANDB_MODE
(pipeline.sh::wandb_autodisable) — no per-step guard.
"""
import glob
import os
import shutil
import subprocess
import sys

REPO = os.environ["REPO_ROOT"]
sys.path.insert(0, REPO)
import pipeline_config  # noqa: E402

OUT = os.environ["OUTPUT_BASE"]

base_model = pipeline_config.get_model_id("base_sft", "")
if not base_model:
    sys.exit("sft.train_lora: models.base_sft not set (configs/default.yaml)")

ds_dir = os.path.join(OUT, "sft_dataset")
ckpt_dir = os.path.join(OUT, "sft_checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)
# Parity with the bash step: clear stale checkpoints (trainer.py trains fresh).
for d in glob.glob(os.path.join(ckpt_dir, "checkpoint-*")):
    shutil.rmtree(d, ignore_errors=True)

# Fallbacks mirror configs/default.yaml::sft (fire only if config is unreadable),
# so degraded-mode matches the real default.
batch = pipeline_config.get_phase_param("sft", "per_device_train_batch_size", 1)
accum = pipeline_config.get_phase_param("sft", "gradient_accumulation_steps", 8)
epochs = pipeline_config.get_phase_param("sft", "num_train_epochs", 3)
ckpt = pipeline_config.get_phase_param("sft", "gradient_checkpointing", True)
domain = os.environ.get("SI_DOMAIN", "neuroscience")
wandb_project = os.environ.get("WANDB_PROJECT") or f"{domain}_sft_kg"

train_args = [
    "--model_name", base_model,
    "--train_dataset_path", ds_dir,
    "--output_dir", ckpt_dir,
    "--wandb_dir", os.path.join(OUT, "wandb_logs"),
    "--wandb_project", wandb_project,
    "--per_device_train_batch_size", str(batch),
    "--gradient_accumulation_steps", str(accum),
    "--num_train_epochs", str(epochs),
    "--gradient_checkpointing", str(ckpt),
]

nproc = os.environ.get("NPROC", "1")
cwd = os.path.join(REPO, "3_si_curriculum", "training")
if nproc == "1":
    cmd = [sys.executable, "trainer.py", *train_args]          # direct: faster startup
else:
    cmd = ["torchrun", f"--nproc_per_node={nproc}", "trainer.py", *train_args]

sys.exit(subprocess.run(cmd, cwd=cwd).returncode)
