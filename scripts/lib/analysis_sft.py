#!/usr/bin/env python3
"""analysis_sft.py — human-facing analysis of the SFT (LoRA) phase.

Unlike RL, the SFT trainer writes no trainer_state.json (save_strategy="no" +
per-epoch EpochCheckpointCallback), so the loss source is the step's stdout log
  OUTPUT_BASE/logs/<RUN_ID>/sft/train_lora.log
which carries HF Trainer's dict lines. At smoke scale (1 epoch, default
logging_steps) that's just the final summary
  {'train_runtime':.., 'train_loss':.., 'mean_token_accuracy':.., 'epoch':1.0}
while pilot/paper (more steps / lower logging_steps) also emit per-step
  {'loss':.., 'grad_norm':.., 'learning_rate':.., 'epoch':..}
lines — this parser handles both.

Sections:
  §source     which log, base model, LoRA config, train/test sizes
  §loss       per-step loss trajectory if present, else the final train_loss
  §accuracy   mean_token_accuracy
  §merge      adapter + merged_final_model size (reuses step_quality helpers)

Thresholds live in configs/default.yaml::expectations.sft.train_lora (read via
step_quality.exp). Exit: 0 clean / 1 fail / 2 warn.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import step_quality as sq   # same-dir; config-aware exp() + safetensors helpers


# ---------------------------------------------------------------------------
# Reporter — mirrors analysis_extract.py / analysis_curriculum.py exactly.
# ---------------------------------------------------------------------------
class Reporter:
    def __init__(self, json_mode: bool = False, quiet: bool = False):
        self.json_mode = json_mode
        self.quiet = quiet
        self.results: list[dict] = []
        self.section_name = ""
        self.fails = 0
        self.warns = 0

    def section(self, name: str) -> None:
        self.section_name = name
        if not self.json_mode:
            print(f"\n  [§{name}]")

    def ok(self, msg: str) -> None:
        if self.json_mode:
            self.results.append({"section": self.section_name, "level": "ok", "msg": msg})
        elif not self.quiet:
            print(f"    OK   {msg}")

    def warn(self, msg: str) -> None:
        self.warns += 1
        if self.json_mode:
            self.results.append({"section": self.section_name, "level": "warn", "msg": msg})
        else:
            print(f"    WARN {msg}")

    def fail(self, msg: str) -> None:
        self.fails += 1
        if self.json_mode:
            self.results.append({"section": self.section_name, "level": "fail", "msg": msg})
        else:
            print(f"    FAIL {msg}")

    def note(self, msg: str) -> None:
        if self.json_mode or self.quiet:
            return
        print(f"         {msg}")

    def emit(self) -> int:
        if self.json_mode:
            print(json.dumps({"fails": self.fails, "warns": self.warns,
                              "results": self.results}, indent=2))
        else:
            verdict = "CLEAN" if self.fails == 0 and self.warns == 0 else \
                      "PASS WITH WARNINGS" if self.fails == 0 else "FAIL"
            print(f"\n  VERDICT: {self.fails} failures, {self.warns} warnings — {verdict}")
        return 1 if self.fails > 0 else (2 if self.warns > 0 else 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rel(path: Path, repo_root: Path) -> str:
    try:
        return f"./{path.relative_to(repo_root)}"
    except ValueError:
        return str(path)


def _find_train_log(ob: Path, repo_root: Path) -> Optional[Path]:
    """Locate sft/train_lora.log: manifest's log_file first (authoritative),
    then conventional globs for both flat and run-subdir OUTPUT_BASE layouts."""
    man = ob / "run_manifest.json"
    if man.exists():
        try:
            doc = json.loads(man.read_text())
            for p in doc.get("run", {}).get("phases", []):
                if p.get("name") != "sft":
                    continue
                for s in p.get("steps", []):
                    if s.get("name") == "train_lora" and s.get("log_file"):
                        for c in (repo_root / s["log_file"], ob / s["log_file"], Path(s["log_file"])):
                            if c.exists():
                                return c
        except Exception:
            pass
    cands = []
    for pat in ("logs/*/sft/train_lora.log", "logs/sft/train_lora.log", "sft/train_lora.log"):
        cands.extend(ob.glob(pat))
    cands = [c for c in cands if c.exists()]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def _parse_metric_dicts(log_path: Path) -> list:
    """HF Trainer prints python-dict lines. Pull each {...} that ast.literal_eval
    accepts and that carries a loss/accuracy key."""
    out = []
    try:
        text = log_path.read_text(errors="replace")
    except Exception:
        return out
    for line in text.splitlines():
        i, j = line.find("{"), line.rfind("}")
        if i == -1 or j <= i:
            continue
        try:
            d = ast.literal_eval(line[i:j + 1])
        except Exception:
            continue
        if isinstance(d, dict) and ("loss" in d or "train_loss" in d or "mean_token_accuracy" in d):
            out.append(d)
    return out


def _grep1(text: str, pattern: str):
    m = re.search(pattern, text)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--step", default=None, choices=["train_lora", "merge_lora", None],
                   help="restrict analysis (train_lora = loss/accuracy; merge_lora = merged model)")
    p.add_argument("--top", type=int, default=10, help="(unused; accepted for analysis.sh parity)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--run", default=None,
                   help="RUN_ID prefix; default = $OUTPUT_BASE or auto-detect latest")
    args = p.parse_args()

    repo_root = Path(args.repo_root)
    output_base = Path(os.environ.get("OUTPUT_BASE", repo_root / "outputs"))

    def _find_run_dir() -> Optional[Path]:
        if (output_base / "sft_checkpoints").exists():
            return output_base
        pat = f"{args.run}*" if args.run else "*"
        cands = sorted((d for d in output_base.glob(pat)
                        if d.is_dir() and (d / "sft_checkpoints").exists()), reverse=True)
        return cands[0] if cands else None

    ob = _find_run_dir()
    r = Reporter(json_mode=args.json, quiet=args.quiet)
    if ob is None:
        r.section("source")
        r.fail(f"no sft_checkpoints/ under {_rel(output_base, repo_root)}")
        return r.emit()

    do_train = args.step in (None, "train_lora")
    do_merge = args.step in (None, "merge_lora")

    loss_hi = sq.exp("sft", "train_lora", "final_loss_warn_high", 3.0)
    acc_lo = sq.exp("sft", "train_lora", "token_acc_warn_low", 0.40)

    if do_train:
        log = _find_train_log(ob, repo_root)
        # --- §source ---
        r.section("source")
        text = ""
        if not log:
            r.warn("no sft/train_lora.log found — loss/accuracy unavailable")
        else:
            text = log.read_text(errors="replace")
            r.ok(_rel(log, repo_root))
            base = _grep1(text, r"Loading model:\s*(\S+)")
            lora = _grep1(text, r"(LoRA config:[^\n]*)")
            sizes = _grep1(text, r"(Dataset sizes[^\n]*)")
            lr = _grep1(text, r"learning_rate\s*:\s*(\S+)")
            if base:
                r.note(f"base model: {base}")
            if sizes:
                r.note(sizes.strip())
            if lora:
                r.note(lora.strip() + (f"  (lr={lr})" if lr else ""))

        # --- §loss ---
        r.section("loss")
        dicts = _parse_metric_dicts(log) if log else []
        steps = [d["loss"] for d in dicts if "loss" in d]
        final = next((d for d in reversed(dicts) if "train_loss" in d), None)
        if steps:
            trend = f"loss {steps[0]:.4f} → {steps[-1]:.4f} (min {min(steps):.4f}, {len(steps)} pts)"
            if steps[-1] > steps[0]:
                r.warn(trend + " — loss did not decrease")
            else:
                r.ok(trend)
        if final and "train_loss" in final:
            tl = final["train_loss"]
            if tl > loss_hi:
                r.warn(f"final train_loss {tl:.4f} > {loss_hi} — barely trained")
            else:
                r.ok(f"final train_loss {tl:.4f} (< {loss_hi})")
            if "train_runtime" in final:
                r.note(f"train_runtime {final['train_runtime']:.0f}s, "
                       f"{final.get('num_tokens', 0):.0f} tokens")
        elif not steps:
            r.warn("no loss found in the SFT log")

        # --- §accuracy ---
        acc = final.get("mean_token_accuracy") if final else None
        if acc is None:
            for d in reversed(dicts):
                if "mean_token_accuracy" in d:
                    acc = d["mean_token_accuracy"]; break
        if acc is not None:
            r.section("accuracy")
            if acc < acc_lo:
                r.warn(f"mean_token_accuracy {acc:.3f} < {acc_lo} — weak fit")
            else:
                r.ok(f"mean_token_accuracy {acc:.3f} (≥ {acc_lo})")

    # --- §merge ---
    if do_merge:
        r.section("merge")
        ckpt = sq._newest(ob, "sft_checkpoints/checkpoint-*")
        if ckpt is None:
            r.warn("no sft checkpoint")
        else:
            nz, tot = sq._lora_B_nonzero(ckpt / "adapter_model.safetensors")
            size = sq._merged_total_size(ckpt / "merged_final_model")
            if tot:
                (r.ok if nz else r.fail)(
                    f"adapter lora_B {nz}/{tot} non-zero" + ("" if nz else " — untrained adapter"))
            if size is None:
                r.warn("no merged_final_model/ (merge_lora not run or failed)")
            else:
                r.ok(f"merged_final_model {size / 1e9:.1f} GB "
                     f"(deep full-model check → diagnose.sh --phase sft)")

    return r.emit()


if __name__ == "__main__":
    sys.exit(main())
