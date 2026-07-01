#!/usr/bin/env python3
"""analysis_rl.py — human-facing analysis of the RL (GRPO) phase.

Primary source is the structured HF/GRPO log_history in the newest
  OUTPUT_BASE/rl_checkpoints/checkpoint-*/trainer_state.json
which carries a per-step record of reward (total + per-reward-function), KL,
loss, grad_norm, and completion-length stats — far more robust than regex over
the multi-MB train_grpo.log. Falls back to that log only for the merged-model
size note.

Sections:
  §source       which trainer_state.json, how many steps / epochs
  §reward       total reward trajectory (first-window → last-window mean); the
                'did RL learn?' signal — the training-side twin of the merge
                probe's "merged == SFT base" no-op check.
  §components   correctness / format / path_alignment reward funcs (KG-path
                reward is path_alignment)
  §stability    KL vs the reference, grad-norm, loss (NaN/exploding guards)
  §completions  mean length + clipped ratio (reasoning hitting max_len?)
  §merge        merged_final_model size (reuses step_quality helpers)

Thresholds live in configs/default.yaml::expectations.rl.train_grpo (read via
step_quality.exp — no duplicated constants). Exit: 0 clean / 1 fail / 2 warn.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# same-dir import: analysis.sh runs `python scripts/lib/analysis_rl.py`, so
# sys.path[0] is scripts/lib. Reuse the config-aware exp() + safetensors helpers.
import step_quality as sq


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


def _series(lh: list, key: str) -> list:
    """Numeric values of `key` across log_history, skipping missing/None/non-finite."""
    out = []
    for e in lh:
        v = e.get(key)
        if isinstance(v, (int, float)) and v == v and abs(v) != float("inf"):  # v==v drops NaN
            out.append(float(v))
    return out


def _window(vals: list, frac: float = 0.1):
    """(first-window mean, last-window mean) using a max(1, len*frac) window, so
    a single noisy first/last step doesn't decide 'did reward improve?'."""
    if not vals:
        return (None, None)
    w = max(1, int(len(vals) * frac))
    fa = sum(vals[:w]) / w
    la = sum(vals[-w:]) / w
    return (fa, la)


def _any_nonfinite(lh: list, key: str) -> bool:
    for e in lh:
        v = e.get(key)
        if isinstance(v, (int, float)) and (v != v or abs(v) == float("inf")):
            return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--step", default=None, choices=["train_grpo", "merge_rl", None],
                   help="restrict analysis (train_grpo = training curves; merge_rl = merged model)")
    p.add_argument("--top", type=int, default=10, help="(unused; accepted for analysis.sh parity)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--run", default=None,
                   help="RUN_ID prefix; default = $OUTPUT_BASE or auto-detect latest")
    args = p.parse_args()

    repo_root = Path(args.repo_root)
    output_base = Path(os.environ.get("OUTPUT_BASE", repo_root / "outputs"))

    # Run resolution: flat OUTPUT_BASE (has rl_checkpoints/) OR a run subdir.
    def _find_run_dir() -> Optional[Path]:
        if (output_base / "rl_checkpoints").exists():
            return output_base
        pat = f"{args.run}*" if args.run else "*"
        cands = sorted((d for d in output_base.glob(pat)
                        if d.is_dir() and (d / "rl_checkpoints").exists()), reverse=True)
        return cands[0] if cands else None

    ob = _find_run_dir()
    r = Reporter(json_mode=args.json, quiet=args.quiet)

    if ob is None:
        r.section("source")
        r.fail(f"no rl_checkpoints/ under {_rel(output_base, repo_root)}")
        return r.emit()

    ckpt = sq._newest(ob, "rl_checkpoints/checkpoint-*")
    state = (ckpt / "trainer_state.json") if ckpt else None

    do_train = args.step in (None, "train_grpo")
    do_merge = args.step in (None, "merge_rl")

    # --- §source + training curves -----------------------------------------
    lh = []
    if do_train:
        r.section("source")
        if not state or not state.exists():
            r.warn(f"no trainer_state.json in newest rl checkpoint ({ckpt.name if ckpt else '—'}) "
                   f"— GRPO curves unavailable")
        else:
            try:
                doc = json.loads(state.read_text())
                lh = doc.get("log_history", [])
                r.ok(f"{_rel(state, repo_root)}: {len(lh)} logged steps, "
                     f"{doc.get('num_train_epochs', '?')} epoch(s), max_steps={doc.get('max_steps', '?')}")
            except Exception as e:
                r.fail(f"trainer_state.json unreadable: {e}")

    if do_train and lh:
        imp_min = sq.exp("rl", "train_grpo", "reward_improve_min", 0.0)
        kl_hi = sq.exp("rl", "train_grpo", "kl_warn_high", 0.5)
        clip_hi = sq.exp("rl", "train_grpo", "clipped_ratio_warn", 0.30)
        gn_hi = sq.exp("rl", "train_grpo", "grad_norm_warn_high", 100.0)

        # §reward — the "did RL learn?" signal
        r.section("reward")
        rw = _series(lh, "reward")
        if not rw:
            r.warn("no 'reward' logged")
        else:
            fa, la = _window(rw)
            delta = la - fa
            msg = f"reward {fa:+.3f} → {la:+.3f} (Δ{delta:+.3f}), range [{min(rw):+.3f}, {max(rw):+.3f}]"
            if delta > imp_min:
                r.ok(msg)
            else:
                r.warn(msg + f" — did not improve (Δ≤{imp_min}); RL may be a no-op over SFT")
            # windowed means drive the verdict (robust to per-step noise); show the
            # raw endpoints + best too, since the step-to-step story is starker.
            r.note(f"per-step: first {rw[0]:+.3f}, final {rw[-1]:+.3f}, best {max(rw):+.3f}")
            rs = _series(lh, "reward_std")
            if rs:
                r.note(f"reward_std {rs[0]:.3f} → {rs[-1]:.3f} (spread across the GRPO group)")

        # §components — per reward function (path_alignment = KG-path reward)
        r.section("components")
        comp = {"correctness": "rewards/correctness_reward_func/mean",
                "format": "rewards/format_reward_func/mean",
                "path_alignment": "rewards/path_alignment_reward_func/mean"}
        for label, key in comp.items():
            s = _series(lh, key)
            if not s:
                r.note(f"{label}: not logged")
                continue
            fa, la = _window(s)
            line = f"{label:<14} {fa:+.3f} → {la:+.3f} (Δ{la - fa:+.3f})"
            # correctness is the task reward — flag if it never moved up
            if label == "correctness" and (la - fa) <= imp_min:
                r.warn(line + " — task reward flat/declining")
            else:
                r.ok(line)

        # §stability — KL / grad-norm / loss
        r.section("stability")
        kl = _series(lh, "kl")
        if kl:
            if kl[-1] > kl_hi:
                r.warn(f"final KL {kl[-1]:.3f} > {kl_hi} — policy drifting far from the reference")
            else:
                r.ok(f"KL {kl[0]:.3f} → {kl[-1]:.3f} (< {kl_hi})")
        if _any_nonfinite(lh, "grad_norm") or _any_nonfinite(lh, "loss"):
            r.fail("NaN/Inf in grad_norm or loss — training diverged")
        else:
            gn = _series(lh, "grad_norm")
            if gn and max(gn) > gn_hi:
                r.warn(f"grad_norm peaked at {max(gn):.1f} (> {gn_hi}) — instability")
            elif gn:
                r.ok(f"grad_norm peak {max(gn):.2f} (< {gn_hi})")
            ls = _series(lh, "loss")
            if ls:
                r.note(f"loss {ls[0]:.4f} → {ls[-1]:.4f}")

        # §completions — length + clipping (truncated reasoning?)
        r.section("completions")
        cl = _series(lh, "completions/mean_length")
        clip = _series(lh, "completions/clipped_ratio")
        if cl:
            r.note(f"mean completion length {cl[0]:.0f} → {cl[-1]:.0f} tokens")
        if clip:
            if clip[-1] > clip_hi:
                r.warn(f"final clipped_ratio {clip[-1]:.0%} > {clip_hi:.0%} — completions hitting "
                       f"max_completion_length; reasoning likely truncated (raise the cap)")
            else:
                r.ok(f"clipped_ratio {clip[0]:.0%} → {clip[-1]:.0%} (< {clip_hi:.0%})")

    # --- §merge — merged model size (reuse step_quality helpers) ------------
    if do_merge:
        r.section("merge")
        if ckpt is None:
            r.warn("no rl checkpoint")
        else:
            size = sq._merged_total_size(ckpt / "merged_final_model")
            if size is None:
                # a full-FT (use_lora=false) run legitimately has no merge
                r.note("no merged_final_model/ (full-FT RL writes full checkpoints directly, or LoRA merge not run)")
            else:
                sft_ckpt = sq._newest(ob, "sft_checkpoints/checkpoint-*")
                bsize = sq._merged_total_size(sft_ckpt / "merged_final_model") if sft_ckpt else None
                tail = f"; SFT base {bsize/1e9:.1f} GB" if bsize else ""
                r.ok(f"merged_final_model {size / 1e9:.1f} GB{tail} "
                     f"(deep 'weights differ from SFT base' check → diagnose.sh --phase rl)")

    return r.emit()


if __name__ == "__main__":
    sys.exit(main())
