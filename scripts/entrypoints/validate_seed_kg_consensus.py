#!/usr/bin/env python3
"""validate.seed_kg_consensus entrypoint (Phase B thin wrapper).

Twin of scripts/phases/validate.sh::step_seed_kg_consensus. Runs the two-LLM
consensus filter (paper §4.2) on extract's seed KG via the UNCHANGED
2_graphmert/utils/llm_scores/fact_score.py, under the graphmert venv.

Preserved from the bash step:
  * models.validate_a + models.validate_b from config, fail-loud if unset.
  * --max_model_len intentionally OMITTED — fact_score.py reads
    graphmert.fact_score_max_model_len from config (profile-tunable in one place).
  * --batch_size 64 mirrors validate.sh (also fact_score.py's own default).
  * drop-rate report with the before==0 GUARD (an empty seed KG gives a clear
    message, not a div-by-zero that masks the real cause).

Input (extract's seed KG) is the upstream finalize_seed_kg output — resolved by
its stable path (§5.1 interim). This step's OWN output is injected as
$STEP_OUTPUT by run_phase.py from the declared `output` in pipeline_execution.yaml.
"""
import os
import subprocess
import sys

REPO = os.environ["REPO_ROOT"]
sys.path.insert(0, REPO)
import pipeline_config  # noqa: E402

OUT = os.environ["OUTPUT_BASE"]

validate_a = pipeline_config.get_model_id("validate_a", "")
validate_b = pipeline_config.get_model_id("validate_b", "")
if not validate_a or not validate_b:
    sys.exit("validate.seed_kg_consensus: models.validate_a + models.validate_b "
             "required (configs/default.yaml)")

seed_kg = os.path.join(OUT, "graphrag", "output", "kg_final.csv")
if not os.path.isfile(seed_kg):
    sys.exit(f"validate.seed_kg_consensus: seed KG not found at {seed_kg} — "
             "run the extract phase first")

# Output: prefer $STEP_OUTPUT (injected from execution.yaml `output` — single
# source); fall back to the canonical path when run standalone (no runner).
validated = os.environ.get("STEP_OUTPUT") or os.path.join(
    OUT, "graphrag", "output", "kg_final_validated.csv")
os.makedirs(os.path.dirname(validated), exist_ok=True)

print(f"validate :: two-LLM consensus  in={seed_kg}  "
      f"models={validate_a} (A) + {validate_b} (B)  out={validated}")

rc = subprocess.run(
    [sys.executable, "utils/llm_scores/fact_score.py",
     "--input_csv", seed_kg,
     "--output_csv", validated,
     "--model_ids", validate_a, validate_b,
     "--batch_size", "64"],
    cwd=os.path.join(REPO, "2_graphmert"),
).returncode
if rc != 0:
    sys.exit(rc)


def _rows(path: str) -> int:
    """Data-row count (minus the CSV header), floored at 0 — mirrors `wc -l - 1`."""
    with open(path) as f:
        return max(sum(1 for _ in f) - 1, 0)


before = _rows(seed_kg)
after = _rows(validated) if os.path.isfile(validated) else 0
drop = before - after
if before > 0:
    print(f"validate :: consensus filter: {before} triples in → {after} passed "
          f"(dropped {drop}, {drop * 100 // before}%)")
else:
    print("validate :: seed KG had 0 triples — nothing to validate "
          "(check the extract phase output)")
