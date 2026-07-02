#!/usr/bin/env python3
"""extract.finalize_seed_kg entrypoint (Phase B thin wrapper).

Twin of extract.sh::step_finalize_seed_kg — graphrag step 5 (clean + finalize),
then convert graphrag's final_relationships.parquet (source/target/relation)
into the kg_final.{csv,parquet} (head,relation,tail) that graphmert step 4 +
curriculum calculate_hops + graphmert merge_kgs consume. Uses the UNCHANGED
1_seed_kg/graphrag_index.py.

Writes BOTH files: the .csv honors $STEP_OUTPUT (declared output
graphrag/output/kg_final.csv, injected by the runner); the .parquet is its
undeclared sibling in the same dir.
"""
import os
import subprocess
import sys

REPO = os.environ["REPO_ROOT"]
OUT = os.environ["OUTPUT_BASE"]
graphrag_dir = os.path.join(OUT, "graphrag")

rc = subprocess.run(
    [sys.executable, "graphrag_index.py", "--root_dir", graphrag_dir, "--step", "5"],
    cwd=os.path.join(REPO, "1_seed_kg"),
).returncode
if rc:
    sys.exit(rc)

# Imported after the graphrag step so a step failure surfaces before paying the
# pandas import cost. pandas lives in the graphrag venv (same as the bash step).
import pandas as pd  # noqa: E402

src = os.path.join(graphrag_dir, "output", "final_relationships.parquet")
csv_path = os.environ.get("STEP_OUTPUT") or os.path.join(graphrag_dir, "output", "kg_final.csv")
pq_path = os.path.join(os.path.dirname(csv_path), "kg_final.parquet")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

df = pd.read_parquet(src)
out = df[["source", "target", "relation"]].rename(columns={"source": "head", "target": "tail"})
out.to_csv(csv_path, index=False)
out.to_parquet(pq_path, index=False)
print(f"wrote {len(out)} triples to {csv_path} and {pq_path}")
