#!/usr/bin/env python3
"""extract.chunk entrypoint (Phase B thin wrapper).

Twin of extract.sh::step_chunk — graphrag step 1 (base text units) via the
UNCHANGED 1_seed_kg/graphrag_index.py, at --root_dir $OUTPUT_BASE/graphrag.
"""
import os
import subprocess
import sys

REPO = os.environ["REPO_ROOT"]
OUT = os.environ["OUTPUT_BASE"]
graphrag_dir = os.path.join(OUT, "graphrag")

sys.exit(subprocess.run(
    [sys.executable, "graphrag_index.py", "--root_dir", graphrag_dir, "--step", "1"],
    cwd=os.path.join(REPO, "1_seed_kg"),
).returncode)
