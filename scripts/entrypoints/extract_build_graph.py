#!/usr/bin/env python3
"""extract.build_graph entrypoint (Phase B thin wrapper).

Twin of extract.sh::step_build_graph — graphrag step 4 (parse LLM responses into
entity/relationship tables) via the UNCHANGED 1_seed_kg/graphrag_index.py.
Graphrag-internal step; the deliverable is written by finalize_seed_kg.
"""
import os
import subprocess
import sys

REPO = os.environ["REPO_ROOT"]
OUT = os.environ["OUTPUT_BASE"]
graphrag_dir = os.path.join(OUT, "graphrag")

sys.exit(subprocess.run(
    [sys.executable, "graphrag_index.py", "--root_dir", graphrag_dir, "--step", "4"],
    cwd=os.path.join(REPO, "1_seed_kg"),
).returncode)
