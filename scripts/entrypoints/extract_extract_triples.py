#!/usr/bin/env python3
"""extract.extract_triples entrypoint (Phase B thin wrapper).

Twin of extract.sh::step_extract_triples — graphrag step 2 (documents) then
step 3 (vLLM head/relation/tail extraction, using models.extract). Fail-loud if
models.extract is unset. Uses the UNCHANGED 1_seed_kg/graphrag_index.py.
"""
import os
import subprocess
import sys

REPO = os.environ["REPO_ROOT"]
sys.path.insert(0, REPO)
import pipeline_config  # noqa: E402

OUT = os.environ["OUTPUT_BASE"]
graphrag_dir = os.path.join(OUT, "graphrag")
cwd = os.path.join(REPO, "1_seed_kg")


def _step(n, *extra) -> int:
    return subprocess.run(
        [sys.executable, "graphrag_index.py", "--root_dir", graphrag_dir,
         "--step", str(n), *extra],
        cwd=cwd,
    ).returncode


rc = _step(2)
if rc:
    sys.exit(rc)

model_id = pipeline_config.get_model_id("extract", "")
if not model_id:
    sys.exit("extract.extract_triples: models.extract required "
             "(configs/default.yaml or domain override)")
sys.exit(_step(3, "--model_id", model_id))
