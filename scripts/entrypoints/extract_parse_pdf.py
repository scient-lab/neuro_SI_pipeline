#!/usr/bin/env python3
"""extract.parse_pdf entrypoint (Phase B thin wrapper).

parse_pdf is the corpus INGEST step: stage the raw .txt corpus into
graphrag/input/ (+ settings.yaml) so the graphrag steps have their workspace
(declared output: graphrag/input/*.txt). The bash phase did this as a phase-level
preamble; the data-driven model makes it parse_pdf's job.

Delegates to scripts/lib/stage_corpus.sh — the SAME helper extract.sh calls — so
there is ZERO logic divergence from the proven bash (CORPUS_PATH resolve, token
expand, S3 auto-pull, subdir flatten, scale warn, settings.yaml). The env it
needs (CORPUS_PATH, S3_URI, SI_DOMAIN, SI_PROFILE, OUTPUT_BASE, REPO_ROOT) is
inherited from the runner; stage_corpus.sh activates the graphrag venv itself.
"""
import os
import subprocess
import sys

REPO = os.environ["REPO_ROOT"]
stager = os.path.join(REPO, "scripts", "lib", "stage_corpus.sh")
sys.exit(subprocess.run(["bash", stager], cwd=REPO).returncode)
