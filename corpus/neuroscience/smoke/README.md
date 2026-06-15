# neuroscience / smoke corpus

10 Wikipedia article extracts (~360 KB) used by the smoke profile.

## Provenance

| Field | Value |
|---|---|
| Source | English Wikipedia, MediaWiki action=query API (`prop=extracts&explaintext=1`) |
| Fetcher | `kg-pipeline/scripts/wikipedia_to_text.sh` |
| Topic list | `kg-pipeline/configs/wikipedia_topics.yaml::domains.neuroscience` |
| Date fetched | 2026-06-15 |
| Total | 10 of 20 requested (10 failed silently — likely Wikipedia redirect chains) |

## Files

```
Acetylcholine.txt           21 KB
Action_potential.txt        68 KB
Dopamine.txt                57 KB
GABA.txt                    15 KB
Glutamate.txt               12 KB
Hippocampus.txt             55 KB
Long-term_potentiation.txt  32 KB
Neuron.txt                  43 KB
Substantia_nigra.txt        19 KB
Synapse.txt                 21 KB
```

## Regenerate

```bash
cd ../../../../kg-pipeline
./scripts/wikipedia_to_text.sh \
    --domain neuroscience \
    --output-dir ../neuro_SI_pipeline/corpus/neuroscience/smoke
```

Re-fetch when content has materially evolved (every 6-12 months) or when
the topic list in `kg-pipeline/configs/wikipedia_topics.yaml` changes.

## How the pipeline consumes this

`configs/profiles/smoke.yaml` sets `extract.input_dir` to point here.

The wire flows like this:

1. `pipeline.sh --profile smoke --phase extract` is invoked.
2. `scripts/phases/extract.sh` reads `extract.input_dir` from the merged
   config (i.e. this directory's path: `corpus/neuroscience/smoke`).
3. extract.sh **copies** (NOT symlinks) `*.txt` from here to
   `outputs/graphrag/input/`, which is where `graphrag_index.py` looks.
   - `cp` is used instead of `ln -s` because symlinks misbehave on RunPod
     volume mounts and some Docker overlay filesystems.
   - README.md and other non-`.txt` files are explicitly skipped so
     graphrag does not try to ingest this metadata as a corpus document.
4. Phase script then invokes `graphrag_index.py --step <step>` (currently
   a stub log line; will be wired in a follow-up).

To run it manually today (before the phase script wire lands fully):

```bash
mkdir -p outputs/graphrag/input
cp corpus/neuroscience/smoke/*.txt outputs/graphrag/input/
source .venvs/graphrag/bin/activate
python 1_seed_kg/graphrag_index.py --root_dir outputs/graphrag --step 1
```
