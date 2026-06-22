# Neuro SI Pipeline

Official implementation of [Knowledge Graph-Driven Expert-Level Reasoning for Neuroscience](https://arxiv.org/abs/2605.25183)

**[Neuro-Bench](https://kg-bottom-up-superintelligence.github.io/neuro-bench/)**: The primary dataset our model was trained and evaluated on, comprising 5,000 high-quality neuroscience reasoning questions systematically generated from knowledge-graph paths and balanced evenly across 1-Hop to 5-Hop complexities.


```
Textbook corpus
    │
    ▼ Part 1 — GraphRAG
Seed KG  (head, relation, tail triples)
    │
    ▼ Part 2 — GraphMERT
Expanded KG  (2–3× coverage via masked language model)
    │
    ▼ Part 3 — SI Curriculum
Multi-hop Q&A dataset  →  SFT  →  RL (GRPO)
```

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Repository Layout](#repository-layout)
3. [Part 1 — Seed KG Generation](#part-1--seed-kg-generation)
4. [Part 2 — GraphMERT Expansion](#part-2--graphmert-expansion)
5. [Part 3 — SI Curriculum, SFT & RL](#part-3--si-curriculum-sft--rl)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Compute
- SLURM HPC cluster with GPU nodes (A100 80 GB recommended for large models)
- Do **not** run GPU steps on login nodes

### Common environment variables
Set these once in your shell or `.bashrc` before running anything:

```bash
export REPO_DIR=/path/to/neuro_SI_pipeline    # path to this repo checkout
export OUTPUT_BASE=/scratch/<cluster>/<you>/neuro_pipeline  # base output dir
export HF_HOME=$OUTPUT_BASE/cache/hf_home     # Hugging Face model cache
```

### Conda environments

Three separate environments, one per stage:

**Part 1 — graphrag**
```bash
conda create -n graphrag python=3.11 -y
conda activate graphrag
pip install torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.7.3 transformers datasets pandas pyarrow
```

**Part 2 — graphmert**
```bash
conda create -n graphmert python=3.10 -y
conda activate graphmert
pip install torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install vllm transformers datasets spacy pandas
python -m spacy download en_core_web_sm
```

**Part 3 — si_curriculum**
```bash
conda create -n si_curriculum python=3.10 -y
conda activate si_curriculum
pip install torch==2.4.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install transformers trl peft deepspeed accelerate datasets \
            google-generativeai openai pandas matplotlib
```

---

## Repository Layout

```
neuro_SI_pipeline/
├── README.md
├── 1_seed_kg/
│   ├── graphrag_index.py          # Steps 1–5: extract KG from corpus
│   ├── count_text_units.py        # Size SLURM array jobs
│   ├── merge_kgs.py               # Merge KG shards
│   ├── prompts_kg.py              # Relation-type prompts
│   ├── settings.yaml              # GraphRAG run settings
│   ├── llm_kg/                    # Standalone LLM-only KG builder
│   └── slurm/job.slurm
│
├── 2_graphmert/
│   ├── run_tokenization.py        # Step 1: tokenise corpus + save stable tokenizer
│   ├── run_dataset_preprocessing.py  # Step 4: co-occurrence grounding
│   ├── run_mlm.py                 # Step 5: train GraphMERT
│   ├── predict_tails_llm.py       # Step 6: predict novel tails
│   ├── graphmert_model.py         # Model architecture
│   ├── models.py                  # Download/cache HF models
│   ├── launch_configs/args_mlm.yaml
│   ├── utils/
│   │   ├── tokenization_utils.py
│   │   ├── dataset_preprocessing_utils.py
│   │   ├── mlm_utils.py
│   │   ├── entity_discovery/      # Step 2: mine head entities
│   │   ├── relation_matching/     # Step 3: assign relations via LLM
│   │   ├── combine_tails/         # Step 7: merge predicted tails
│   │   └── llm_scores/            # Step 7: two-LLM fact validation
│   └── slurm/{train_graphmert,predict_tails,entity_discovery}.slurm
│
└── 3_si_curriculum/
    ├── calculate_hops.py          # Pre-step: compute hop distances
    ├── curriculum_generator/
    │   ├── generate_curriculum.py # Step 1: generate Q&A items
    │   └── verify_questions.py    # Step 2: two-LLM filter
    ├── training/
    │   ├── data_prep.py           # Step 3: prepare SFT dataset
    │   ├── trainer.py             # Step 4: LoRA SFT training
    │   └── merge_lora.py          # Step 5: merge adapter into base
    ├── RL/
    │   ├── data_prep.py           # Step 6: prepare RL dataset
    │   └── rl_training.py         # Step 7: GRPO RL training
    ├── test_models/
    │   ├── eval_models.py         # Step 8: multi-checkpoint eval
    │   ├── data_analysis.py       # Accuracy + hop-breakdown plots
    │   └── correctness_similarity.py  # Error overlap analysis
    └── slurm/{generate_curriculum,verify_questions,sft_trainer,
               rl_training,eval_models}.slurm
```

---

## Part 1 — Seed KG Generation

**Environment:** `graphrag`

### 1.1 Prepare your corpus

Place `.txt` files (one per section/chapter) in:
```
${OUTPUT_BASE}/graphrag/input/
```

If starting from PDFs, extract body text only (skip captions and footers).
Inspect the text manually before proceeding — garbage input produces a garbage KG.

### 1.2 Size the SLURM array job

```bash
conda activate graphrag
cd $REPO_DIR

python 1_seed_kg/count_text_units.py \
    --root_dir    $OUTPUT_BASE/graphrag \
    --rows_per_job 512
# Prints: "Use --array=0-<N>" — note this number for 1.4
```

### 1.3 Steps 1 & 2 — chunk text and build document records

```bash
python 1_seed_kg/graphrag_index.py \
    --root_dir  $OUTPUT_BASE/graphrag \
    --model_id  /path/to/extraction/model \
    --step 1

python 1_seed_kg/graphrag_index.py \
    --root_dir  $OUTPUT_BASE/graphrag \
    --model_id  /path/to/extraction/model \
    --step 2
```

### 1.4 Step 3 — LLM extraction (GPU, SLURM array)

```bash
# Set env vars consumed by job.slurm:
export REPO_DIR OUTPUT_BASE
export MODEL_ID=/path/to/extraction/model   # e.g. Qwen3-32B local path

sbatch --array=0-<N> 1_seed_kg/slurm/job.slurm

# Or run a single shard directly (SLURM_ARRAY_TASK_ID controls the shard):
SLURM_ARRAY_TASK_ID=0 python 1_seed_kg/graphrag_index.py \
    --root_dir  $OUTPUT_BASE/graphrag \
    --model_id  $MODEL_ID \
    --step 3
```

### 1.5 Steps 4 & 5 — parse responses and finalise KG

```bash
python 1_seed_kg/graphrag_index.py \
    --root_dir $OUTPUT_BASE/graphrag \
    --step 4

python 1_seed_kg/graphrag_index.py \
    --root_dir $OUTPUT_BASE/graphrag \
    --step 5
```

**Output:** `$OUTPUT_BASE/graphrag/output/kg_final.parquet`

### 1.6 (Optional) Merge incremental runs

```bash
python 1_seed_kg/merge_kgs.py \
    --new  $OUTPUT_BASE/graphrag/output/kg_final.parquet \
    --old  $OUTPUT_BASE/graphrag/output/kg_old.parquet \
    --out  $OUTPUT_BASE/graphrag/output/kg_merged.parquet
# --old is optional; omit it to use the new file as-is
```

**Expected output:** 3,000–10,000 validated triples for an average-sized textbook.

---

## Part 2 — GraphMERT Expansion

**Environment:** `graphmert`

GraphMERT treats each KG triple `(head, relation, tail)` as a tree rooted on
the head. It learns to predict masked tail entities from head + relation context,
then generates novel (head, relation, ?) completions to expand the KG.

### 2.1 Step 1 — Tokenise corpus & create stable tokenizer

```bash
conda activate graphmert
cd $REPO_DIR

python 2_graphmert/run_tokenization.py \
    --input_dir   $OUTPUT_BASE/graphrag/input \
    --output_dir  $OUTPUT_BASE/graphmert \
    --tokenizer   dmis-lab/biobert-base-cased-v1.2 \
    --max_seq_length      128 \
    --validation_split_pct 5 \
    --num_workers         8 \
    --seed                0
```

**Outputs:**
- `$OUTPUT_BASE/graphmert/stable_tokenizer/` — **use this path for ALL subsequent steps**
- `$OUTPUT_BASE/graphmert/tokenized_inputs/train_<hash>_tokenized/`
- `$OUTPUT_BASE/graphmert/tokenized_inputs/validation_<hash>_tokenized/`

> The stable tokenizer adds explicit `[PAD]` and `[MASK]` tokens with fixed IDs.
> Every downstream step must load from this saved path to guarantee consistent
> token IDs throughout the pipeline.

### 2.2 Step 2 — Entity discovery

Identify neuroscience entity mentions in each tokenized text chunk:

```bash
# Submit via SLURM:
export REPO_DIR OUTPUT_BASE
export MODEL_ID=/path/to/qwen3-32b
sbatch 2_graphmert/slurm/entity_discovery.slurm

# Or run directly:
python 2_graphmert/utils/entity_discovery/entity_discovery.py \
    --tokenized_dir  $OUTPUT_BASE/graphmert/tokenized_inputs/train_<hash>_tokenized \
    --output_dir     $OUTPUT_BASE/graphmert/entity_discovery \
    --model_id       /path/to/qwen3-32b \
    --tokenizer      $OUTPUT_BASE/graphmert/stable_tokenizer
```

Find exact token positions for each discovered head entity:

```bash
python 2_graphmert/utils/entity_discovery/find_heads_positions.py \
    --heads_chunks_dir  $OUTPUT_BASE/graphmert/entity_discovery \
    --output_dir        $OUTPUT_BASE/graphmert/head_positions \
    --tokenizer         $OUTPUT_BASE/graphmert/stable_tokenizer
```

### 2.3 Step 3 — Relation matching

Assign seed-KG relations to discovered entities via LLM:

```bash
python 2_graphmert/utils/relation_matching/add_llm_relations.py \
    --dataset_path  $OUTPUT_BASE/graphmert/head_positions \
    --output_root   $OUTPUT_BASE/graphmert/llm_relations \
    --output_name   relations_all \
    --model_id      /path/to/qwen3-14b \
    --tokenizer     $OUTPUT_BASE/graphmert/stable_tokenizer
```

Clean, validate, and split into train/eval:

```bash
python 2_graphmert/utils/relation_matching/clean_llm_relations.py \
    --input_dir   $OUTPUT_BASE/graphmert/llm_relations/relations_all \
    --output_dir  $OUTPUT_BASE/graphmert/llm_relations/relations_clean \
    --tokenizer   $OUTPUT_BASE/graphmert/stable_tokenizer
```

**Outputs:**
- `$OUTPUT_BASE/graphmert/llm_relations/relations_clean_train/`
- `$OUTPUT_BASE/graphmert/llm_relations/relations_clean_eval/`

### 2.4 Step 4 — Build training dataset (co-occurrence grounding)

For each `(head, relation, tail)` triple in the seed KG, finds text snippets
where both head AND tail appear together and builds GraphMERT training samples.

```bash
python 2_graphmert/run_dataset_preprocessing.py \
    --yaml_file    2_graphmert/launch_configs/args_mlm.yaml \
    --seed_kg_path $OUTPUT_BASE/graphrag/output/kg_final.parquet \
    --train_src    $OUTPUT_BASE/graphmert/llm_relations/relations_clean_train \
    --eval_src     $OUTPUT_BASE/graphmert/llm_relations/relations_clean_eval \
    --tokenizer    $OUTPUT_BASE/graphmert/stable_tokenizer \
    --output_dir   $OUTPUT_BASE/graphmert/dataset
```

**Outputs:**
- `$OUTPUT_BASE/graphmert/dataset/ready_for_training_train/`
- `$OUTPUT_BASE/graphmert/dataset/ready_for_training_eval/`
- `$OUTPUT_BASE/graphmert/dataset/relation_map.json`

### 2.5 Step 5 — Train GraphMERT

Edit `2_graphmert/launch_configs/args_mlm.yaml` — every field marked
`<YOUR_SCRATCH>` must be replaced with your actual paths (the yaml is the
single source of truth for training config).

Key parameters to review:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `root_nodes` | 512 | Tokens per root node |
| `num_leaves` | 3 | Leaf nodes per training sample |
| `max_nodes` | 2048 | Total node budget (= root_nodes × (1 + num_leaves)) |
| `num_train_epochs` | 10 | Training epochs |
| `per_device_train_batch_size` | 8 | Batch size per GPU |
| `learning_rate` | 1e-4 | Starting LR |
| `tokenizer_name` | — | **Must point to your stable_tokenizer directory** |

```bash
# Submit via SLURM (recommended):
export REPO_DIR OUTPUT_BASE
sbatch 2_graphmert/slurm/train_graphmert.slurm

# Or directly:
python 2_graphmert/run_mlm.py 2_graphmert/launch_configs/args_mlm.yaml
```

**Quality check:** After training, inspect eval loss. If it plateaus above 5.0,
the model has not learned useful representations — increase seed KG size or
reduce model size before continuing.

### 2.6 Step 6 — Predict novel tails (SLURM array)

```bash
# Array job (4 shards, one per GPU node):
export REPO_DIR OUTPUT_BASE
export MODEL_ID=/path/to/qwen3-32b   # path to a vLLM-compatible LLM, NOT the GraphMERT checkpoint
sbatch --array=0-3 2_graphmert/slurm/predict_tails.slurm

# Or single shard:
python 2_graphmert/predict_tails_llm.py \
    --model_id    /path/to/qwen3-32b \
    --tokenizer   $OUTPUT_BASE/graphmert/stable_tokenizer \
    --dataset     $OUTPUT_BASE/graphmert/llm_relations/relations_clean_eval \
    --output_dir  $OUTPUT_BASE/graphmert/predictions \
    --num_shards  4 \
    --shard_id    0
```

**Quality check (LLM predictor):** If all tails are generic or hallucinated, check that `--dataset` points to the cleaned relations output from step 2.5.

### 2.6b Step 6b — Predict tails with GraphMERT MLM (optional)

Run the trained GraphMERT checkpoint directly (masked leaf-slot prediction):

```bash
python 2_graphmert/utils/predict_tails.py \
    --model_dir    $OUTPUT_BASE/graphmert/checkpoints/best \
    --tokenizer    $OUTPUT_BASE/graphmert/stable_tokenizer \
    --relation_map $OUTPUT_BASE/graphmert/relation_map.json \
    --dataset      $OUTPUT_BASE/graphmert/llm_relations/relations_clean_eval \
    --output_dir   $OUTPUT_BASE/graphmert/predictions_graphmert \
    --topk         20 \
    --batch_size   8
```

**Quality check (GraphMERT predictor):** Inspect `inspection_preview.txt` in the output dir. If all top-5 predictions are generic tokens (`the`, `of`, `a`), the MLM did not learn meaningful entity representations — retrain before continuing.

### 2.7 Step 7 — Combine tails and two-LLM validation

Merge all shard predictions and deduplicate:

```bash
python 2_graphmert/utils/combine_tails/combine_tails.py \
    --pred_dir    $OUTPUT_BASE/graphmert/predictions \
    --output_dir  $OUTPUT_BASE/graphmert/combined \
    --model_id    /path/to/qwen3-14b
```

Score each candidate triple with two independent LLMs — keep only triples
both models agree are factually supported:

```bash
python 2_graphmert/utils/llm_scores/fact_score.py \
    --input_csv   $OUTPUT_BASE/graphmert/combined/expanded_triples.csv \
    --output_csv  $OUTPUT_BASE/graphmert/final_kg/validated_triples.csv \
    --model_ids   /path/to/model-A /path/to/model-B \
    --batch_size  64 \
    --max_model_len 4096 \
    --tensor_parallel_size 1
# --model_ids requires exactly 2 paths
```

**Output:** `$OUTPUT_BASE/graphmert/final_kg/validated_triples.csv`

**Expected output:** 15,000–50,000 validated triples.

---

## Part 3 — SI Curriculum, SFT & RL

**Environment:** `si_curriculum`

### Pre-step — Compute hop distances

Annotate each triple with its minimum graph-hop distance from the seed KG.
This drives hop-stratified sampling in curriculum generation.

```bash
conda activate si_curriculum
cd $REPO_DIR

python 3_si_curriculum/calculate_hops.py \
    --kg_path      $OUTPUT_BASE/graphmert/final_kg/validated_triples.csv \
    --seed_kg_path $OUTPUT_BASE/graphrag/output/kg_final.parquet \
    --output_path  $OUTPUT_BASE/curriculum/kg_manifest.json
```

**Output:** `$OUTPUT_BASE/curriculum/kg_manifest.json`

### 3.1 Step 1 — Generate Q&A curriculum

Generates multi-hop multiple-choice questions from the KG. Requires a Gemini
API key (set `GOOGLE_API_KEY` env var) or pass `--api_key` directly.

```bash
python 3_si_curriculum/curriculum_generator/generate_curriculum.py \
    --manifest_path  $OUTPUT_BASE/curriculum/kg_manifest.json \
    --output_dir     $OUTPUT_BASE/curriculum \
    --min_hops       3 \
    --max_hops       5 \
    --target_count   50000 \
    --api_key        $GOOGLE_API_KEY \
    --seed           42

# Or via SLURM:
export REPO_DIR OUTPUT_BASE MANIFEST_PATH=$OUTPUT_BASE/curriculum/kg_manifest.json
sbatch 3_si_curriculum/slurm/generate_curriculum.slurm
```

**Output:** `$OUTPUT_BASE/curriculum/curriculum.json`

### 3.2 Step 2 — Two-LLM verification

Keep only Q&A items where two independent LLMs agree on the answer:

```bash
python 3_si_curriculum/curriculum_generator/verify_questions.py \
    --input_json   $OUTPUT_BASE/curriculum/curriculum.json \
    --output_json  $OUTPUT_BASE/curriculum_verified/curriculum_verified.json \
    --model_ids    /path/to/model-A /path/to/model-B \
    --batch_size   64 \
    --tensor_parallel_size 1
# --model_ids requires exactly 2 paths

# Or via SLURM:
export REPO_DIR OUTPUT_BASE
sbatch 3_si_curriculum/slurm/verify_questions.slurm
```

**Output:** `$OUTPUT_BASE/curriculum_verified/curriculum_verified.json`

### 3.3 Step 3 — Prepare SFT dataset

```bash
python 3_si_curriculum/training/data_prep.py \
    --input_file  $OUTPUT_BASE/curriculum_verified/curriculum_verified.json \
    --output_path $OUTPUT_BASE/sft_dataset \
    --model_name  /path/to/base/model \
    --max_length  32768 \
    --cache_dir   $HF_HOME
```

**Output:** HuggingFace dataset saved to `$OUTPUT_BASE/sft_dataset/`

### 3.4 Step 4 — SFT training (LoRA)

`trainer.py` uses HuggingFace argument parsing — all fields can be passed as
`--field_name value` CLI args, or set via env vars (`MODEL_NAME`, `DATASET_PATH`,
`WANDB_DIR`):

```bash
# Multi-GPU with torchrun (recommended):
torchrun --nproc_per_node=4 3_si_curriculum/training/trainer.py \
    --model_name          /path/to/base/model \
    --train_dataset_path  $OUTPUT_BASE/sft_dataset \
    --output_dir          $OUTPUT_BASE/sft_checkpoints \
    --wandb_dir           $OUTPUT_BASE/wandb_logs \
    --wandb_project       neuro_si_sft \
    --lora_r              32 \
    --lora_alpha          64 \
    --lora_dropout        0.05 \
    --block_size          32768

# Or via SLURM (sets MODEL_NAME, DATASET_PATH, OUTPUT_BASE env vars):
export REPO_DIR OUTPUT_BASE
export MODEL_NAME=/path/to/base/model
export DATASET_PATH=$OUTPUT_BASE/sft_dataset
sbatch 3_si_curriculum/slurm/sft_trainer.slurm
```

**Output:** checkpoints in `$OUTPUT_BASE/sft_checkpoints/checkpoint-*/`

### 3.5 Step 5 — Merge LoRA adapter

```bash
python 3_si_curriculum/training/merge_lora.py \
    --base_model   /path/to/base/model \
    --adapter_path $OUTPUT_BASE/sft_checkpoints/checkpoint-XXXX
```

**Output:** `$OUTPUT_BASE/sft_checkpoints/checkpoint-XXXX/merged_final_model/`

### 3.6 Step 6 — Prepare RL dataset

`RL/data_prep.py` is configured at the top of the file via env vars:

```bash
INPUT_PATH=$OUTPUT_BASE/curriculum_verified/curriculum_verified.json \
OUTPUT_PATH=$OUTPUT_BASE/rl_dataset \
python 3_si_curriculum/RL/data_prep.py
# MODE defaults to "rl"; edit top of file to change LAST_N / ENABLE_THINKING
```

**Output:** HuggingFace dataset saved to `$OUTPUT_BASE/rl_dataset/`

### 3.7 Step 7 — RL training (GRPO)

`rl_training.py` uses HuggingFace argument parsing — all fields can be passed
as CLI args:

```bash
python 3_si_curriculum/RL/rl_training.py \
    --model_name              /path/to/sft/merged/model \
    --dataset_path            $OUTPUT_BASE/rl_dataset \
    --output_dir              $OUTPUT_BASE/rl_checkpoints \
    --deepspeed               3_si_curriculum/RL/deepspeed_config.json \
    --learning_rate           8e-7 \
    --beta                    0.12 \
    --num_generations         4 \
    --max_completion_length   1280 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs        10 \
    --save_steps              100 \
    --eval_steps              50 \
    --wandb_project           neuro_si_rl

# Resume from checkpoint:
python 3_si_curriculum/RL/rl_training.py \
    --model_name   /path/to/sft/merged/model \
    --dataset_path $OUTPUT_BASE/rl_dataset \
    --output_dir   $OUTPUT_BASE/rl_checkpoints \
    --deepspeed    3_si_curriculum/RL/deepspeed_config.json
# Set env var to resume: RESUME_CHECKPOINT=$OUTPUT_BASE/rl_checkpoints/checkpoint-YYY

# Or via SLURM:
export REPO_DIR OUTPUT_BASE
export MODEL_NAME=/path/to/sft/merged/model
export DATASET_PATH=$OUTPUT_BASE/rl_dataset
sbatch 3_si_curriculum/slurm/rl_training.slurm
```

**Output:** checkpoints in `$OUTPUT_BASE/rl_checkpoints/checkpoint-*/`

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Empty leaf nodes in GraphMERT | Tokenizer mismatch between steps | Ensure every step uses `--tokenizer $OUTPUT_BASE/graphmert/stable_tokenizer` — the same directory saved by `run_tokenization.py` |
| MLM predicts only generic tokens (`the`, `of`) | Model did not converge | Increase seed KG size; lower learning rate; verify `relation_map.json` has ≥10 relations |
| Zero samples after co-occurrence grounding | Seed KG entities don't appear in corpus | Check entity spelling in KG matches tokenised text; lower co-occurrence threshold |
| `ValueError: output_dir is required` in rl_training | `--output_dir` not set | Pass `--output_dir /path/to/output` explicitly |
| SLURM OOM | Model too large for allocated GPU | Increase `--gres=gpu:2` or reduce `--max_model_len` / `--tensor_parallel_size` |
| `ModuleNotFoundError: No module named 'vllm'` | Wrong conda env | vLLM is in `graphrag` env; use that env for inference scripts |
| `load_metric` import error | Outdated `datasets` version | `pip install --upgrade datasets evaluate` in the graphmert env |

---
## Citations

This pipeline was developed from two projects in the Jha Lab. 

If you use Stage 1 or 2 (GraphMERT code, models, data or data processing scripts) in your work, please cite the following paper:

```bibtex
@article{
    belova2026graphmert,
    title={Graph{MERT}: {E}fficient and Scalable Distillation of Reliable Knowledge Graphs from Unstructured Data},
    author={Margarita Belova and Jiaxin Xiao and Shikhar Tuli and Niraj Jha},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2026},
    url={[https://openreview.net/forum?id=tnXSdDhvqc](https://openreview.net/forum?id=tnXSdDhvqc)},
}
```

If you use Stage 3, please cite the paper below: 

```bibtex

@misc{dedhia2025bottomupdomainspecificsuperintelligencereliable,
      title={Bottom-up Domain-specific Superintelligence: A Reliable Knowledge Graph is What We Need}, 
      author={Bhishma Dedhia and Yuval Kansal and Niraj K. Jha},
      year={2025},
      eprint={2507.13966},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={[https://arxiv.org/abs/2507.13966](https://arxiv.org/abs/2507.13966)},
}

