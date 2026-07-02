# SFT + RL Phase Wiring Audit (2026-06-22)

## TL;DR
- **10 active bugs** found (will crash on first end-to-end run of curriculum→sft→rl).
  Several are clustered upstream in `curriculum.sh` — that phase will block sft
  from ever getting a verified curriculum, but the sft/rl phases also have
  their own latent crashes once the file does arrive.
- **3 latent bugs** (won't fire today but should be fixed before scaling).
- **4 false alarms** cleared.
- **Phases audited**: sft (prepare_data, train_lora, merge_lora, eval_sft);
  rl (setup_reward, train_grpo, eval_rl); plus the immediate upstream feeder
  steps in curriculum (generate_qa, validate_qa) because SFT/RL both read
  `curriculum_verified.json` and the producers don't currently write it.

---

## Active bugs (will crash)

### curriculum.generate_qa: `QAGenerator.generate_from_path()` does not exist
- **File:line(s)**: producer `3_si_curriculum/curriculum_generator/generate_curriculum.py:154`
  calls `generator.generate_from_path(path_data)`; consumer class
  `3_si_curriculum/curriculum_generator/generate_questions.py:597-672` defines
  `QAGenerator` with methods `generate_questions`, `quality_filtering`,
  `generate_thinking_trace`, `correctness_filtering`, and
  `combine_question_and_thinking_trace_with_answer` — but **no
  `generate_from_path`**.
- **Problem**: First iteration of the `while len(results) < target_count`
  loop raises `AttributeError`. No `curriculum.json` is ever written. SFT
  and RL phases will fail at their input-file load.
- **Suggested fix**: either rename to an existing method or implement
  `generate_from_path(path_data)` that wraps the existing 4-step pipeline
  (`generate_question` → `quality_filtering` → `generate_thinking_trace` →
  `correctness_filtering`) seen in `generate_questions.py:702-749`.

### curriculum.generate_qa: `QAGenerator(api_key=...)` is missing required `kg_dir`
- **File:line(s)**: producer `3_si_curriculum/curriculum_generator/generate_curriculum.py:127`
  `QAGenerator(api_key=args.api_key)`; consumer
  `3_si_curriculum/curriculum_generator/generate_questions.py:597-608` requires
  `kg_dir` (raises `ValueError` if both arg and `KG_DIR` env var are unset).
- **Problem**: `curriculum.sh` never sets `KG_DIR`. Even before the
  `generate_from_path` AttributeError, `QAGenerator()` instantiation raises
  `ValueError: kg_dir (path to final_kg directory) is required.`
- **Suggested fix**: have `curriculum.sh` export `KG_DIR=$GRAPHMERT_DIR/final_kg`
  (or wherever `vocab.txt` / `neuro_graph.pickle` / `vocab_freq.json` get
  written upstream), then pass `--kg_dir "$KG_DIR"` through the CLI. Or
  audit whether those three files actually exist anywhere in the
  end-to-end pipeline today — I couldn't find a producer.

### curriculum.validate_qa: phase script reads `curriculum.json`, producer writes timestamped `curriculum_dataset_hop_*_to_*_*.json`
- **File:line(s)**: producer `3_si_curriculum/curriculum_generator/generate_curriculum.py:134-138`
  writes `curriculum_dataset_hop_{min_hops}_to_{max_hops}_{timestamp}.json`;
  consumer `scripts/phases/curriculum.sh:95` sets
  `INPUT="$CURRICULUM_DIR/curriculum.json"` for `verify_questions.py`.
- **Problem**: Same family of bug as the `expanded_triples.csv` →
  `final_kg_scientific_only.csv` mismatch already fixed in graphmert. The
  filename in the producer has both a hop-range stem and a timestamp;
  the consumer hardcodes a single filename that never exists.
- **Suggested fix**: either rename the producer's output to a stable
  `curriculum.json`, or have the shell script glob the producer's actual
  pattern (`ls -t "$CURRICULUM_DIR"/curriculum_dataset_hop_*.json | head -1`).

### sft.prepare_data: `data_prep.py` saves a flat Dataset; `trainer.py` expects a DatasetDict with `train`/`test`
- **File:line(s)**: producer `3_si_curriculum/training/data_prep.py:82-88` does
  `Dataset.from_dict(...)` then `dataset.save_to_disk(...)` — no train/test
  split; consumer `3_si_curriculum/training/trainer.py:321`
  `dataset = load_from_disk(...)` then `dataset['train']` /
  `dataset['test']` at lines 333-343, 406-407.
- **Problem**: `Dataset.__getitem__("train")` is a column lookup, not a
  split lookup. On a flat Dataset this raises `KeyError: "train"` (or
  worse, returns a column of input_ids and silently feeds the wrong shape
  into SFTTrainer).
- **Suggested fix**: have `data_prep.py` build a `DatasetDict({"train":...,
  "test":...})` with a small held-out slice (parallels what
  `RL/data_prep.py::prepare_sft_dataset` already does), or rewrite
  `trainer.py` to handle a flat Dataset and split it itself.

### sft.prepare_data: dataset has no `text` column, but `trainer.py` sets `dataset_text_field='text'`
- **File:line(s)**: producer `3_si_curriculum/training/data_prep.py:82-85`
  writes columns `input_ids` and `attention_mask` only; consumer
  `3_si_curriculum/training/trainer.py:375`
  `args.dataset_text_field = 'text'`.
- **Problem**: TRL `SFTTrainer` with `dataset_text_field='text'` will look
  up a `text` column on each example to apply its own formatting; not
  present → KeyError. Also conflicts with the intent of pre-tokenized
  data (where you'd typically pass `packing=False` and no text field).
- **Suggested fix**: either save the formatted string in a `text` column
  in `data_prep.py` and drop the pre-tokenization (let TRL tokenize), or
  unset `dataset_text_field` in `trainer.py` and rely on the custom
  collator + pre-tokenized inputs.

### sft.train_lora: trainer looks for DeepSeek `<｜Assistant｜>` token; default model is Qwen3-14B
- **File:line(s)**: producer `configs/default.yaml:52` `base_sft: Qwen/Qwen3-14B`;
  consumer `3_si_curriculum/training/trainer.py:183-221`
  `_resolve_response_template_ids(tokenizer)` hardcodes the fullwidth
  `<｜Assistant｜>` (U+FF5C) special token used by `DeepSeek-R1-0528-Qwen3-8B`.
- **Problem**: Qwen3-14B's tokenizer uses `<|im_start|>assistant<|im_end|>`
  not `<｜Assistant｜>`. All three fallback paths in
  `_resolve_response_template_ids` will miss; the final `raise ValueError`
  at line 218-221 fires. trainer never gets past the collator setup.
- **Suggested fix**: dispatch the response-template lookup on the
  tokenizer family (Qwen vs. DeepSeek) — or, more simply, search for
  whichever assistant-turn token the loaded tokenizer actually defines
  (`tokenizer.chat_template` is the source of truth). Alternative: align
  `data_prep.py` chat template and `_resolve_response_template_ids`
  around a single chosen tokenizer.

### sft.train_lora: chat-template / model-family mismatch in `data_prep.py`
- **File:line(s)**: `3_si_curriculum/training/data_prep.py:42-46` uses Qwen
  ChatML format (`<|im_start|>system... <|im_end|>`) — but the
  response-template lookup in `trainer.py:183-221` (above) expects
  DeepSeek's fullwidth pipe delimiter.
- **Problem**: Sister bug to the one above — even after fixing the
  trainer's hardcoded token, the formatted text in the dataset has to
  use the same chat template the base model's tokenizer expects, or
  collator masking will silently mismatch (mask too much or too little,
  silently destroying the SFT signal — much worse than a crash).
- **Suggested fix**: build the chat string via `tokenizer.apply_chat_template`
  rather than a hand-written `CHAT_TEMPLATE` constant, so it tracks the
  active base model.

### rl.train_grpo: `--model_name` is wired to a field the loader never reads
- **File:line(s)**: producer `scripts/phases/rl.sh:63` passes
  `--model_name "$SFT_MERGED_MODEL"`; consumer
  `3_si_curriculum/RL/rl_training.py:91-94` declares both `model_name`
  and `sft_checkpoint_path` fields, but line 589 sets
  `merged_path = config.sft_checkpoint_path` and line 624
  `AutoModelForCausalLM.from_pretrained(merged_path, ...)` only uses
  `sft_checkpoint_path`. `config.model_name` is read nowhere.
- **Problem**: rl.sh sets `--model_name`, but rl_training reads
  `--sft_checkpoint_path` (defaults to `""`). `from_pretrained("")` →
  `OSError` / "Can't load tokenizer for ''."
- **Suggested fix**: either change `rl.sh:63` to
  `--sft_checkpoint_path "$SFT_MERGED_MODEL"`, or rewrite
  `rl_training.py:589` to prefer `config.model_name` (whichever has a
  non-empty value).

### rl.train_grpo: `--dataset_path` defaults to a placeholder path
- **File:line(s)**: producer `scripts/phases/rl.sh:64` passes
  `--dataset_path "$RL_DATASET_DIR"` correctly; consumer
  `3_si_curriculum/RL/rl_training.py:98`
  `dataset_path: str = field(default="/path/to/your/rl_dataset")`.
- **Problem**: the wiring itself is fine — but if anyone runs
  `python rl_training.py` directly without `--dataset_path`, they hit a
  silent default of `/path/to/your/rl_dataset` instead of an error. The
  shell phase passes it, so this won't crash via `rl.sh`. Borderline
  active vs. latent; flagging as latent below would also be reasonable.
  Listing here because `test_rl.py:371-375` raises `ValueError` on the
  same kind of unset path but `rl_training.py:582-583` only checks
  `output_dir`.
- **Suggested fix**: add the same `if not config.dataset_path: raise
  ValueError(...)` block at line 582 that already exists for
  `output_dir`. (Will not affect the pipeline.sh path; only protects
  ad-hoc runs.)

### rl.setup_reward: `data_prep.py` returns flat `Dataset` in RL mode; `preprocess_grpo_dataset` re-loads and re-processes the SAME data
- **File:line(s)**: producer `scripts/phases/rl.sh:50-52` sets
  `INPUT_PATH`/`OUTPUT_PATH` and runs `RL/data_prep.py` which (in RL
  mode) loads JSON, slices, saves a `DatasetDict({"train": full_ds})` to
  `OUTPUT_PATH + "_tmp_sliced"`, runs `preprocess_grpo_dataset` on it,
  removes the tmp, and saves the processed flat Dataset to `OUTPUT_PATH`
  (`3_si_curriculum/RL/data_prep.py:282-307`).  Then consumer
  `rl_training.py:603-610` calls `preprocess_grpo_dataset` **again** on
  the same `dataset_path`.
- **Problem**: `preprocess_grpo_dataset` reads
  `loaded["question_and_explanation"]` (line 206), but the *first*
  invocation in `data_prep.py:296-306` has already transformed those rows
  into `{"prompt", "answer", "paths"}` and discarded
  `question_and_explanation`. So the second call inside `rl_training.py`
  raises `KeyError: 'question_and_explanation'` (the second
  `process_batch` will not find that column).
- **Suggested fix**: either skip the pre-processing step in rl.sh and
  have rl_training do it once, or change the setup_reward step to write
  the raw sliced dataset and let rl_training do the full
  `preprocess_grpo_dataset` call. (Current setup duplicates work and
  guarantees a missing-column error.)

---

## Latent bugs (defensive code-quality fixes)

### rl: `path_alignment_thinking_tokens` config knob is silently overridden by hardcoded 550
- **File:line(s)**: `3_si_curriculum/RL/rl_training.py:248` reads
  `PATH_ALIGNMENT_THINKING_TOKENS` from config; but
  `path_alignment_reward_func` at line 371 calls
  `truncate_thinking_for_coverage(thinking, max_tokens=550)` with a
  hardcoded `550` (and similarly `correctness_reward_func` at line 315
  hardcodes `soft_start=550, hard_cap=1500`).
- **Problem**: This is bug pattern #5 from the audit checklist — a knob
  declared in YAML (`configs/default.yaml:186-188`
  `length_penalty_soft_start`, `length_penalty_hard_cap`,
  `path_alignment_thinking_tokens`) that's read into module-level
  constants but then ignored at the call site. A profile that sets
  these to different values will silently get the 550/1500/550 hardcodes
  instead. Tunability theater.
- **Suggested fix**: replace the literal `550` / `1500` in
  `correctness_reward_func` and `path_alignment_reward_func` with the
  module-level constants (which already exist).

### rl: `rl.algorithm` and `rl.reward_source` YAML knobs have no consumer
- **File:line(s)**: declared in `configs/default.yaml:165` (`algorithm:
  grpo`) and `:178` (`reward_source: kg_path_alignment`); no
  `get_phase_param('rl', 'algorithm', ...)` or `'reward_source'` call
  exists anywhere in `3_si_curriculum/RL/`.
- **Problem**: bug pattern #5 again — config keys declared but never
  read. Anyone editing them expects behavior change; gets nothing.
- **Suggested fix**: either consume them (gate the trainer on
  `algorithm`, switch reward functions on `reward_source`), or delete
  them from default.yaml. Documentation-only purpose should be made
  explicit in a comment.

### sft.merge_lora: forces CPU device map; comment claims "400 GB SLURM RAM"
- **File:line(s)**: `3_si_curriculum/training/merge_lora.py:16-21`
  `device_map={"": "cpu"}` with a comment "We rely on your 400GB of
  SLURM system RAM instead of the GPUs."
- **Problem**: not a wiring bug, but RunPod single-pod runs (the
  documented smoke/pilot scale per `RUNPOD_QUICKREF.md`) have nowhere
  near 400 GB of host RAM. Qwen3-14B in bf16 (~28 GB) plus a LoRA
  adapter merge probably fits, but the merge step risks OOMing on
  smaller pods (32-64 GB RAM tier) for any larger base model. Latent
  because it works for the current default; would bite when scaling.
- **Suggested fix**: make `device_map` profile-driven (CPU on SLURM,
  `"auto"` on workstations with enough VRAM) — or at least add an
  env-var override.

---

## False alarms (audited, cleared)

- **rl.train_grpo `vllm.LLM(model=...)` check (bug pattern #4)**: the only
  `LLM(model=...)` callsite in `3_si_curriculum/` scope is in
  `curriculum_generator/verify_questions.py:88` (validate_qa step). It uses
  `model=model_id` where `model_id` is one of the two args passed via
  `--model_ids` (sourced from `curriculum_check_a` / `curriculum_check_b`
  in `configs/default.yaml:48-49`, both `Qwen/Qwen3-14B` and
  `mistralai/Mistral-Nemo-Instruct-2407` — valid causal LMs, valid repos,
  no mxfp4 quantization). Neither SFT nor RL training calls
  `vllm.LLM(model=...)` directly.

- **HF repo id check (bug pattern #6)**: `base_sft: Qwen/Qwen3-14B`
  (valid, canonical), `curriculum_check_a: Qwen/Qwen3-14B` (valid),
  `curriculum_check_b: mistralai/Mistral-Nemo-Instruct-2407` (valid —
  not the bogus `Mistral-Nemo-12B`). No model ids in the SFT/RL hot path
  are 404.

- **mxfp4 quantization check (bug pattern #7)**: the only formerly-mxfp4
  ids (`openai/gpt-oss-20b`) are documented as swapped out in
  `configs/default.yaml:21-25, 45-48`. No remaining model id in `models:`
  is an mxfp4 build.

- **Function-local import shadowing (bug pattern #3)**: every `.py` file
  in `3_si_curriculum/training/` and `3_si_curriculum/RL/` imports
  `from pipeline_config` exactly once (at module top). No shadow risk
  in scope.

---

## Items not yet verified (out of scope or needs runtime data)

- **End-to-end shape of `curriculum_verified.json`** vs. what
  `training/data_prep.py` reads (`item["question"]`,
  `item.get("thinking_trace", item.get("explanation"))`,
  `item["answer"]`) vs. what `RL/data_prep.py` reads
  (`example["question_and_explanation"]`, optional `example["paths"]`).
  These two consumers want different schemas of the SAME file, but I
  cannot verify the producer's actual output until the
  `generate_from_path` AttributeError above is fixed and a real run
  emits one. Strong likelihood of a downstream schema-mismatch bug — see
  `RL/data_prep.py:106-141` (the `to_messages_format` fallback at
  line 142-149 will silently mask malformed data).

- **`KG_DIR` provenance**: I could find no producer of `vocab.txt`,
  `neuro_graph.pickle`, or `vocab_freq.json` anywhere in the
  orchestration scripts. These are required by `QAGenerator.__init__`
  (`generate_questions.py:609-614`). Either upstream graphmert is
  expected to write them and currently doesn't, or they live in a
  side-loaded artifact that the runbook needs to call out. Either way,
  curriculum will fail before SFT even runs.

- **`merge_final_model/` directory layout**: `sft.sh:42` finds the
  merged model via `ls -d "$OUTPUT_BASE/sft_checkpoints"/checkpoint-*/merged_final_model`.
  `merge_lora.py:32` writes `merged_output_dir = os.path.join(adapter_path,
  "merged_final_model")`, and `adapter_path` is whatever
  `sft.sh:92` resolves (last `checkpoint-*` dir). The path agreement is
  fine as long as the trainer actually emits a `checkpoint-*` directory;
  with `save_strategy="no"` (trainer.py:367) it relies on
  `EpochCheckpointCallback` to emit `checkpoint-epoch-{n}/`. So sft.sh's
  glob `checkpoint-*` will catch `checkpoint-epoch-0/`, `checkpoint-epoch-1/`,
  etc. — that's OK. But sft.sh `tail -1` picks lexicographically last,
  not numerically last; `checkpoint-epoch-10` sorts before
  `checkpoint-epoch-2` (string order). This is a real bug but only
  triggers at >=10 epochs (smoke uses 1, pilot uses 3, so it's latent).
  Flag for paper-scale runs.

- **`save_safetensors = False` (trainer.py:374)** — produces `.bin`
  weights. `merge_lora.py:36` does `save_pretrained(merged_output_dir,
  safe_serialization=True)` which writes `.safetensors`. Then rl_training
  loads from the merged dir at `rl_training.py:625` — should work
  regardless of input/output serialization (each save_pretrained call
  picks one format independently). No bug, but worth noting these are
  intentionally heterogeneous.

---

## Clean steps (audited, no issues found)

- **sft.eval_sft** — explicit no-op (`sft.sh:105-107`); operator runs
  `test_models/eval_models.py` separately. No wiring to audit.

- **rl.eval_rl** — same explicit no-op (`rl.sh:71-73`).

- **rl.train_grpo deepspeed config wiring** — `rl.sh:45` resolves
  `DEEPSPEED_CFG` correctly to
  `3_si_curriculum/RL/deepspeed_config.json` (which exists); passes via
  `--deepspeed`; `rl_training.py:105` declares the field and line 585-586
  absolutizes the path. Wired cleanly.

- **rl: `RL_DO_EVAL` / `RL_NUM_EVAL_EXAMPLES` env-var overrides**
  (`rl_training.py:54-55`) — read once at module load; documented; do
  not conflict with the dataclass CLI surface. Clean.

- **sft.train_lora wandb wiring** — `sft.sh:71-72, 82-83` passes
  `--wandb_dir` and `--wandb_project`; `trainer.py:52-53, 75-76`
  consumes them via the dataclass + `__post_init__`. Clean.

- **curriculum.validate_qa `verify_questions.py` CLI** —
  `curriculum.sh:99-104` passes `--input_json`, `--output_json`,
  `--model_ids`, `--batch_size`, `--tensor_parallel_size`. Consumer
  `verify_questions.py:47-60` declares all five exactly. Clean (modulo
  the upstream input-file issue logged above).

- **`pipeline_config.get_phase_param` lookups** — the keys SFT/RL code
  reads (`block_size`, `lora_r`, `lora_alpha`, `lora_dropout`,
  `dataloader_num_workers`, `dataloader_prefetch_factor`,
  `learning_rate`, `beta`, `num_generations`, `max_completion_length`,
  `per_device_train_batch_size`, `gradient_accumulation_steps`,
  `num_train_epochs`, `max_grad_norm`, `eval_size`, `eval_steps`,
  `save_steps`, `generation_dump_every`, `generation_temperature`,
  `generation_top_p`, `generation_repetition_penalty`,
  `length_penalty_soft_start`, `length_penalty_hard_cap`,
  `path_alignment_thinking_tokens`) all have matching entries in
  `configs/default.yaml::sft` or `::rl`. Profile yamls
  (`smoke.yaml`, `pilot.yaml`) use the same names. No misspellings
  (modulo the hardcode-override bug logged in Latent above).
