"""Single source of truth for the GraphMERT architecture invariants
shared across mlm_utils, dataset_preprocessing_utils, and predict_tails.

These three modules must agree on ROOT_NODES, NUM_LEAVES, and the
resulting MAX_NODES, otherwise tensor shapes mismatch between the
preprocess (dataset_preprocessing_utils), train (mlm_utils), and
inference (predict_tails) stages.

Values match the upstream commit (2d7b782) exactly. In Part 2 of the
config refactor these will become YAML-overridable; for now they stay
as the hardcoded baseline so behaviour is bit-identical to upstream.
"""

ROOT_NODES = 512
NUM_LEAVES = 3
MAX_NODES = ROOT_NODES * (1 + NUM_LEAVES)  # 2048
