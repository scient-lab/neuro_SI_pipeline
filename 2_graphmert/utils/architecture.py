"""Single source of truth for the GraphMERT architecture invariants
shared across mlm_utils, dataset_preprocessing_utils, and predict_tails.

These three modules must agree on ROOT_NODES, NUM_LEAVES, and the
resulting MAX_NODES, otherwise tensor shapes mismatch between the
preprocess (dataset_preprocessing_utils), train (mlm_utils), and
inference (predict_tails) stages.

Values resolve in this order:
  1. configs/profiles/<SI_PROFILE>.yaml :: graphmert.config.root_nodes
     and graphmert.config.num_leaves   (operator override per profile)
  2. Fall back to upstream commit (2d7b782) defaults: 512 / 3.

MAX_NODES is always derived from ROOT_NODES * (1 + NUM_LEAVES) — never
read from YAML — so the three values can't accidentally desync.
"""
import os
import sys

# Upstream commit 2d7b782 defaults. Used if pipeline_config import fails
# or the profile YAML omits a graphmert.config block.
_DEFAULT_ROOT_NODES = 512
_DEFAULT_NUM_LEAVES = 3


def _load_graphmert_config() -> dict:
    """Read graphmert.config from the active profile via pipeline_config.
    Falls back to an empty dict on any failure so module import never
    breaks the training process."""
    try:
        utils_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(utils_dir, "..", ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from pipeline_config import get_phase_param  # type: ignore
        cfg = get_phase_param("graphmert", "config", {})
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


_cfg = _load_graphmert_config()
ROOT_NODES = int(_cfg.get("root_nodes", _DEFAULT_ROOT_NODES))
NUM_LEAVES = int(_cfg.get("num_leaves", _DEFAULT_NUM_LEAVES))
MAX_NODES = ROOT_NODES * (1 + NUM_LEAVES)
