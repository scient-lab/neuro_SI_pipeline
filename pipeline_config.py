"""pipeline_config - config loader for the neuro_SI orchestration pipeline.

All configuration lives in this repo:

    configs/default.yaml                  - operational defaults
    configs/profiles/<profile>.yaml       - scaling profile overrides
    configs/platforms/<platform>.yaml     - platform-specific overrides
    domains/<domain>.yaml                  - domain vocabulary + few-shot + focus
    prompts/<phase>.yaml                   - LLM prompt templates

The loader walks the layers at runtime and returns a merged dict.

Environment variables (all optional):

    SI_DOMAIN     - which domain (default: 'neuroscience').
    SI_PROFILE    - scaling profile (smoke / pilot / paper). Optional; if unset,
                     no profile layer is applied.
    SI_PLATFORM   - platform (local / runpod / aws / princeton). Optional.

API:

    load_config()                                 -> dict   (merged)
    get_relations()                               -> list[str]
    get_relation_descriptions()                   -> dict[str, str]
    get_entity_categories()                       -> list[str]
    get_entity_category_descriptions()            -> dict[str, str]
    get_few_shot_examples()                       -> list[dict]
    get_focus_instructions()                      -> str
    get_domain()                                  -> str
    get_model_id(key, default=None)               -> str | None
    get_phase_param(phase, key, default=None)     -> Any
    get_platform_param(key, default=None)         -> Any
    get_prompt(name)                              -> dict
    describe(prefix='  ')                          -> str   (diagnostic)
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml


_REPO_ROOT = Path(__file__).resolve().parent


# --- Env resolution ------------------------------------------------------

def get_si_home() -> Path:
    return _REPO_ROOT


def get_domain_name() -> str:
    return os.environ.get("SI_DOMAIN", "neuroscience").strip()


def get_profile_name() -> Optional[str]:
    name = os.environ.get("SI_PROFILE", "").strip()
    return name or None


def get_platform_name() -> Optional[str]:
    name = os.environ.get("SI_PLATFORM", "").strip()
    return name or None


# --- Merge primitives ----------------------------------------------------

def _read(p: Path) -> dict[str, Any]:
    if not p.is_file():
        return {}
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p} must parse to a dict; got {type(data).__name__}")
    return data


def _merge(base: Any, overlay: Any) -> Any:
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        return overlay
    out = dict(base)
    for k, v in overlay.items():
        out[k] = _merge(base.get(k), v) if k in base else v
    return out


# --- Layered config load --------------------------------------------------

@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Merge the config layers in order (later wins on key conflict):

      1. configs/default.yaml
      2. domains/<SI_DOMAIN>.yaml
      3. configs/profiles/<SI_PROFILE>.yaml      (if SI_PROFILE set)
      4. configs/platforms/<SI_PLATFORM>.yaml    (if SI_PLATFORM set)
    """
    home = _REPO_ROOT
    domain = get_domain_name()

    cfg: dict[str, Any] = {}
    cfg = _merge(cfg, _read(home / "configs" / "default.yaml"))
    cfg = _merge(cfg, _read(home / "domains" / f"{domain}.yaml"))

    profile = get_profile_name()
    if profile:
        cfg = _merge(cfg, _read(home / "configs" / "profiles" / f"{profile}.yaml"))

    platform = get_platform_name()
    if platform:
        cfg = _merge(cfg, _read(home / "configs" / "platforms" / f"{platform}.yaml"))

    return cfg


# --- Effective-config ledger (runtime self-recording) --------------------
# Every value resolved through the accessors below is recorded as
# name -> {value, source-layer}; at process exit it is flushed to
#     $OUTPUT_BASE/config/<SI_PHASE>.<SI_STEP>.yaml
# so each pipeline STEP writes the config it ACTUALLY used — post-merge and
# post-env-override — which a static snapshot of the input yaml cannot show.
# `source: fallback` means the key was absent from EVERY config layer and the
# hard-coded default arg won (the profile-key-name-trap tell). Push-only
# provenance: it lives under outputs/, so the existing outputs->S3 sync carries
# it; the repo's configs/ stay the single source of truth (never pulled over).

import atexit as _atexit

_MISSING = object()
_LEDGER: dict[str, dict[str, Any]] = {"models": {}, "params": {}, "prompts": {}}
_LAYER_PRECEDENCE = ("platform", "profile", "domain", "default")  # high -> low


@lru_cache(maxsize=None)
def _raw_layer(name: str) -> dict[str, Any]:
    """One UN-merged config layer, read for source attribution."""
    home = _REPO_ROOT
    if name == "default":
        return _read(home / "configs" / "default.yaml")
    if name == "domain":
        return _read(home / "domains" / f"{get_domain_name()}.yaml")
    if name == "profile":
        p = get_profile_name()
        return _read(home / "configs" / "profiles" / f"{p}.yaml") if p else {}
    if name == "platform":
        pf = get_platform_name()
        return _read(home / "configs" / "platforms" / f"{pf}.yaml") if pf else {}
    return {}


def _source_of(*path: str) -> str:
    """Highest-precedence layer file that defines <path> (e.g.
    ('models','extract') or ('graphmert','fact_score_max_tokens')); 'fallback'
    if no layer has it (so the hard-coded default arg won)."""
    for layer in _LAYER_PRECEDENCE:
        node: Any = _raw_layer(layer)
        for k in path:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                node = _MISSING
                break
        if node is not _MISSING:
            return layer
    return "fallback"


def _record(bucket: str, name: str, value: Any, source: str) -> None:
    _LEDGER[bucket][name] = {"value": value, "source": source}


def _git_sha() -> str:
    sha = os.environ.get("SI_GIT_SHA") or os.environ.get("GIT_SHA")
    if sha:
        return sha.strip()
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def _now_iso() -> str:
    try:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    except Exception:
        return ""


def _flush_ledger() -> None:
    """Write the keys THIS process read to $OUTPUT_BASE/config/<phase>.<step>.yaml.
    No-op unless something was read AND we're inside a pipeline step (SI_PHASE
    set) — so ad-hoc imports / the monitor's quality probe don't litter. Never
    raises: provenance must not break a run."""
    if not any(_LEDGER.values()):
        return
    phase = os.environ.get("SI_PHASE", "").strip()
    if not phase:
        return
    step = os.environ.get("SI_STEP", "").strip()
    base = os.environ.get("OUTPUT_BASE", "").strip() or str(_REPO_ROOT / "outputs")
    out_dir = Path(base) / "config"
    path = out_dir / (f"{phase}.{step}.yaml" if step else f"{phase}.yaml")
    run_id = os.environ.get("RUN_ID", "")
    env_keys = ("SI_DOMAIN", "SI_PROFILE", "SI_PLATFORM", "RUN_ID", "CORPUS_PATH",
                "OUTPUT_BASE", "SI_PHASE", "SI_STEP")
    doc: dict[str, Any] = {
        "_meta": {
            "run_id": run_id, "phase": phase, "step": step,
            "domain": get_domain_name(), "profile": get_profile_name() or "",
            "platform": get_platform_name() or "", "git_sha": _git_sha(),
            "written_at": _now_iso(), "pid": os.getpid(),
        },
        "env": {k: os.environ[k] for k in env_keys if k in os.environ},
        "models": dict(_LEDGER["models"]),
        "params": dict(_LEDGER["params"]),
        "prompts": dict(_LEDGER["prompts"]),
    }
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Union keys with an existing SAME-run file, so the several python
        # processes a step may spawn don't clobber each other (last value wins).
        if path.is_file():
            prev = _read(path)
            if prev.get("_meta", {}).get("run_id") == run_id:
                for b in ("models", "params", "prompts"):
                    m = dict(prev.get(b, {}) or {})
                    m.update(doc[b])
                    doc[b] = m
                if isinstance(prev.get("env"), dict):
                    e = dict(prev["env"])
                    e.update(doc["env"])
                    doc["env"] = e
        with open(path, "w") as f:
            yaml.safe_dump(doc, f, default_flow_style=False, sort_keys=True)
    except Exception:
        pass  # provenance must never break a run


_atexit.register(_flush_ledger)


# --- Vocabulary helpers ---------------------------------------------------

def get_domain(default: Optional[str] = None) -> Optional[str]:
    return load_config().get("domain", default)


def get_relations() -> list[str]:
    return [r["id"] if isinstance(r, dict) else r for r in load_config().get("relations", [])]


def get_relation_descriptions() -> dict[str, str]:
    out: dict[str, str] = {}
    for r in load_config().get("relations", []):
        if isinstance(r, dict):
            out[r["id"]] = r.get("description", "")
    return out


def get_entity_categories() -> list[str]:
    return [c["id"] if isinstance(c, dict) else c for c in load_config().get("entity_categories", [])]


def get_entity_category_descriptions() -> dict[str, str]:
    out: dict[str, str] = {}
    for c in load_config().get("entity_categories", []):
        if isinstance(c, dict):
            out[c["id"]] = c.get("description", "")
    return out


def get_few_shot_examples() -> list[dict[str, Any]]:
    return load_config().get("few_shot_examples", []) or []


def get_focus_instructions() -> str:
    return load_config().get("focus_instructions", "") or ""


# --- Per-phase domain content (consumed by prompts/<phase>.yaml templates) ---
# Each of these returns a string that's substituted into the corresponding
# {{slot}} in the prompt template. They are read from domains/<SI_DOMAIN>.yaml
# so cross-domain swaps (physics, biomed, finance) are YAML-only.

def get_relation_meanings() -> str:
    """Render-ready relation semantics block for prompts (one bullet per
    relation). Source: domains/<name>.yaml::relation_meanings (free text).
    Used by add_llm_relations and combine_tails prompts.
    """
    return str(load_config().get("relation_meanings", "") or "")


def get_relation_examples() -> str:
    """Render-ready set of (head | relation | tail) example lines.
    Source: domains/<name>.yaml::relation_examples (free text).
    Used by add_llm_relations prompts.
    """
    return str(load_config().get("relation_examples", "") or "")


def get_predict_tails_examples() -> list[dict[str, Any]]:
    """Few-shot examples for the predict_tails prompt. Each item is a
    dict with text/head/relation/json keys.
    Source: domains/<name>.yaml::predict_tails_examples.
    """
    items = load_config().get("predict_tails_examples", []) or []
    return [it for it in items if isinstance(it, dict)]


def get_combine_tails_examples() -> list[dict[str, str]]:
    """Few-shot examples for the combine_tails prompt. Each item is a
    dict with user/assistant keys (raw assistant message text).
    Source: domains/<name>.yaml::combine_tails_examples.
    """
    items = load_config().get("combine_tails_examples", []) or []
    return [it for it in items if isinstance(it, dict)]


def get_fact_score_scope() -> str:
    """Domain scope text inserted into the fact-score validity prompt.
    e.g. 'brain regions, cell types, ...'.
    Source: domains/<name>.yaml::fact_score_scope.
    """
    return str(load_config().get("fact_score_scope", "") or "")


def get_relation_examples_block() -> str:
    """Raw head|relation|tail example block for add_llm_relations prompt.
    Source: domains/<name>.yaml::relation_examples_block.
    """
    return str(load_config().get("relation_examples_block", "") or "")


def get_relations_allowed_block() -> str:
    """Raw allowed-relations list (quoted, comma-separated, indented) for
    add_llm_relations prompt. Source: domains/<name>.yaml::relations_allowed_block.
    Distinct from get_relations() (which returns the broader 40-relation
    domain vocab) — this is the 29-subset specifically used in the
    add_llm_relations system prompt.
    """
    return str(load_config().get("relations_allowed_block", "") or "")


def get_relation_meanings_detailed() -> str:
    """Detailed (didactic-phrasing) relation semantics block for the
    add_llm_relations prompt. DIFFERENT from get_relation_meanings()
    (combine_tails-style). Source: domains/<name>.yaml::relation_meanings_detailed.
    """
    return str(load_config().get("relation_meanings_detailed", "") or "")


def get_add_llm_relations_examples() -> list[dict[str, str]]:
    """Few-shot in-context dialogs (user / assistant / explanation triples)
    for add_llm_relations. Source: domains/<name>.yaml::add_llm_relations_examples.
    """
    items = load_config().get("add_llm_relations_examples", []) or []
    return [it for it in items if isinstance(it, dict)]


def get_domain_expert_role() -> str:
    """The persona phrase used in eval / SFT / RL system prompts
    ("You are an {{domain_expert_role}}."). Source:
    domains/<name>.yaml::domain_expert_role. Falls back to
    "expert {{domain}}" if absent so neutral pipelines still produce a
    sensible string.
    """
    cfg = load_config()
    explicit = cfg.get("domain_expert_role")
    if isinstance(explicit, str) and explicit.strip():
        return explicit
    return f"expert {get_domain_name()}"


# --- Operational helpers --------------------------------------------------

def get_model_id(key: str, default: Optional[str] = None) -> Optional[str]:
    """cfg['models'][key] or default. e.g. get_model_id('extract')."""
    val = (load_config().get("models") or {}).get(key, default)
    _record("models", key, val, _source_of("models", key))
    return val


def get_phase_param(phase: str, key: str, default: Any = None) -> Any:
    """cfg[<phase>][<key>] or default."""
    val = (load_config().get(phase) or {}).get(key, default)
    _record("params", f"{phase}.{key}", val, _source_of(phase, key))
    return val


def get_platform_param(key: str, default: Any = None) -> Any:
    """cfg['platform'][<key>] or default."""
    val = (load_config().get("platform") or {}).get(key, default)
    _record("params", f"platform.{key}", val, _source_of("platform", key))
    return val


# --- Exception / retry semantics loader -----------------------------------

@lru_cache(maxsize=1)
def _load_exceptions_yaml() -> dict[str, Any]:
    """Read configs/exceptions.yaml once per process. Empty dict if missing —
    callers fall back to in-code defaults (don't fail the run just because
    the file is absent on an unbootstrapped checkout).
    """
    path = _REPO_ROOT / "configs" / "exceptions.yaml"
    if not path.exists():
        return {}
    return _read(path)


def get_exception_config(library: str) -> dict[str, Any]:
    """Library-binding retry semantics from configs/exceptions.yaml.

    Returns a dict with keys:
        transient_markers:     list[str]  (substring-matched against str(e).lower())
        initial_delay_seconds: float      (first retry wait; doubles each retry)
        max_retries:           int        (retry count cap)
    Missing keys → empty dict (caller must supply its own defaults).

    Example:
        cfg = get_exception_config('gemini')
        markers = tuple(str(m).lower() for m in cfg.get('transient_markers', []))

    FUTURE: this whole helper is obsoleted by LiteLLM adoption — see the
    "FUTURE: LITELLM MIGRATION" block in configs/exceptions.yaml. Until
    then, maintain the YAML.
    """
    return _load_exceptions_yaml().get(library, {}) or {}


# --- Prompt loader --------------------------------------------------------

def get_prompt(name: str) -> dict[str, Any]:
    """Return the prompt template for a phase by name.

    Lookup order (first hit wins):
      1. prompts/overrides/<SI_DOMAIN>/<name>.yaml   (per-domain override)
      2. prompts/<name>.yaml                          (canonical)

    Returns the raw template dict. For slot-substituted system + user
    messages ready to send to an LLM, use render_prompt() instead.
    """
    domain = get_domain_name()
    override = _REPO_ROOT / "prompts" / "overrides" / domain / f"{name}.yaml"
    if override.is_file():
        _record("prompts", name, f"prompts/overrides/{domain}/{name}.yaml", "override")
        return _read(override)
    _record("prompts", name, f"prompts/{name}.yaml", "canonical")
    return _read(_REPO_ROOT / "prompts" / f"{name}.yaml")


def render_prompt(name: str, **slots: Any) -> dict[str, Any]:
    """Load prompts/<name>.yaml and return system + user messages with
    {{slot}} placeholders substituted.

    Slots auto-filled from the active domain config (caller-passed kwargs
    override these):
      - {{domain}}              from get_domain_name()
      - {{focus_instructions}}  from get_focus_instructions()
      - {{categories}}          formatted from get_entity_categories()
                                + get_entity_category_descriptions()
      - {{relations}}           formatted from get_relations()
                                + get_relation_descriptions()
      - {{few_shot}}            formatted from get_few_shot_examples()

    Caller-only slots typically include {{text}}, {{head}}, {{relation}},
    etc. — anything not in the auto-fill list.

    Returns a dict:
      {
        "system":     "<system message with slots filled>",
        "user":       "<user message with slots filled>",
        "generation": {temperature, max_tokens, ...},
        "name":       "<prompt name>",
        "phase":      "<phase identifier>",
      }

    Raises FileNotFoundError if prompts/<name>.yaml does not exist — we
    refuse to silently fall back to a hardcoded prompt body because that
    is the exact failure mode (diabetes prompts run on neuroscience text)
    that motivated the YAML migration.
    """
    template = get_prompt(name)
    if not template:
        raise FileNotFoundError(
            f"prompts/{name}.yaml not found or empty. "
            f"Cannot render prompt '{name}'. "
            f"See docs/PROMPT_MIGRATION.md for the inventory of supported names."
        )

    _domain_lower = get_domain_name()
    defaults = {
        "domain": _domain_lower,
        # Title-cased variant for proper-noun positions ("Neuroscience KG"
        # vs "neuroscience text"). Used where capitalization matters
        # — e.g. combine_tails.yaml "{{Domain}} Knowledge Graph curator".
        "Domain": _domain_lower.title(),
        "focus_instructions": get_focus_instructions(),
        "categories": _format_categories(),
        "relations": _format_relations(),
        "few_shot": _format_few_shot(),
        # New phase-specific slots — used by graphmert sub-step prompts
        # (add_llm_relations, combine_tails, predict_tails, fact_score).
        "relation_meanings": get_relation_meanings(),
        "relation_examples": get_relation_examples(),
        "predict_tails_examples": _format_predict_tails_examples(),
        "fact_score_scope": get_fact_score_scope(),
        # Slots specific to add_llm_relations (#4 in PROMPT_MIGRATION):
        "relation_examples_block": get_relation_examples_block(),
        "relations_allowed_block": get_relations_allowed_block(),
        "relation_meanings_detailed": get_relation_meanings_detailed(),
        # Persona phrase used in SFT/RL/eval MCQ prompts (#11/#12/#13/#14).
        "domain_expert_role": get_domain_expert_role(),
        # Extract-phase (#1) content slots — see prompts/extract.yaml and
        # the matching extract_* keys in domains/<SI_DOMAIN>.yaml.
        "extract_kg_topic":             str(load_config().get("extract_kg_topic", "") or ""),
        "extract_entity_types_list":    str(load_config().get("extract_entity_types_list", "") or ""),
        "extract_user_entity_types":    str(load_config().get("extract_user_entity_types", "") or ""),
        "extract_entity_subcategories": str(load_config().get("extract_entity_subcategories", "") or ""),
        "extract_example_text":         str(load_config().get("extract_example_text", "") or ""),
        "extract_example_tuples":       str(load_config().get("extract_example_tuples", "") or ""),
    }
    merged: dict[str, Any] = {**defaults, **slots}

    # Substitute every string-valued top-level key in the template (not
    # just system/user) so prompt files can carry several named sub-prompts
    # — e.g. rl_mcq.yaml has {system, task_instructions}; eval_models.yaml
    # has {system, gemini_system, recovery}. Non-string keys (generation
    # block, lists, etc.) are passed through unchanged.
    rendered: dict[str, Any] = {}
    skip = {"name", "phase", "generation"}
    for key, val in template.items():
        if key in skip:
            continue
        if isinstance(val, str):
            rendered[key] = _substitute(val, merged)
        else:
            rendered[key] = val
    rendered.setdefault("system", "")
    rendered.setdefault("user", "")
    rendered["generation"] = template.get("generation", {}) or {}
    rendered["name"] = template.get("name", name)
    rendered["phase"] = template.get("phase", "")
    return rendered


# --- Slot-formatting helpers ----------------------------------------------
# These render structured domain config into the text form prompts expect.

def _substitute(text: str, slots: dict[str, Any]) -> str:
    """Replace every {{slot_name}} occurrence with str(slots[slot_name]).
    Unknown slots are left as-is so missing values are visible in the
    rendered prompt (and surface as obvious LLM-side errors) rather than
    silently filled with empty string.
    """
    for k, v in slots.items():
        text = text.replace("{{" + k + "}}", str(v) if v is not None else "")
    return text


def _format_categories() -> str:
    """Render entity_categories as 'id: description' bullets."""
    ids = get_entity_categories()
    desc = get_entity_category_descriptions()
    if not ids:
        return ""
    lines = []
    for cat in ids:
        d = desc.get(cat, "")
        lines.append(f"  - {cat}" + (f": {d}" if d else ""))
    return "\n".join(lines)


def _format_relations() -> str:
    """Render relations as 'id: description' bullets."""
    ids = get_relations()
    desc = get_relation_descriptions()
    if not ids:
        return ""
    lines = []
    for rel in ids:
        d = desc.get(rel, "")
        lines.append(f"  - {rel}" + (f": {d}" if d else ""))
    return "\n".join(lines)


def _format_predict_tails_examples() -> str:
    """Render predict_tails_examples (list of dicts with text/head/relation/json)
    into the multi-line text block predict_tails_llm.py's builder produced
    inline before the YAML migration. Bit-identical to the pre-migration
    output so the LLM sees the same prompt across the migration boundary.
    """
    import json as _json
    items = get_predict_tails_examples()
    if not items:
        return ""
    blocks = []
    for ex in items:
        text = ex.get("text", "")
        head = ex.get("head", "")
        relation = ex.get("relation", "")
        j = ex.get("json", {})
        blocks.append(
            f"TEXT: {text}\nHEAD: {head}\nRELATION: {relation}\n"
            f"JSON: {_json.dumps(j) if not isinstance(j, str) else j}"
        )
    return "\n\n".join(blocks)


def _format_few_shot() -> str:
    """Render few_shot_examples as 'head | relation | tail' lines.

    Supports the existing few_shot_examples shape in domains/<name>.yaml
    (list of dicts with head / relation / tail keys); falls back to
    str(example) for entries that aren't dict-shaped.
    """
    examples = get_few_shot_examples()
    if not examples:
        return ""
    lines = []
    for ex in examples:
        if isinstance(ex, dict):
            h = ex.get("head") or ex.get("subject") or ""
            r = ex.get("relation") or ex.get("predicate") or ""
            t = ex.get("tail") or ex.get("object") or ""
            if h or r or t:
                lines.append(f"  {h} | {r} | {t}")
        else:
            lines.append(f"  {ex}")
    return "\n".join(lines)


# --- Diagnostic ----------------------------------------------------------

def describe(prefix: str = "  ") -> str:
    cfg = load_config()
    lines = [
        f"{prefix}SI_HOME         : {_REPO_ROOT}",
        f"{prefix}SI_DOMAIN       : {get_domain_name()}",
        f"{prefix}SI_PROFILE      : {get_profile_name() or '(unset)'}",
        f"{prefix}SI_PLATFORM     : {get_platform_name() or '(unset)'}",
        f"{prefix}categories      : {len(cfg.get('entity_categories') or [])}",
        f"{prefix}relations       : {len(cfg.get('relations') or [])}",
        f"{prefix}few_shot        : {len(cfg.get('few_shot_examples') or [])}",
        f"{prefix}models          : {list((cfg.get('models') or {}).keys())}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe(prefix=""))
