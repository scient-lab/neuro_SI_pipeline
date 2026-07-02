#!/usr/bin/env python3
"""catalog_mermaid.py — render the pipeline as a Mermaid flowchart.

Projection of the two authored sources (both keyed by the frozen <phase>.<step>
ids, per PIPELINE_CATALOG_CONTRACT / DATA_DRIVEN_PIPELINE_EXECUTOR_PLAN):
  configs/pipeline_catalog.yaml    — phases, ordered steps, `kind`, names (TASKS)
  configs/pipeline_execution.yaml  — each step's `output` file/dir (+ venv/entrypoint)

Layout: each phase is a horizontal SWIMLANE (subgraph, `direction LR`) with its
steps flowing left→right. A step's output artifact is the LABEL ON THE EDGE to
the next step (task -->|file/dir| next task) — no separate node, no special
shape, renders on any Mermaid. Phases connect box-to-box (subgraph→subgraph) so
the parent TB direction can't flatten the lanes.

Usage:
  uv run --with pyyaml python scripts/lib/catalog_mermaid.py                 # mermaid → stdout
  uv run --with pyyaml python scripts/lib/catalog_mermaid.py --md            # fenced for markdown
  uv run --with pyyaml python scripts/lib/catalog_mermaid.py --doc --out docs/PIPELINE_DIAGRAM.md
"""
import argparse
import sys
from pathlib import Path

# kind -> (label emoji, mermaid classDef style). Enum per the catalog _schema.
KIND_STYLE = {
    "llm":       ("🤖", "fill:#e8f0fe,stroke:#4285f4,color:#111"),
    "graphmert": ("🕸", "fill:#fce8f3,stroke:#d81b8c,color:#111"),
    "train":     ("🎯", "fill:#fef7e0,stroke:#f9ab00,color:#111"),
    "transform": ("🔧", "fill:#e6f4ea,stroke:#34a853,color:#111"),
    "deploy":    ("🚀", "fill:#f3e8fd,stroke:#a142f4,color:#111"),
    "noop":      ("⚪", "fill:#f1f3f4,stroke:#9aa0a6,color:#5f6368"),
}
REGEN = "uv run --with pyyaml python scripts/lib/catalog_mermaid.py --doc --out docs/PIPELINE_DIAGRAM.md"


def esc(s: str) -> str:
    return str(s).replace('"', "'")


def build_mermaid(catalog: dict, execution: dict, direction: str) -> str:
    phases = catalog.get("phases", {})
    split = catalog.get("split_pending", {})
    emap = (execution or {}).get("phases", {})   # phase -> {step: {output, venv, ...}}
    tnode = lambda p, s: f"{p}__{s}"

    out = [f"flowchart {direction}"]
    for k, (_, style) in KIND_STYLE.items():
        out.append(f"  classDef {k} {style};")
    out.append("  classDef pending stroke-dasharray:4 3,stroke-width:2px;")

    order, pending, last_out = [], [], {}
    for i, (pid, p) in enumerate(phases.items(), 1):
        order.append((pid, p))
        steps = list((p.get("steps") or {}).items())
        virtual = bool(p.get("virtual"))
        title = f"{i} · {esc(p.get('name', pid))}" + (" (planned)" if virtual else "")
        out.append(f'  subgraph {pid}["{title}"]')
        out.append("    direction LR")
        splitset = {sid for fine in (split.get(pid, {}) or {}).values() for sid in fine}
        phase_out = {sid: (b or {}).get("output")
                     for sid, b in (emap.get(pid, {}) or {}).items()}

        for sid, s in steps:
            kind = s.get("kind", "transform")
            emoji = KIND_STYLE.get(kind, ("", ""))[0]
            tn = tnode(pid, sid)
            out.append(f'    {tn}["{emoji} {esc(s.get("name", sid))}"]:::{kind}')
            if sid in splitset:
                pending.append(tn)
        # intra-phase edges: label each with the artifact the SOURCE step produces
        for j in range(len(steps) - 1):
            a, b = tnode(pid, steps[j][0]), tnode(pid, steps[j + 1][0])
            lbl = phase_out.get(steps[j][0])
            out.append(f'    {a} -->|"{esc(lbl)}"| {b}' if lbl else f'    {a} --> {b}')
        out.append("  end")
        if virtual:
            out.append(f"  style {pid} stroke-dasharray:6 4")
        # the phase's hand-off artifact = the last step that actually produces one
        deliverable = None
        for sid, _ in steps:
            if phase_out.get(sid):
                deliverable = phase_out[sid]
        last_out[pid] = deliverable

    # phase → phase, connected by SUBGRAPH ID (box-to-box) so each lane keeps its
    # inner LR; label the edge with the upstream phase's hand-off artifact.
    for (a_pid, _), (b_pid, b) in zip(order, order[1:]):
        lbl = last_out.get(a_pid)
        if b.get("virtual"):
            out.append(f'  {a_pid} -. "{esc(lbl)}" .-> {b_pid}' if lbl else f"  {a_pid} -.-> {b_pid}")
        else:
            out.append(f'  {a_pid} -->|"{esc(lbl)}"| {b_pid}' if lbl else f"  {a_pid} --> {b_pid}")

    if pending:
        out.append("  class " + ",".join(pending) + " pending")

    out.append('  subgraph legend["Legend"]')
    out.append("    direction LR")
    for k, (emoji, _) in KIND_STYLE.items():
        out.append(f'    leg_{k}["{emoji} {k} task"]:::{k}')
    out.append("  end")
    out.append('  legend_note["Edge label = the file / dir the step produces, consumed by the next step · '
               'dashed border = 1 code step today (split_pending) · dashed phase = virtual / planned"]')
    return "\n".join(out)


def build_doc(catalog: dict, execution: dict, mermaid: str) -> str:
    phases = catalog.get("phases", {})
    aliases = catalog.get("id_aliases", {})
    n_phase = len(phases)
    n_virtual = sum(1 for p in phases.values() if p.get("virtual"))
    steps = [(pid, sid, (s or {}).get("kind", "transform"))
             for pid, p in phases.items() for sid, s in (p.get("steps") or {}).items()]
    by_kind = {}
    for _, _, k in steps:
        by_kind[k] = by_kind.get(k, 0) + 1
    kinds = ", ".join(f"{by_kind[k]} {k}" for k in KIND_STYLE if k in by_kind)
    n_art = sum(1 for ph in (execution or {}).get("phases", {}).values()
                for st in (ph or {}).values() if (st or {}).get("output"))
    alias_line = ", ".join(f"`{o}`→`{n}`" for o, n in aliases.items()) or "none"

    return "\n".join([
        "# Pipeline Diagram",
        "",
        "> **Auto-generated — do not hand-edit.** Projection of "
        "`configs/pipeline_catalog.yaml` (tasks) + `configs/pipeline_execution.yaml` "
        "(outputs + bindings), both keyed by the frozen `<phase>.<step>` ids "
        "(see `PIPELINE_CATALOG_CONTRACT` / `DATA_DRIVEN_PIPELINE_EXECUTOR_PLAN`).",
        f"> Regenerate: `{REGEN}`",
        "",
        f"**At a glance:** {n_phase} phases "
        f"({n_phase - n_virtual} active + {n_virtual} planned), "
        f"{len(steps)} steps ({kinds}); {n_art} declared output artifacts.",
        "",
        "```mermaid",
        mermaid,
        "```",
        "",
        "## Encodings",
        "",
        "- **Each phase = a horizontal swimlane**; its steps flow **left→right**. "
        "Phases stack top-to-bottom and connect lane→lane.",
        "- **Node = task (step)**, coloured by `kind`: 🤖 llm · 🕸 graphmert · "
        "🎯 train · 🔧 transform · 🚀 deploy · ⚪ noop (the enum the UI uses for icons).",
        "- **Edge label = the output artifact** the source step writes and the next "
        "step consumes (path relative to the run dir `$OUTPUT_BASE`; `*` = a "
        "run-varying component, e.g. `checkpoint-<N>`).",
        "- **Unlabeled edge** = the source step writes no separate artifact at this "
        "granularity: a noop (`prune_paths`, `eval_sft`, `eval_rl`), a graphrag-internal "
        "substep (`chunk`, `extract_triples`), or a `split_pending` substep whose file "
        "is its coarse code step's output.",
        "- **Dashed border** = `split_pending`: catalog-granular steps that are still "
        "**one code step today** (graphmert `preprocess`, `validate_predictions`).",
        "- **Dashed phase** = `virtual: true` (planned; no `scripts/phases/*.sh` yet).",
        f"- **Code-side `id_aliases`** (not shown; diagram uses the frozen catalog ids): {alias_line}.",
        "",
        "> Output paths are the **canonical/static** view declared in "
        "`configs/pipeline_execution.yaml`; the per-run truth moves to "
        "`run_manifest.json` `outputs` when DATA_DRIVEN plan §5.1 lands.",
        "",
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="configs/pipeline_catalog.yaml")
    ap.add_argument("--execution", default="configs/pipeline_execution.yaml")
    ap.add_argument("--direction", default="TB", choices=["TB", "LR"],
                    help="lane stacking: TB stacks phases as swimlanes (default)")
    ap.add_argument("--out", default=None, help="write to file (default stdout)")
    ap.add_argument("--md", action="store_true", help="wrap in a ```mermaid fence")
    ap.add_argument("--doc", action="store_true", help="full markdown doc (banner + chart + encodings)")
    a = ap.parse_args()

    import yaml
    catalog = yaml.safe_load(Path(a.catalog).read_text())
    epath = Path(a.execution)
    execution = yaml.safe_load(epath.read_text()) if epath.exists() else {}
    mermaid = build_mermaid(catalog, execution, a.direction)

    if a.doc:
        text = build_doc(catalog, execution, mermaid)
    elif a.md:
        text = "```mermaid\n" + mermaid + "\n```"
    else:
        text = mermaid

    if a.out:
        Path(a.out).write_text(text + ("" if text.endswith("\n") else "\n"))
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
