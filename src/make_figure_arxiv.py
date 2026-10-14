"""arXiv Figure: channel strength and authorship on Grid A.

Reads results/arxiv/positive_controls.json (written by analyze_arxiv.py) and
plots per-cell fidelity F for every Grid A condition at 500 words plus the
positive controls, cross cells (receiver != author) on the left and self cells
on the right. Same palette as make_figure.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PC = ROOT / "results" / "arxiv" / "positive_controls.json"
SUMMARY = ROOT / "results" / "arxiv" / "grid_summary.json"
OUT = ROOT / "arxiv" / "fig3_positive_controls.png"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, SEC, MUTED, GRID_C, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#ffffff"

# (key, label) in display order; key matches arm names in the json
CONDITIONS = [
    ("B", "generic\nprompt ($B$)"),
    ("neutral", "self-\ndescription"),
    ("adopt", "self-desc.\n+ adopt"),
    ("3p", "observer-\nwritten"),
    ("selfobs-500", "self-\nobserved"),
    ("optimized-500", "optimized\n(5 rounds)"),
    ("fewshot-30", "demos\nk=30"),
    ("fewshot-60", "demos\nk=60"),
    ("fewshot-90", "demos\nk=90"),
    ("oracle-110", "answer key\nk=110"),
]


def main() -> None:
    pc = json.loads(PC.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    ceiling = summary.get("perturbation_ladder", {}).get("retest_C_mean")

    vals = {"cross": {k: [] for k, _ in CONDITIONS}, "self": {k: [] for k, _ in CONDITIONS}}
    for pair, b in pc["pairs_B"].items():
        vals["cross"]["B"].append(b)
    for c in pc["grid_a_reference"] + [c for c in pc["cells"] if c.get("grid", "A") == "A"]:
        key = c.get("cond", c["arm"])  # sprint reference cells keyed by arm, added ones by cond
        if key in vals[c["kind"]]:
            vals[c["kind"]][key].append(c["F"])

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6), dpi=200, sharey=True,
                             gridspec_kw={"width_ratios": [10, 9]})
    fig.patch.set_facecolor(SURFACE)
    for ax, kind, color, marker, title in (
            (axes[0], "cross", ORANGE, "^", "Cross cells: another model runs the text"),
            (axes[1], "self", BLUE, "o", "Self cells: the author's own weights run the text")):
        ax.set_facecolor(SURFACE)
        conds = [(k, l) for k, l in CONDITIONS if not (kind == "self" and k == "B")]
        if ceiling is not None:
            ax.axhline(ceiling, color=BASE, lw=1.0, ls=(0, (2, 2)), zorder=1)
            ax.text(len(conds) - 0.55, ceiling + 0.003, "test-retest ceiling", fontsize=7.5,
                    color=MUTED, ha="right", va="bottom")
        if kind == "self" and vals["cross"]["B"]:
            b = float(np.mean(vals["cross"]["B"]))
            ax.axhline(b, color=BASE, lw=1.0, zorder=1)
            ax.text(-0.5, b + 0.003, "$B$ (unrelated pairs)", fontsize=7.5,
                    color=MUTED, ha="left", va="bottom")
        for x, (k, label) in enumerate(conds):
            ys = vals[kind][k]
            if not ys:
                continue
            jitter = np.linspace(-0.16, 0.16, len(ys)) if len(ys) > 1 else [0.0]
            ax.scatter(x + np.array(jitter), ys, s=30, marker=marker, facecolor=color,
                       edgecolor=SURFACE, linewidth=0.7, zorder=4)
            ax.hlines(np.mean(ys), x - 0.3, x + 0.3, color=color, lw=2.2, zorder=5)
            ax.text(x, 0.62, f"{np.mean(ys):.3f}", ha="center", va="bottom", fontsize=7.5,
                    color=SEC)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([l for _, l in conds], fontsize=7.2, color=SEC)
        ax.set_xlim(-0.6, len(conds) - 0.4)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.grid(True, axis="y", color=GRID_C, lw=0.6, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASE)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_ylim(0.6, 1.0)
    axes[0].set_ylabel(r"Fidelity to the author  $F = 1 - \overline{\rm JSD}$", fontsize=9.5,
                       color=SEC)
    fig.tight_layout()
    fig.savefig(OUT, facecolor=SURFACE, bbox_inches="tight")
    for kind in ("cross", "self"):
        print(kind, {k: (len(v), round(float(np.mean(v)), 3)) for k, v in vals[kind].items() if v})
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
