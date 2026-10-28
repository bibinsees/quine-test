"""arXiv summary figure (two panels, generated from analysis outputs).

Panel A: pooled transfer gain T = F_cross - B per condition with joint item
bootstrap 95% CIs (results/arxiv/strength.json, analyze_strength.py).
Panel B: per-model perturbation ladder: fidelity of each model to its own
generic-prompt profile under each prefix (results/arxiv/self_ladder.json and
placebo_lengths*.json). Writes arxiv/fig2_summary.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
STRENGTH = ROOT / "results" / "arxiv" / "strength.json"
LADDER = ROOT / "results" / "arxiv" / "self_ladder.json"
PLACEBO = {"mixed": ROOT / "results" / "arxiv" / "placebo_lengths.json",
           "expo": ROOT / "results" / "arxiv" / "placebo_lengths_expo.json"}
OUT = ROOT / "arxiv" / "fig2_summary.png"

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#2e8b57"
INK, SEC, MUTED, GRID_C, BASE = "#0b0b0b", "#2e2e2e", "#4a4a4a", "#dcdcdc", "#9a9a9a"

# (key in strength.json, label, group) in display order (top to bottom)
ROWS = [
    ("neutral-100 (all grids)", "self-description, 100 words (20 pairs, all grids)", "desc"),
    ("neutral-500 (all grids)", "self-description, 500 words (20 pairs, all grids)", "desc"),
    ("neutral-2000 (all grids)", "self-description, 2,000 words (20 pairs, all grids)", "desc"),
    ("framed-500 (all grids)", "purpose-framed, 500 words (20 pairs, all grids)", "desc"),
    ("para-500 (A)", "paraphrased (Grid A)", "desc"),
    ("adopt-500 (A)", "with adoption instruction (Grid A)", "desc"),
    ("neutral_user-500 (A)", "in the user message (Grid A)", "desc"),
    ("optimized-500 (A)", "best of original + 5 rewrites (Grid A)", "desc"),
    ("3p-500 (A)", "third-person-500: observer-written from 90 unstable items (Grid A)", "obs"),
    ("selfobs_stable-500 (A)", "self-observation, stable items (Grid A)", "obs"),
    ("fewshot-30 (A)", "demonstrations-30, chosen-first format (Grid A)", "demo"),
    ("demos_bal-55 (A)", "demonstrations-55, balanced, stable items (Grid A)", "demo"),
    ("oracle-110 (A)", "answer key, 110 test items (Grid A)", "key"),
    ("oracle-110 (B)", "answer key (Grid B, preregistered)", "key"),
    ("oracle-110 (local)", "answer key (local trio, preregistered)", "key"),
    ("policy", "policy text, text-installed profile (Grid A, preregistered)", "sem"),
    ("policy_desc", "self-description written under the policy (Grid A, preregistered)", "sem"),
]
COLOR = {"desc": ORANGE, "obs": "#8a5cc2", "demo": GREEN, "key": BLUE, "sem": "#b8860b"}
POLICY = ROOT / "results" / "arxiv" / "policy_persona.json"

LADDER_COLS = [("retest", "test-\nretest"), ("no_prompt", "no\nprompt"), ("u100", "unrelated\nprose\n100 w"),
               ("u500", "unrelated\nprose\n500 w"), ("u2000", "unrelated\nprose\n2,000 w"),
               ("own_desc", "own\nself-\ndescr."), ("other_desc", "another\nmodel's\nself-descr.")]


def main() -> None:
    st = json.loads(STRENGTH.read_text(encoding="utf-8"))["pooled_T"]
    if POLICY.exists():
        po = json.loads(POLICY.read_text(encoding="utf-8"))["pooled"]
        st["policy"] = {"mean": po["T_policy"]["mean"], "ci": po["T_policy"]["ci"]}
        st["policy_desc"] = {"mean": po["T_desc"]["mean"], "ci": po["T_desc"]["ci"]}
    lad = json.loads(LADDER.read_text(encoding="utf-8"))
    unrel: dict = {}  # model -> length -> [F per style]
    for key, p in PLACEBO.items():
        if p.exists():
            for row in json.loads(p.read_text(encoding="utf-8"))["table8"]["models"]:
                unrel.setdefault(row["model"], {}).setdefault(row["length"], []).append(row["placebo"])

    # two panels stacked, each full text width, so labels stay readable in print
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(8.8, 12.0), dpi=200, gridspec_kw={"height_ratios": [1.35, 1]})
    fig.patch.set_facecolor("white")

    # ---- Panel A: forest plot of pooled T
    rows = [(k, l, g) for k, l, g in ROWS if k in st]
    # y positions: one row per condition, plus a 0.8-row gap at every group start
    # so the group titles sit clear of the interval bar above them
    GAP = 0.8
    ys_list, y = [], 0.0
    for i, (_, _, g) in enumerate(rows):
        if i > 0 and rows[i - 1][2] != g:
            y += GAP
        ys_list.append(y)
        y += 1.0
    ys = (y - 1.0) - np.array(ys_list)  # top row first
    axA.axvline(0, color=SEC, lw=1.0, zorder=1)
    for y, (k, label, grp) in zip(ys, rows):
        r = st[k]
        lo, hi = r["ci"]
        axA.plot([lo, hi], [y, y], color=COLOR[grp], lw=2.2, solid_capstyle="round", zorder=3)
        axA.scatter([r["mean"]], [y], s=46, color=COLOR[grp], edgecolor="white", linewidth=0.8, zorder=4)
        axA.text(0.333, y, f"{r['mean']:+.3f}", va="center", ha="right", fontsize=10, color=SEC)
    axA.set_yticks(ys)
    axA.set_yticklabels([l for _, l, _ in rows], fontsize=10, color=INK)
    axA.set_xlim(-0.175, 0.335)
    axA.set_xlabel(r"Pooled transfer gain $T_{\rm script} = F_{\rm cross} - B$, 95% joint item bootstrap CI",
                   fontsize=10.5, color=SEC)
    axA.set_title("A. Does the text bring a receiver closer to the author?", fontsize=12, color=INK, loc="left")
    axA.grid(True, axis="x", color=GRID_C, lw=0.6, zorder=0)
    groups = [g for _, _, g in rows]
    starts = [i for i, g in enumerate(groups) if i == 0 or groups[i - 1] != g]
    for i in starts[1:]:
        axA.axhline(ys[i] + 0.5 + GAP / 2, color=GRID_C, lw=0.8, zorder=0)
    titles = {"desc": "self-descriptions", "obs": "descriptions from recorded choices",
              "demo": "examples on other items", "key": "answers to the test items",
              "sem": "known semantic persona (gain over policy baseline $B'$, 57 items)"}
    for i in starts:
        axA.text(-0.17, ys[i] + 0.5, titles[groups[i]], fontsize=10, color=COLOR[groups[i]], va="bottom")
    axA.set_ylim(-0.8, ys[0] + 1.1)

    # ---- Panel B: perturbation ladder
    models = [r["model"] for r in lad["rows"]]
    vals = {c: [] for c, _ in LADDER_COLS}
    for r in lad["rows"]:
        vals["retest"].append(r["retest"])
        vals["no_prompt"].append(r["no_prompt"])
        vals["own_desc"].append(r["own_desc"])
        vals["other_desc"].append(r["other_desc"])
        for n in (100, 500, 2000):
            fs = unrel.get(r["model"], {}).get(n, [])
            vals[f"u{n}"].append(float(np.mean(fs)) if fs else np.nan)
    xs = np.arange(len(LADDER_COLS))
    for i, m in enumerate(models):
        y = [vals[c][i] for c, _ in LADDER_COLS]
        axB.plot(xs, y, color="#b0b0b0", lw=0.9, zorder=2)
        axB.scatter(xs, y, s=20, color="#6a6a6a", zorder=3)
    means = [float(np.nanmean(vals[c])) for c, _ in LADDER_COLS]
    axB.plot(xs, means, color=INK, lw=2.4, zorder=5)
    axB.scatter(xs, means, s=60, color=INK, zorder=6)
    for x, m in zip(xs, means):
        axB.text(x, 0.835, f"{m:.3f}", ha="center", fontsize=10, color=SEC)
    axB.set_xticks(xs)
    axB.set_xticklabels([l for _, l in LADDER_COLS], fontsize=10, color=INK)
    axB.set_ylim(0.83, 1.0)
    axB.set_ylabel("Fidelity to the model's own generic-prompt profile", fontsize=10.5, color=SEC)
    axB.set_title("B. How far does each prefix move a model from its usual choices?", fontsize=12, color=INK,
                  loc="left")
    axB.grid(True, axis="y", color=GRID_C, lw=0.6, zorder=0)
    axB.text(len(xs) - 0.6, 0.975, "ten models; black = mean", fontsize=10, color=SEC, ha="right")
    for ax in (axA, axB):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASE)
        ax.tick_params(colors=SEC, labelsize=9)
    fig.tight_layout(h_pad=2.5)
    # panel B does not need panel A's wide label margin: let it span the full width
    posA, posB = axA.get_position(), axB.get_position()
    axB.set_position([0.075, posB.y0, posA.x1 - 0.075, posB.height])
    fig.savefig(OUT, facecolor="white", bbox_inches="tight")
    print("panel A rows:", len(rows), "| panel B means:", dict(zip([c for c, _ in LADDER_COLS], [round(m, 3) for m in means])))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
