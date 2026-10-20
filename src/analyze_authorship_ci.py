"""Pooled authorship contrast on Grid A with a joint item bootstrap: self-fidelity of the
self-observation-500 description minus the mean self-fidelity of the two observer-written
(third-person-500) descriptions, per target, pooled over the three targets. Also the same
contrast for cross cells (receiver != author). Writes results/arxiv/authorship_ci.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_grid as ag
from analyze_strength import joint_pooled, tbl_of

ROOT = Path(__file__).resolve().parent.parent
MERGED = ROOT / "results" / "arxiv" / "grid_merged.jsonl"
PC = ROOT / "runs" / "arxiv_positive_controls.jsonl"
OUT = ROOT / "results" / "arxiv" / "authorship_ci.json"


def main() -> None:
    grid = ag.load_jsonl(MERGED)
    runs = ag.load_jsonl(PC)
    screen = ag.load_jsonl(ag.SCREEN)
    final_items = set(grid["item_id"].unique())
    ref = ag.p_hat_table(screen[screen["item_id"].isin(final_items)], "short")
    for m, t in ag.p_hat_table(grid[grid["arm"] == "baseline"], "receiver").items():
        ref[m] = t
    out = {}
    for kind in ("self", "cross"):
        pairs = []
        for target in ag.GRID_A:
            for receiver in ag.GRID_A:
                if (receiver == target) != (kind == "self"):
                    continue
                so = runs[(runs["arm"] == "selfobs") & (runs["author"] == target) & (runs["receiver"] == receiver)]
                ob = grid[(grid["arm"] == "3p") & (grid["author"] == target) & (grid["receiver"] == receiver)]
                if len(so) < 400 or len(ob) < 400:
                    continue
                f_so = ag.fidelity(ref[target], tbl_of(so))
                # observer-written: average per-item fidelity over the describers (cell_id carries the describer)
                per_desc = []
                for _, g in ob.groupby("cell_id"):
                    f = ag.fidelity(ref[target], tbl_of(g))
                    if f:
                        per_desc.append(dict(zip(f["items"], f["per_item"])))
                if not f_so or not per_desc:
                    continue
                dso = dict(zip(f_so["items"], f_so["per_item"]))
                common = set(dso)
                for d in per_desc:
                    common &= set(d)
                pairs.append({i: dso[i] - float(np.mean([d[i] for d in per_desc])) for i in common})
        res = joint_pooled(pairs)
        out[kind] = res
        print(f"{kind}: self-observation minus observer-written, pooled over {res['n_pairs']} cells: "
              f"{res['mean']:+.3f} [{res['ci'][0]:+.3f}, {res['ci'][1]:+.3f}] (items {res['n_items']})")
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
