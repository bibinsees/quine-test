"""Specificity gain at neutral-500 (Section 4.3): for each receiver R and author A in
the same grid, F(A-ref | R running A's description) minus F(A-ref | R running a third
model C's description), over the 18 such comparisons (three trios, six ordered
(A, R) pairs, one C each). Two intervals: joint item resampling (one item resample per
replicate applied to every comparison; the primary procedure) and within-comparison
item resampling (sensitivity check). Writes results/arxiv/specificity.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import analyze_grid as ag
from analyze_strength import tbl_of

ROOT = Path(__file__).resolve().parent.parent
MERGED = ROOT / "results" / "arxiv" / "grid_merged.jsonl"
OUT = ROOT / "results" / "arxiv" / "specificity.json"
N_BOOT = 10_000


def main() -> None:
    grid = ag.load_jsonl(MERGED)
    screen = ag.load_jsonl(ag.SCREEN)
    final_items = set(grid["item_id"].unique())
    ref = ag.p_hat_table(screen[screen["item_id"].isin(final_items)], "short")
    for m, t in ag.p_hat_table(grid[grid["arm"] == "baseline"], "receiver").items():
        ref[m] = t
    n500 = grid[(grid["arm"] == "neutral") & (grid["length"] == 500) & (grid["regen"] == 0)]
    comps = []
    for trio in (ag.GRID_A, ag.GRID_B, ag.LOCALS):
        for a in trio:
            for r in trio:
                if a == r:
                    continue
                (c,) = [m for m in trio if m not in (a, r)]
                own = ag.fidelity(ref[a], tbl_of(n500[(n500["author"] == a) & (n500["receiver"] == r)]))
                oth = ag.fidelity(ref[a], tbl_of(n500[(n500["author"] == c) & (n500["receiver"] == r)]))
                if not own or not oth:
                    continue
                da = dict(zip(own["items"], own["per_item"]))
                db = dict(zip(oth["items"], oth["per_item"]))
                d = {i: da[i] - db[i] for i in set(da) & set(db)}
                comps.append({"author": a, "receiver": r, "other": c, "gain": float(np.mean(list(d.values()))),
                              "n_items": len(d), "d": d})
    gains = np.array([x["gain"] for x in comps])
    # joint item resampling on the items common to all comparisons
    common = sorted(set.intersection(*(set(x["d"]) for x in comps)))
    D = np.array([[x["d"][i] for i in common] for x in comps])
    rng = np.random.default_rng(14)
    idx = rng.integers(0, D.shape[1], size=(N_BOOT, D.shape[1]))
    joint = np.array([D[:, ix].mean() for ix in idx])
    # within-comparison resampling (each comparison resamples its own items)
    rng = np.random.default_rng(14)
    arrs = [np.array(list(x["d"].values())) for x in comps]
    within = np.array([np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in arrs]) for _ in range(N_BOOT)])
    out = {"n_comparisons": len(comps), "mean_gain": float(gains.mean()),
           "joint": {"n_items": len(common), "mean": float(D.mean()),
                     "ci": [float(np.percentile(joint, 2.5)), float(np.percentile(joint, 97.5))]},
           "within": {"ci": [float(np.percentile(within, 2.5)), float(np.percentile(within, 97.5))]},
           "comparisons": [{k: v for k, v in x.items() if k != "d"} for x in comps]}
    print(f"specificity gain: n={len(comps)} mean={gains.mean():+.4f}")
    print(f"  joint item resampling ({len(common)} common items): mean={D.mean():+.4f} "
          f"CI [{out['joint']['ci'][0]:+.4f}, {out['joint']['ci'][1]:+.4f}]")
    print(f"  within-comparison resampling: CI [{out['within']['ci'][0]:+.4f}, {out['within']['ci'][1]:+.4f}]")
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
