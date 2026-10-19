"""Pooled transfer gain with a JOINT item bootstrap: items are resampled once per
replicate and the same resample is applied to every cross pair, preserving the
shared-question dependence. Reports mean T and 95% CI for the added Grid A
conditions. Reads results/arxiv/grid_merged.jsonl and the arxiv run logs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_grid as ag

ROOT = Path(__file__).resolve().parent.parent
MERGED = ROOT / "results" / "arxiv" / "grid_merged.jsonl"
RUNS = ["arxiv_positive_controls.jsonl", "arxiv_demos_dose.jsonl", "arxiv_demos_balanced.jsonl",
        "arxiv_obs_stable.jsonl", "arxiv_usermsg.jsonl"]
CONDS = [("fewshot", 30), ("fewshot", 60), ("fewshot", 90), ("oracle", 110), ("selfobs", 500),
         ("fewshot_bal", 30), ("demos_bal", 55), ("selfobs_stable", 500), ("neutral_user", 500),
         ("oracle_user", 110)]
N_BOOT = 10_000


def tbl_of(g):
    ok = g[g["choice"].notna()]
    return {i: (gg["choice"] == "option_a").mean() for i, gg in ok.groupby("item_id")}


def main() -> None:
    grid = ag.load_jsonl(MERGED)
    screen = ag.load_jsonl(ag.SCREEN)
    runs = pd.concat([ag.load_jsonl(ROOT / "runs" / f) for f in RUNS if (ROOT / "runs" / f).exists()],
                     ignore_index=True)
    final_items = set(grid["item_id"].unique())
    ref = ag.p_hat_table(screen[screen["item_id"].isin(final_items)], "short")
    for m, t in ag.p_hat_table(grid[grid["arm"] == "baseline"], "receiver").items():
        ref[m] = t
    rng = np.random.default_rng(14)
    out = {}
    # neutral-500 (sprint) on Grid A for comparison
    sources = {("neutral", 500): grid[(grid["arm"] == "neutral") & (grid["length"] == 500) & (grid["regen"] == 0)
                                      & grid["receiver"].isin(ag.GRID_A) & grid["author"].isin(ag.GRID_A)]}
    for arm, n in CONDS:
        sources[(arm, n)] = runs[(runs["arm"] == arm) & (runs["length"] == n) & runs["receiver"].isin(ag.GRID_A)]
    print(f"{'condition':<22} {'pairs':>5} {'mean T':>8} {'joint 95% CI':>20}")
    for (arm, n), df in sources.items():
        pairs = []
        for (r, a), g in df.groupby(["receiver", "author"]):
            if r == a or len(g) < 400:
                continue
            fc = ag.fidelity(ref[a], tbl_of(g))
            b = ag.fidelity(ref[a], ref[r])
            if fc is None:
                continue
            dc, db = dict(zip(fc["items"], fc["per_item"])), dict(zip(b["items"], b["per_item"]))
            pairs.append({i: dc[i] - db[i] for i in set(dc) & set(db)})
        if not pairs:
            continue
        common = sorted(set.intersection(*(set(p) for p in pairs)))
        D = np.array([[p[i] for i in common] for p in pairs])  # pairs x items
        means = np.array([D[:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(N_BOOT)])
        lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
        out[f"{arm}-{n}"] = {"n_pairs": len(pairs), "n_items": len(common), "mean_T": float(D.mean()), "ci": [lo, hi]}
        print(f"{arm + '-' + str(n):<22} {len(pairs):>5} {D.mean():>+8.3f} [{lo:+.3f}, {hi:+.3f}]  (items {len(common)})")
    (ROOT / "results" / "arxiv" / "pooled_T_joint.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
