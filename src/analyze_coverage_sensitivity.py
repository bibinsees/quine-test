"""Sensitivity of the pooled transfer gain to parsed-answer coverage: pooled T (joint item
bootstrap) for the self-description conditions using only cross cells whose parsed-answer
rate is at or above a threshold. Writes results/arxiv/coverage_sensitivity.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import analyze_grid as ag
from analyze_strength import joint_pooled, tbl_of

ROOT = Path(__file__).resolve().parent.parent
MERGED = ROOT / "results" / "arxiv" / "grid_merged.jsonl"
OUT = ROOT / "results" / "arxiv" / "coverage_sensitivity.json"
CONDS = [("neutral", 100), ("neutral", 500), ("neutral", 2000), ("framed", 100), ("framed", 500), ("framed", 2000)]
THRESHOLDS = [0.0, 0.90, 0.95]


def main() -> None:
    grid = ag.load_jsonl(MERGED)
    screen = ag.load_jsonl(ag.SCREEN)
    final_items = set(grid["item_id"].unique())
    ref = ag.p_hat_table(screen[screen["item_id"].isin(final_items)], "short")
    for m, t in ag.p_hat_table(grid[grid["arm"] == "baseline"], "receiver").items():
        ref[m] = t
    out = {}
    print(f"{'condition':<14}{'threshold':>10}{'pairs':>6}{'mean T':>9}{'95% CI':>22}{'upper95':>9}")
    for arm, n in CONDS:
        df = grid[(grid["arm"] == arm) & (grid["length"] == n) & (grid["regen"] == 0)]
        for thr in THRESHOLDS:
            pairs = []
            for (r, a), g in df.groupby(["receiver", "author"]):
                if r == a or a not in ref or len(g) < 400:
                    continue
                rate = g["choice"].notna().mean()
                if rate < thr:
                    continue
                fc, b = ag.fidelity(ref[a], tbl_of(g)), ag.fidelity(ref[a], ref[r])
                if fc is None or b is None:
                    continue
                dc, db = dict(zip(fc["items"], fc["per_item"])), dict(zip(b["items"], b["per_item"]))
                pairs.append({i: dc[i] - db[i] for i in set(dc) & set(db)})
            res = joint_pooled(pairs)
            if res:
                out[f"{arm}-{n}|>={thr:.2f}"] = res
                print(f"{arm + '-' + str(n):<14}{thr:>10.2f}{res['n_pairs']:>6}{res['mean']:>+9.3f}   "
                      f"[{res['ci'][0]:+.3f}, {res['ci'][1]:+.3f}]{res['upper95']:>+9.3f}")
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
