"""Formal tests for the perturbation claim: a model's own self-description moves
it further from its own baseline than unrelated prose of the same length.

Per length (100, 500, 2000):
  pooled paired bootstrap over all (model, item) differences
      d_mi = F_item(own-desc, model m) - F_item(unrelated, model m)
  (10,000 resamples of the 10 x 110 difference matrix, resampling items and
  models jointly, i.e. a two-way cluster bootstrap), plus a sign test across
  models (how many of the 10 have mean d < 0, two-sided binomial p).
Also for the original ocean-currents placebo at 500 words.
Reads runs/arxiv_placebo_lengths.jsonl and the merged grid; no new calls."""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np

import analyze_grid as ag

ROOT = Path(__file__).resolve().parent.parent
MERGED = ROOT / "results" / "arxiv" / "grid_merged.jsonl"
RUNS = ROOT / "runs" / "arxiv_placebo_lengths.jsonl"
OUT = ROOT / "results" / "arxiv" / "perturbation_test.json"
N_BOOT = 10_000


def tbl_of(g):
    ok = g[g["choice"].notna()]
    return {i: (gg["choice"] == "option_a").mean() for i, gg in ok.groupby("item_id")}


def per_item(ref, tbl):
    fid = ag.fidelity(ref, tbl)
    return dict(zip(fid["items"], fid["per_item"]))


def sign_test(k: int, n: int) -> float:
    """two-sided binomial p for k successes of n at p=0.5"""
    tail = sum(comb(n, j) for j in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def cluster_boot(D: np.ndarray, rng) -> tuple[float, float]:
    m, n = D.shape
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        rows = rng.integers(0, m, m)
        cols = rng.integers(0, n, n)
        means[b] = D[np.ix_(rows, cols)].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    grid = ag.load_jsonl(MERGED if MERGED.exists() else ag.GRID)
    screen = ag.load_jsonl(ag.SCREEN)
    runs = ag.load_jsonl(RUNS)
    final_items = set(grid["item_id"].unique())
    ref = ag.p_hat_table(screen[screen["item_id"].isin(final_items)], "short")
    for model, t in ag.p_hat_table(grid[grid["arm"] == "baseline"], "receiver").items():
        ref[model] = t
    models = sorted(m for m in ref if m in set(grid["receiver"]))

    def own_desc(m, n):
        c = grid[(grid["receiver"] == m) & (grid["author"] == m) & (grid["arm"] == "neutral")
                 & (grid["length"] == n) & (grid["regen"] == 0)]
        return tbl_of(c) if len(c) else None

    conditions = {100: ("mixed", 100), 500: ("mixed", 500), 2000: ("mixed", 2000), "500-ocean": ("ocean", 500)}
    out = {}
    rng = np.random.default_rng(14)
    print(f"{'condition':<10} {'n_models':>8} {'mean d':>8} {'95% CI (cluster boot)':>24} {'models d<0':>11} {'sign-test p':>12}")
    for key, (src, n) in conditions.items():
        rows = []
        for m in models:
            od = own_desc(m, n)
            if od is None:
                continue
            if src == "mixed":
                g = runs[(runs["receiver"] == m) & (runs["length"] == n)]
            else:
                g = grid[(grid["receiver"] == m) & (grid["arm"] == "placebo")]
            if len(g) < 440:
                continue
            a, b = per_item(ref[m], od), per_item(ref[m], tbl_of(g))
            rows.append({i: a[i] - b[i] for i in set(a) & set(b)})
        common = sorted(set.intersection(*(set(r) for r in rows)))  # items answered by every model
        D = np.array([[r[i] for i in common] for r in rows])
        lo, hi = cluster_boot(D, rng)
        k = int((D.mean(axis=1) < 0).sum())
        p = sign_test(k, D.shape[0])
        out[str(key)] = {"n_models": int(D.shape[0]), "mean_d": float(D.mean()), "ci": [lo, hi],
                         "models_negative": k, "sign_test_p": p}
        print(f"{str(key):<10} {D.shape[0]:>8} {D.mean():>+8.3f} {f'[{lo:+.3f}, {hi:+.3f}]':>24} "
              f"{k:>5}/{D.shape[0]:<5} {p:>12.3f}")
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
