"""Strength analyses on existing data (no new calls).

1. Pooled transfer gain T with a JOINT item bootstrap for every self-description
   condition over all grids, plus the added conditions per grid, with the 95%
   upper bound (what size of average gain the data exclude).
2. Movement decomposition at neutral-500: how much a receiver moves under a
   description (magnitude) and how much of that movement points toward the
   author (slope of movement on the author-receiver difference), against
   unrelated prose of the same length. Movement is measured from an
   independent baseline sample (the test-retest cell) so that shared baseline
   noise does not induce a spurious slope.
3. Common receiver shift: per receiver, the correlation of per-item movement
   across different authors' descriptions vs across unrelated prose texts.
4. Per-category transfer gain at neutral-500 (exploratory).
5. Author identifiability: can the author be picked out from the receiver's
   behavior under its description (nearest author by fidelity), chance 1/2.

Reads results/arxiv/grid_merged.jsonl, runs/screening.jsonl and the arxiv run
logs; writes results/arxiv/strength.json.
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_grid as ag

ROOT = Path(__file__).resolve().parent.parent
MERGED = ROOT / "results" / "arxiv" / "grid_merged.jsonl"
ITEMS = ROOT / "battery" / "items_final.json"
OUT = ROOT / "results" / "arxiv" / "strength.json"
RUNS = ["arxiv_positive_controls.jsonl", "arxiv_demos_dose.jsonl", "arxiv_demos_balanced.jsonl",
        "arxiv_obs_stable.jsonl", "arxiv_usermsg.jsonl", "arxiv_optimized.jsonl", "arxiv_gridb.jsonl",
        "arxiv_local_demos.jsonl",
        "arxiv_placebo_lengths.jsonl", "arxiv_placebo_expo.jsonl"]
GROUPS = [ag.GRID_A, ag.GRID_B, ag.LOCALS, ["hermes3-70b", "llama3.3-70b"]]
GRID_OF = {m: "A" for m in ag.GRID_A} | {m: "B" for m in ag.GRID_B} | {m: "local" for m in ag.LOCALS} \
    | {"hermes3-70b": "local"}
N_BOOT = 10_000
rng = np.random.default_rng(14)


def tbl_of(g: pd.DataFrame) -> dict:
    ok = g[g["choice"].notna()]
    return {i: (gg["choice"] == "option_a").mean() for i, gg in ok.groupby("item_id")}


def partners(m: str) -> list[str]:
    return [p for grp in GROUPS if m in grp for p in grp if p != m]


def joint_pooled(pairs: list[dict]) -> dict | None:
    """pairs: list of {item: value}; pooled mean with joint item bootstrap."""
    if not pairs:
        return None
    common = sorted(set.intersection(*(set(p) for p in pairs)))
    D = np.array([[p[i] for i in common] for p in pairs])
    idx = rng.integers(0, D.shape[1], size=(N_BOOT, D.shape[1]))
    means = np.array([D[:, ix].mean() for ix in idx])
    return {"n_pairs": len(pairs), "n_items": len(common), "mean": float(D.mean()),
            "ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "upper95": float(np.percentile(means, 95))}


def main() -> None:
    grid = ag.load_jsonl(MERGED)
    screen = ag.load_jsonl(ag.SCREEN)
    runs = pd.concat([ag.load_jsonl(ROOT / "runs" / f) for f in RUNS if (ROOT / "runs" / f).exists()],
                     ignore_index=True)
    items = json.loads(ITEMS.read_text(encoding="utf-8"))
    category = {it["id"]: it["category"] for it in items}
    final_items = set(grid["item_id"].unique())
    ref = ag.p_hat_table(screen[screen["item_id"].isin(final_items)], "short")
    for m, t in ag.p_hat_table(grid[grid["arm"] == "baseline"], "receiver").items():
        ref[m] = t
    retest = {m: tbl_of(g) for m, g in grid[grid["arm"] == "retest"].groupby("receiver")}
    out: dict = {}

    def per_item_T(a: str, r: str, cell: dict) -> dict | None:
        fc, b = ag.fidelity(ref[a], cell), ag.fidelity(ref[a], ref[r])
        if fc is None or b is None:
            return None
        dc, db = dict(zip(fc["items"], fc["per_item"])), dict(zip(b["items"], b["per_item"]))
        return {i: dc[i] - db[i] for i in set(dc) & set(db)}

    # ---------- 1. pooled T, joint bootstrap, all conditions ----------
    sources: dict[str, pd.DataFrame] = {}
    for arm in ("neutral", "framed"):
        for n in (100, 500, 2000):
            sources[f"{arm}-{n} (all grids)"] = grid[(grid["arm"] == arm) & (grid["length"] == n) & (grid["regen"] == 0)]
    for arm, n in (("para", 500), ("adopt", 500), ("3p", 500)):
        sources[f"{arm}-{n} (A)"] = grid[(grid["arm"] == arm) & (grid["length"] == n) & (grid["regen"] == 0)]
    for arm, n in (("selfobs", 500), ("selfobs_stable", 500), ("optimized", 500), ("fewshot", 30), ("fewshot", 60),
                   ("fewshot", 90), ("fewshot_bal", 30), ("demos_bal", 55), ("oracle", 110), ("neutral_user", 500),
                   ("oracle_user", 110)):
        sources[f"{arm}-{n} (A)"] = runs[(runs["arm"] == arm) & (runs["length"] == n) & runs["receiver"].isin(ag.GRID_A)]
    for arm, n in (("oracle", 110), ("fewshot", 30), ("selfobs", 500)):
        sources[f"{arm}-{n} (B)"] = runs[(runs["arm"] == arm) & (runs["length"] == n) & runs["receiver"].isin(ag.GRID_B)]
    for arm, n in (("oracle", 110), ("fewshot", 30), ("fewshot", 90)):
        sources[f"{arm}-{n} (local)"] = runs[(runs["arm"] == arm) & (runs["length"] == n) & runs["receiver"].isin(ag.LOCALS)]
    pooled = {}
    print("\n1. POOLED T (joint item bootstrap)")
    print(f"{'condition':<28}{'pairs':>6}{'mean T':>9}{'95% CI':>20}{'upper95':>9}")
    for name, df in sources.items():
        pairs = []
        for (r, a), g in df.groupby(["receiver", "author"]):
            if r == a or a not in ref or len(g) < 400:
                continue
            d = per_item_T(a, r, tbl_of(g))
            if d:
                pairs.append(d)
        res = joint_pooled(pairs)
        if res:
            pooled[name] = res
            print(f"{name:<28}{res['n_pairs']:>6}{res['mean']:>+9.3f}   [{res['ci'][0]:+.3f}, {res['ci'][1]:+.3f}]"
                  f"{res['upper95']:>+9.3f}")
    out["pooled_T"] = pooled

    # ---------- 2. movement decomposition at neutral-500 ----------
    def movement(r: str, a: str, cell: dict) -> dict | None:
        """m = p_cell - p_retest(r); t = p_ref(a) - p_ref(r); independent baseline samples."""
        common = sorted(set(cell) & set(retest.get(r, {})) & set(ref[r]) & set(ref[a]))
        if len(common) < ag.MIN_ITEMS:
            return None
        m = np.array([cell[i] - retest[r][i] for i in common])
        t = np.array([ref[a][i] - ref[r][i] for i in common])
        return {"items": common, "m": m, "t": t}

    def summarize(mv: dict) -> dict:
        m, t = mv["m"], mv["t"]
        slope = float((m * t).sum() / (t * t).sum())
        cos = float((m * t).sum() / (np.linalg.norm(m) * np.linalg.norm(t)))
        return {"magnitude": float(np.abs(m).mean()), "slope": slope, "cos": cos,
                "target_size": float(np.abs(t).mean())}

    prose500 = {}  # receiver -> list of cells (two styles)
    for r, g in grid[grid["arm"] == "placebo"].groupby("receiver"):
        prose500.setdefault(r, []).append(tbl_of(g))
    for r, g in runs[(runs["arm"] == "placebo_mixed") & (runs["length"] == 500)].groupby("receiver"):
        prose500.setdefault(r, []).append(tbl_of(g))
    n500 = grid[(grid["arm"] == "neutral") & (grid["length"] == 500) & (grid["regen"] == 0)]
    rows, desc_mv, prose_mv = [], [], []
    for (r, a), g in n500.groupby(["receiver", "author"]):
        if r == a or len(g) < 400:
            continue
        mv = movement(r, a, tbl_of(g))
        if mv is None:
            continue
        s = summarize(mv)
        pm = [movement(r, a, c) for c in prose500.get(r, [])]
        pm = [x for x in pm if x]
        ps = [summarize(x) for x in pm]
        noise = float(np.mean([abs(ref[r][i] - retest[r][i]) for i in mv["items"]]))
        row = {"receiver": r, "author": a, "grid": GRID_OF[r], **{f"desc_{k}": v for k, v in s.items()},
               "prose_magnitude": float(np.mean([p["magnitude"] for p in ps])) if ps else None,
               "prose_slope": float(np.mean([p["slope"] for p in ps])) if ps else None,
               "prose_cos": float(np.mean([p["cos"] for p in ps])) if ps else None,
               "retest_noise_magnitude": noise}
        rows.append(row)
        desc_mv.append(mv)
        prose_mv.append(pm)
    # joint bootstrap over items for pooled slope (desc) and pooled slope difference (desc - prose)
    common = sorted(set.intersection(*([set(mv["items"]) for mv in desc_mv]
                                        + [set(x["items"]) for pm in prose_mv for x in pm])))
    M = np.array([[dict(zip(mv["items"], mv["m"]))[i] for i in common] for mv in desc_mv])
    Tt = np.array([[dict(zip(mv["items"], mv["t"]))[i] for i in common] for mv in desc_mv])
    P = np.array([np.mean([[dict(zip(x["items"], x["m"]))[i] for i in common] for x in pm], axis=0)
                  for pm in prose_mv])  # mean prose movement per pair

    def pooled_slope(Mx, Tx, ix):
        return float(np.mean([(Mx[k, ix] * Tx[k, ix]).sum() / (Tx[k, ix] ** 2).sum() for k in range(Mx.shape[0])]))

    full = np.arange(len(common))
    boots_d = np.array([pooled_slope(M, Tt, ix) for ix in rng.integers(0, len(common), size=(2000, len(common)))])
    boots_p = np.array([pooled_slope(P, Tt, ix) for ix in rng.integers(0, len(common), size=(2000, len(common)))])
    boots_diff = np.array([pooled_slope(M, Tt, ix) - pooled_slope(P, Tt, ix)
                           for ix in rng.integers(0, len(common), size=(2000, len(common)))])
    mv_summary = {
        "n_pairs": len(rows),
        "desc_magnitude": float(np.mean([r["desc_magnitude"] for r in rows])),
        "prose_magnitude": float(np.mean([r["prose_magnitude"] for r in rows])),
        "retest_noise_magnitude": float(np.mean([r["retest_noise_magnitude"] for r in rows])),
        "target_size": float(np.mean([r["desc_target_size"] for r in rows])),
        "desc_slope": pooled_slope(M, Tt, full), "desc_slope_ci": [float(np.percentile(boots_d, 2.5)), float(np.percentile(boots_d, 97.5))],
        "prose_slope": pooled_slope(P, Tt, full), "prose_slope_ci": [float(np.percentile(boots_p, 2.5)), float(np.percentile(boots_p, 97.5))],
        "slope_diff": pooled_slope(M, Tt, full) - pooled_slope(P, Tt, full),
        "slope_diff_ci": [float(np.percentile(boots_diff, 2.5)), float(np.percentile(boots_diff, 97.5))],
        "desc_cos": float(np.mean([r["desc_cos"] for r in rows])),
        "prose_cos": float(np.mean([r["prose_cos"] for r in rows])),
        "n_pairs_slope_pos": int(sum(r["desc_slope"] > 0 for r in rows)),
        "n_pairs_desc_slope_gt_prose": int(sum(r["desc_slope"] > r["prose_slope"] for r in rows)),
        "n_pairs_desc_mag_gt_prose": int(sum(r["desc_magnitude"] > r["prose_magnitude"] for r in rows)),
    }
    out["movement"] = {"summary": mv_summary, "pairs": rows}
    print("\n2. MOVEMENT at neutral-500 (m = p_cell - p_retest; t = p_author - p_receiver)")
    print(f"  mean |m| description {mv_summary['desc_magnitude']:.3f}  unrelated prose {mv_summary['prose_magnitude']:.3f}"
          f"  retest noise {mv_summary['retest_noise_magnitude']:.3f}  mean |t| {mv_summary['target_size']:.3f}")
    print(f"  slope of m on t: description {mv_summary['desc_slope']:+.3f} {mv_summary['desc_slope_ci']}  "
          f"prose {mv_summary['prose_slope']:+.3f} {mv_summary['prose_slope_ci']}  diff {mv_summary['slope_diff']:+.3f} {mv_summary['slope_diff_ci']}")
    print(f"  pairs with slope>0: {mv_summary['n_pairs_slope_pos']}/{len(rows)}; desc slope > prose slope: "
          f"{mv_summary['n_pairs_desc_slope_gt_prose']}/{len(rows)}; desc |m| > prose |m|: {mv_summary['n_pairs_desc_mag_gt_prose']}/{len(rows)}")
    for r in rows:
        print(f"    {r['author']:>14} -> {r['receiver']:<16} |m| {r['desc_magnitude']:.3f} (prose {r['prose_magnitude']:.3f})"
              f"  slope {r['desc_slope']:+.3f} (prose {r['prose_slope']:+.3f})  cos {r['desc_cos']:+.2f}")

    # ---------- 3. common receiver shift ----------
    def vec(r: str, cell: dict, keys: list[str]) -> np.ndarray:
        return np.array([cell[i] - ref[r][i] for i in keys])

    def mean_corr(vs: list[np.ndarray]) -> float | None:
        cs = [np.corrcoef(x, y)[0, 1] for x, y in combinations(vs, 2)]
        return float(np.mean(cs)) if cs else None

    def cross_corr(a: list[np.ndarray], b: list[np.ndarray]) -> float | None:
        cs = [np.corrcoef(x, y)[0, 1] for x in a for y in b]
        return float(np.mean(cs)) if cs else None

    common_rows = []
    for r in sorted(ref):
        if r not in set(grid["receiver"]):
            continue
        keys = sorted(ref[r])
        # descriptions of OTHER authors (neutral + framed, all lengths), one vector per cell
        others = {}
        sel = grid[(grid["receiver"] == r) & (grid["author"] != r) & grid["arm"].isin(["neutral", "framed"]) & (grid["regen"] == 0)]
        for (a, arm, n), g in sel.groupby(["author", "arm", "length"]):
            c = tbl_of(g)
            if len(set(c) & set(keys)) == len(keys):
                others.setdefault(a, []).append(vec(r, c, keys))
        own = []
        sel = grid[(grid["receiver"] == r) & (grid["author"] == r) & grid["arm"].isin(["neutral", "framed"]) & (grid["regen"] == 0)]
        for (arm, n), g in sel.groupby(["arm", "length"]):
            c = tbl_of(g)
            if len(set(c) & set(keys)) == len(keys):
                own.append(vec(r, c, keys))
        prose = []
        for g in [grid[(grid["receiver"] == r) & (grid["arm"] == "placebo")]] + \
                 [gg for _, gg in runs[(runs["receiver"] == r) & runs["arm"].isin(["placebo_mixed", "placebo_expo"])].groupby(["arm", "length"])]:
            if len(g):
                c = tbl_of(g)
                if len(set(c) & set(keys)) == len(keys):
                    prose.append(vec(r, c, keys))
        # across different authors only
        diff_author = [np.corrcoef(x, y)[0, 1] for a1, a2 in combinations(sorted(others), 2)
                       for x in others[a1] for y in others[a2]]
        all_desc = [v for vs in others.values() for v in vs] + own
        row = {"receiver": r, "n_desc_cells": len(all_desc), "n_prose_cells": len(prose),
               "corr_desc_different_authors": float(np.mean(diff_author)) if diff_author else None,
               "corr_own_vs_others": cross_corr(own, [v for vs in others.values() for v in vs]),
               "corr_prose_prose": mean_corr(prose), "corr_desc_prose": cross_corr(all_desc, prose)}
        common_rows.append(row)
    keys_c = ["corr_desc_different_authors", "corr_own_vs_others", "corr_prose_prose", "corr_desc_prose"]
    common_summary = {k: float(np.mean([r[k] for r in common_rows if r[k] is not None])) for k in keys_c}
    common_summary["n_receivers_desc_gt_prose"] = int(sum(
        r["corr_desc_different_authors"] > r["corr_prose_prose"] for r in common_rows
        if r["corr_desc_different_authors"] is not None and r["corr_prose_prose"] is not None))
    out["common_shift"] = {"summary": common_summary, "receivers": common_rows}
    print("\n3. COMMON RECEIVER SHIFT (correlation of per-item movement vectors, within receiver)")
    print(f"  {'receiver':<16}{'desc(diff authors)':>20}{'own vs others':>15}{'prose-prose':>13}{'desc-prose':>12}")
    for r in common_rows:
        f = lambda v: f"{v:+.2f}" if v is not None else "  n/a"
        print(f"  {r['receiver']:<16}{f(r['corr_desc_different_authors']):>20}{f(r['corr_own_vs_others']):>15}"
              f"{f(r['corr_prose_prose']):>13}{f(r['corr_desc_prose']):>12}  (desc cells {r['n_desc_cells']}, prose {r['n_prose_cells']})")
    print("  MEAN " + " ".join(f"{k}={v:+.2f}" for k, v in common_summary.items() if isinstance(v, float)))

    # ---------- 4. per-category T at neutral-500 ----------
    cat_pairs = defaultdict(list)
    all_pairs = []
    for (r, a), g in n500.groupby(["receiver", "author"]):
        if r == a or len(g) < 400:
            continue
        d = per_item_T(a, r, tbl_of(g))
        if not d:
            continue
        all_pairs.append(d)
        for cat in set(category.values()):
            sub = {i: v for i, v in d.items() if category.get(i) == cat}
            if sub:
                cat_pairs[cat].append(sub)
    percat = {}
    print("\n4. PER-CATEGORY T at neutral-500 (20 pairs, joint item bootstrap within category)")
    for cat in sorted(cat_pairs, key=lambda c: -len(set().union(*cat_pairs[c]))):
        res = joint_pooled(cat_pairs[cat])
        if res:
            percat[cat] = res
            print(f"  {cat:<18} items {res['n_items']:>3}  T {res['mean']:+.3f}  [{res['ci'][0]:+.3f}, {res['ci'][1]:+.3f}]")
    out["per_category_T"] = percat

    # ---------- 5. author identifiability ----------
    trials = []
    sel = grid[(grid["author"].notna()) & grid["arm"].isin(["neutral", "framed"]) & (grid["regen"] == 0)]
    for (r, a, arm, n), g in sel.groupby(["receiver", "author", "arm", "length"]):
        cands = partners(r)
        if r == a or a not in cands or len(cands) < 2 or len(g) < 400:
            continue
        c = tbl_of(g)
        scores = {}
        for cand in cands:
            f = ag.fidelity(ref[cand], c)
            if f:
                scores[cand] = f["F"]
        if len(scores) < 2:
            continue
        # relative to the receiver's generic baseline toward each candidate
        rel = {cand: scores[cand] - ag.fidelity(ref[cand], ref[r])["F"] for cand in scores}
        trials.append({"receiver": r, "author": a, "cond": f"{arm}-{n}", "grid": GRID_OF[r],
                       "nearest_by_F": max(scores, key=scores.get) == a,
                       "nearest_by_gain": max(rel, key=rel.get) == a})
    n = len(trials)
    k_f = sum(t["nearest_by_F"] for t in trials)
    k_g = sum(t["nearest_by_gain"] for t in trials)
    p_bin = lambda k: float(sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n)
    by_cond = defaultdict(lambda: [0, 0])
    for t in trials:
        by_cond[t["cond"]][0] += t["nearest_by_gain"]
        by_cond[t["cond"]][1] += 1
    ident = {"n_trials": n, "nearest_by_F": k_f, "p_by_F": p_bin(k_f), "nearest_by_gain": k_g, "p_by_gain": p_bin(k_g),
             "by_cond": {c: {"correct": v[0], "n": v[1]} for c, v in sorted(by_cond.items())}, "chance": 0.5}
    out["identifiability"] = ident
    print(f"\n5. AUTHOR IDENTIFIABILITY (nearest of 2 candidate authors; chance 0.5): "
          f"by F {k_f}/{n} (p={ident['p_by_F']:.3f}); by gain over baseline {k_g}/{n} (p={ident['p_by_gain']:.3f})")
    for c, v in ident["by_cond"].items():
        print(f"    {c:<14} {v['correct']}/{v['n']}")

    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
