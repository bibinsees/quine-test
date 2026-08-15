"""Main-grid analysis: the decomposition, dose-response, and all contrasts.

Reads runs/grid.jsonl (+ runs/screening.jsonl for API-model baselines and
reference behavior) and produces:
  1. Reference behavior per model (generic prompt): p_ref[model][item]
  2. Fidelity of every cell vs its AUTHOR's reference:
       F = 1 - mean_i JSD(p_ref_author, p_cell)   (primary)
       + Pearson r and majority agreement (companions)
  3. The decomposition per condition: B, F_cross, F_self,
       T_script = F_cross - B,  R_weight = F_self - F_cross
     with 10k-item-bootstrap CIs on each contrast.
  4. Dose-response table: actual description words vs F (per arm).
  5. Specificity gain: F(A-ref vs B+desc_A) - F(A-ref vs B+desc_C).
  6. Same-base test: hermes3<->llama3.3 F_cross vs unrelated-pair F_cross.
  7. Framed-vs-neutral contrast at 500 words.
Writes results/grid_summary.json and prints tables. Safe on partial data
(cells with < MIN_ITEMS items covered are skipped and listed).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GRID = ROOT / "runs" / "grid.jsonl"
SCREEN = ROOT / "runs" / "screening.jsonl"
MANIFEST = ROOT / "descriptions" / "manifest.jsonl"
OUT = ROOT / "results" / "grid_summary.json"

MIN_ITEMS = 60          # skip cells with fewer covered items (partial data)
N_BOOT = 10_000
RNG = np.random.default_rng(14)  # fixed seed for reproducibility

GRID_A = ["deepseek-v3", "glm-4.5-air", "grok-4.3"]
GRID_B = ["gpt5.2", "kimi-k2.5-nr", "haiku-4.5"]
LOCALS = ["llama3.3-70b", "gemma4-31b", "mistral32-local"]
SAME_BASE = {"hermes3-70b", "llama3.3-70b"}


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    df = pd.DataFrame(rows)
    return df.sort_values("ts").drop_duplicates("call_id", keep="last")


def p_hat_table(df: pd.DataFrame, group_col: str) -> dict:
    """{group: {item_id: p_hat_option_a}} using parsed choices only."""
    out: dict = defaultdict(dict)
    ok = df[df["choice"].notna()]
    for (grp, item), g in ok.groupby([group_col, "item_id"]):
        out[grp][item] = (g["choice"] == "option_a").mean()
    return out


def jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    def h(x):
        x = np.clip(x, 1e-12, 1 - 1e-12)
        return -(x * np.log2(x) + (1 - x) * np.log2(1 - x))
    m = (p + q) / 2
    return h(m) - (h(p) + h(q)) / 2


def fidelity(ref: dict, cell: dict) -> dict | None:
    """Fidelity of cell p-hats vs reference p-hats. Returns per-item arrays too."""
    common = sorted(set(ref) & set(cell))
    if len(common) < MIN_ITEMS:
        return None
    p = np.array([ref[i] for i in common])
    q = np.array([cell[i] for i in common])
    per_item = 1 - jsd(p, q)
    r = float(np.corrcoef(p, q)[0, 1]) if p.std() > 0 and q.std() > 0 else np.nan
    return {"F": float(per_item.mean()), "r": r,
            "agree": float(np.mean((p >= 0.5) == (q >= 0.5))),
            "n_items": len(common), "per_item": per_item, "items": common}


def boot_ci(per_item: np.ndarray) -> tuple[float, float]:
    idx = RNG.integers(0, len(per_item), size=(N_BOOT, len(per_item)))
    means = per_item[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_contrast(a: np.ndarray, b: np.ndarray, items_a: list, items_b: list) -> dict:
    """Paired-where-possible bootstrap for mean(a) - mean(b)."""
    common = sorted(set(items_a) & set(items_b))
    map_a = dict(zip(items_a, a))
    map_b = dict(zip(items_b, b))
    da = np.array([map_a[i] for i in common])
    db = np.array([map_b[i] for i in common])
    diff = da - db
    idx = RNG.integers(0, len(diff), size=(N_BOOT, len(diff)))
    means = diff[idx].mean(axis=1)
    return {"delta": float(diff.mean()),
            "ci": [float(np.percentile(means, 2.5)),
                   float(np.percentile(means, 97.5))],
            "p_gt_0": float((means > 0).mean()), "n": len(common)}


def main() -> None:
    grid = load_jsonl(GRID)
    screen = load_jsonl(SCREEN)
    words = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("ok"):
            words[rec["desc_id"]] = rec["words"]

    final_items = set(grid["item_id"].unique())

    # ---- reference behavior (generic prompt) per model
    ref = p_hat_table(screen[screen["item_id"].isin(final_items)], "short")
    base_cells = grid[grid["arm"] == "baseline"]
    for model, tbl in p_hat_table(base_cells, "receiver").items():
        ref[model] = tbl  # locals + hermes3 baselines come from the grid

    # ---- perturbation ladder: retest ceiling vs placebo vs self-description
    ladder = {}
    for arm in ("retest", "placebo"):
        for receiver, g in grid[grid["arm"] == arm].groupby("receiver"):
            ok = g[g["choice"].notna()]
            tbl = {i: (gg["choice"] == "option_a").mean()
                   for i, gg in ok.groupby("item_id")}
            fid = fidelity(ref.get(receiver, {}), tbl)
            if fid:
                ladder.setdefault(receiver, {})[arm] = fid["F"]

    # ---- per-cell fidelity vs author reference
    cells = grid[~grid["arm"].isin(["baseline", "retest", "placebo"])]
    cell_p = {}
    for cell_id, g in cells.groupby("cell_id"):
        ok = g[g["choice"].notna()]
        tbl = {i: (gg["choice"] == "option_a").mean()
               for i, gg in ok.groupby("item_id")}
        meta = g.iloc[0]
        cell_p[cell_id] = (meta, tbl)

    results, skipped = [], []
    for cell_id, (meta, tbl) in cell_p.items():
        author = meta["author"]
        if author not in ref:
            skipped.append((cell_id, "no author reference"))
            continue
        fid = fidelity(ref[author], tbl)
        if fid is None:
            skipped.append((cell_id, "insufficient items"))
            continue
        lo, hi = boot_ci(fid["per_item"])
        desc_id = f"{author}|{meta['arm']}|{meta['length']}|{meta['regen']}"
        results.append({
            "cell_id": cell_id, "receiver": meta["receiver"], "author": author,
            "arm": meta["arm"], "length": int(meta["length"]),
            "regen": int(meta["regen"]),
            "kind": "self" if meta["receiver"] == author else "cross",
            "desc_words": words.get(desc_id),
            **{k: fid[k] for k in ("F", "r", "agree", "n_items")},
            "F_ci": [lo, hi],
            "_per_item": fid["per_item"], "_items": fid["items"],
        })
    rdf = pd.DataFrame(results)

    # ---- baseline B per (author, receiver) pair: receiver's generic behavior
    B = {}
    for author in ref:
        for receiver in ref:
            if author == receiver:
                continue
            fid = fidelity(ref[author], ref[receiver])
            if fid:
                B[(author, receiver)] = fid

    print("=" * 78)
    print("1. DECOMPOSITION per condition (mean over pairs; F = 1 - JSD)")
    print("=" * 78)
    summary = {"conditions": {}}
    for (arm, length), g in rdf[rdf["regen"] == 0].groupby(["arm", "length"]):
        selfs = g[g["kind"] == "self"]
        crosses = g[g["kind"] == "cross"]
        b_vals = [B[(a, r)]["F"] for a, r in
                  zip(crosses["author"], crosses["receiver"]) if (a, r) in B]
        row = {"F_self": float(selfs["F"].mean()) if len(selfs) else None,
               "F_cross": float(crosses["F"].mean()) if len(crosses) else None,
               "B": float(np.mean(b_vals)) if b_vals else None,
               "n_self_cells": len(selfs), "n_cross_cells": len(crosses)}
        if row["F_self"] is not None and row["F_cross"] is not None:
            row["R_weight"] = row["F_self"] - row["F_cross"]
        if row["F_cross"] is not None and row["B"] is not None:
            row["T_script"] = row["F_cross"] - row["B"]
        summary["conditions"][f"{arm}-{length}"] = row
        print(f"  {arm}-{length:<5} F_self={row['F_self'] and round(row['F_self'],3)}"
              f"  F_cross={row['F_cross'] and round(row['F_cross'],3)}"
              f"  B={row['B'] and round(row['B'],3)}"
              f"  T_script={row.get('T_script') and round(row['T_script'],3)}"
              f"  R_weight={row.get('R_weight') and round(row['R_weight'],3)}"
              f"  ({row['n_self_cells']}s/{row['n_cross_cells']}c cells)")

    # ---- headline contrasts with bootstrap (neutral-500, all pairs pooled)
    print()
    print("=" * 78)
    print("2. HEADLINE CONTRASTS (neutral-500, item-level paired bootstrap)")
    print("=" * 78)
    n500 = rdf[(rdf["arm"] == "neutral") & (rdf["length"] == 500) & (rdf["regen"] == 0)]
    contrasts = {}
    for (a, r) in {(row["author"], row["receiver"])
                   for _, row in n500[n500["kind"] == "cross"].iterrows()}:
        cell = n500[(n500["author"] == a) & (n500["receiver"] == r)]
        if not len(cell) or (a, r) not in B:
            continue
        c = cell.iloc[0]
        key = f"{a}->{r}"
        contrasts[key] = boot_contrast(
            c["_per_item"], B[(a, r)]["per_item"], c["_items"], B[(a, r)]["items"])
    t_deltas = [v["delta"] for v in contrasts.values()]
    print(f"  T_script (F_cross - B) per directed pair: n={len(t_deltas)}, "
          f"mean delta={np.mean(t_deltas):.3f}" if t_deltas else "  (no pairs yet)")
    for k, v in sorted(contrasts.items(), key=lambda kv: -kv[1]["delta"]):
        sig = "*" if (v["ci"][0] > 0 or v["ci"][1] < 0) else " "
        print(f"   {sig} {k:<32} d={v['delta']:+.3f} CI[{v['ci'][0]:+.3f},"
              f"{v['ci'][1]:+.3f}] P(>0)={v['p_gt_0']:.3f}")
    summary["T_script_pairs_n500"] = {k: {kk: vv for kk, vv in v.items()}
                                      for k, v in contrasts.items()}

    # ---- dose-response
    print()
    print("=" * 78)
    print("3. DOSE-RESPONSE (neutral arm, actual words vs F)")
    print("=" * 78)
    dr = rdf[(rdf["arm"] == "neutral") & (rdf["regen"] == 0)]
    for kind in ("self", "cross"):
        sub = dr[dr["kind"] == kind].dropna(subset=["desc_words"])
        pts = sorted(zip(sub["desc_words"], sub["F"]))
        print(f"  {kind}: " + "  ".join(f"({int(w)}w, {f:.3f})" for w, f in pts[:24]))

    # ---- framed vs neutral at 500
    print()
    print("=" * 78)
    print("4. FRAMED vs NEUTRAL at 500 words (per receiver-author cell)")
    print("=" * 78)
    f500 = rdf[(rdf["arm"] == "framed") & (rdf["length"] == 500) & (rdf["regen"] == 0)]
    fn = []
    for _, frow in f500.iterrows():
        match = n500[(n500["author"] == frow["author"])
                     & (n500["receiver"] == frow["receiver"])]
        if len(match):
            fn.append((frow["author"], frow["receiver"], frow["kind"],
                       frow["F"], match.iloc[0]["F"]))
    for a, r, kind, ff, nf in sorted(fn, key=lambda x: -(x[3] - x[4])):
        print(f"  {kind:<5} {a:>16} -> {r:<16} framed={ff:.3f} neutral={nf:.3f} "
              f"d={ff-nf:+.3f}")
    if fn:
        d = [x[3] - x[4] for x in fn]
        summary["framed_minus_neutral_500"] = {"mean": float(np.mean(d)),
                                               "n_cells": len(d)}
        print(f"  MEAN framed-neutral = {np.mean(d):+.3f} over {len(d)} cells")

    # ---- specificity gain
    print()
    print("=" * 78)
    print("5. SPECIFICITY (neutral-500): F(A-ref | B+desc_A) - F(A-ref | B+desc_C)")
    print("=" * 78)
    gains = []
    for trio in (GRID_A, GRID_B, LOCALS):
        for a in trio:
            for b in trio:
                if a == b:
                    continue
                own = n500[(n500["author"] == a) & (n500["receiver"] == b)]
                if not len(own) or a not in ref:
                    continue
                for c in trio:
                    if c in (a, b):
                        continue
                    other_cell = n500[(n500["author"] == c) & (n500["receiver"] == b)]
                    if not len(other_cell):
                        continue
                    fid_other = fidelity(ref[a], dict(zip(other_cell.iloc[0]["_items"],
                                                          other_cell.iloc[0]["_per_item"])))
                    # fidelity of A-ref vs (B running C's desc), computed from raw p-hats
                    meta_tbl = cell_p.get(other_cell.iloc[0]["cell_id"])
                    fid_other = fidelity(ref[a], meta_tbl[1]) if meta_tbl else None
                    if fid_other:
                        gains.append(own.iloc[0]["F"] - fid_other["F"])
    if gains:
        print(f"  mean specificity gain = {np.mean(gains):+.3f} (n={len(gains)}; "
              f">0 means descriptions are author-specific, not generic)")
        summary["specificity_gain_n500"] = {"mean": float(np.mean(gains)),
                                            "n": len(gains)}

    # ---- same-base test
    print()
    print("=" * 78)
    print("6. SAME-BASE TEST (hermes3-70b <-> llama3.3-70b)")
    print("=" * 78)
    sb = rdf[(rdf["kind"] == "cross") & (rdf["regen"] == 0)
             & rdf["author"].isin(SAME_BASE) & rdf["receiver"].isin(SAME_BASE)]
    other_cross = rdf[(rdf["kind"] == "cross") & (rdf["regen"] == 0)
                      & ~(rdf["author"].isin(SAME_BASE) & rdf["receiver"].isin(SAME_BASE))]
    for (arm, length), g in sb.groupby(["arm", "length"]):
        comp = other_cross[(other_cross["arm"] == arm)
                           & (other_cross["length"] == length)]
        print(f"  {arm}-{length}: same-base F={g['F'].mean():.3f} vs "
              f"unrelated-pair F={comp['F'].mean():.3f} (n={len(g)}/{len(comp)})")

    # ---- perturbation ladder report
    if ladder:
        print()
        print("=" * 78)
        print("7. PERTURBATION LADDER: retest ceiling C vs placebo vs own description")
        print("   (each vs the receiver's own generic-prompt reference)")
        print("=" * 78)
        n500_self = {r.iloc[0]["receiver"]: r.iloc[0]["F"] for _, r in
                     rdf[(rdf["arm"] == "neutral") & (rdf["length"] == 500)
                         & (rdf["regen"] == 0)
                         & (rdf["kind"] == "self")].groupby("cell_id")}
        rows_l = []
        for receiver, vals in sorted(ladder.items()):
            c = vals.get("retest")
            p = vals.get("placebo")
            s = n500_self.get(receiver)
            rows_l.append((receiver, c, p, s))
            print(f"  {receiver:<16} retest_C={c and round(c,3)}  "
                  f"placebo={p and round(p,3)}  self_desc(n500)={s and round(s,3)}")
        cs = [r[1] for r in rows_l if r[1] is not None]
        ps = [r[2] for r in rows_l if r[2] is not None]
        ss = [r[3] for r in rows_l if r[3] is not None]
        print(f"  MEANS: retest_C={np.mean(cs):.3f}  placebo={np.mean(ps):.3f}  "
              f"self_desc={np.mean(ss):.3f}")
        summary["perturbation_ladder"] = {
            "retest_C_mean": float(np.mean(cs)) if cs else None,
            "placebo_mean": float(np.mean(ps)) if ps else None,
            "self_n500_mean": float(np.mean(ss)) if ss else None,
        }

    if skipped:
        print(f"\nskipped cells ({len(skipped)}):")
        for cid, why in skipped[:12]:
            print(f"  {cid}: {why}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nsummary written to {OUT}")


if __name__ == "__main__":
    main()
