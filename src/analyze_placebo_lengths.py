"""Unrelated-prose ablation analysis, both designs, with paired bootstraps.

Table-2 design (per author A -> receiver R pair of each grid, per length L):
  B            R generic vs A
  placebo_L    R running the mixed placebo of length L, vs A
  desc_L       R running A's neutral-L self-description, vs A   (= F_cross)
  contrasts    placebo_L - B, desc_L - placebo_L   (paired item bootstrap)
Table-8 design (per model M, per length L):
  placebo_L vs M's own baseline, own-desc_L vs own baseline, contrast own-desc_L - placebo_L.
Reads runs/arxiv_placebo_lengths.jsonl; writes results/arxiv/placebo_lengths.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_grid as ag

ROOT = Path(__file__).resolve().parent.parent
MERGED = ROOT / "results" / "arxiv" / "grid_merged.jsonl"
RUNS = ROOT / "runs" / "arxiv_placebo_lengths.jsonl"
OUT = ROOT / "results" / "arxiv" / "placebo_lengths.json"
GRIDS = {"A": ag.GRID_A, "B": ag.GRID_B, "local": ag.LOCALS,
         "same-base": ["hermes3-70b", "llama3.3-70b"]}  # the same 20 cross pairs as Table 2
LENGTHS = [100, 500, 2000]


def tbl_of(g):
    ok = g[g["choice"].notna()]
    return {i: (gg["choice"] == "option_a").mean() for i, gg in ok.groupby("item_id")}


def seed_rng(key: str) -> None:
    ag.RNG = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))


def sig(ci):
    return "+" if ci[0] > 0 else ("-" if ci[1] < 0 else " ")


def main() -> None:
    grid = ag.load_jsonl(MERGED if MERGED.exists() else ag.GRID)
    screen = ag.load_jsonl(ag.SCREEN)
    runs = ag.load_jsonl(RUNS)
    final_items = set(grid["item_id"].unique())
    ref = ag.p_hat_table(screen[screen["item_id"].isin(final_items)], "short")
    for model, t in ag.p_hat_table(grid[grid["arm"] == "baseline"], "receiver").items():
        ref[model] = t
    plac = {}
    for (r, n), g in runs.groupby(["receiver", "length"]):
        if len(g) >= 440:
            plac[(r, int(n))] = tbl_of(g)

    def desc_tbl(r, a, n):
        cell = grid[(grid["receiver"] == r) & (grid["author"] == a) & (grid["arm"] == "neutral")
                    & (grid["length"] == n) & (grid["regen"] == 0)]
        return tbl_of(cell) if len(cell) else None

    out = {"table2": {"pairs": [], "pooled": {}}, "table8": {"models": [], "pooled": {}}}

    print("=== Table-2 design: receiver scored against the AUTHOR ===")
    for n in LENGTHS:
        rows = []
        for gname, models in GRIDS.items():
            for a in models:
                for r in models:
                    if a == r or (r, n) not in plac:
                        continue
                    b = ag.fidelity(ref[a], ref[r])
                    p = ag.fidelity(ref[a], plac[(r, n)])
                    td = desc_tbl(r, a, n)
                    d = ag.fidelity(ref[a], td) if td else None
                    seed_rng(f"{a}|{r}|{n}|pb")
                    c_pb = ag.boot_contrast(p["per_item"], b["per_item"], p["items"], b["items"])
                    c_dp = None
                    if d:
                        seed_rng(f"{a}|{r}|{n}|dp")
                        c_dp = ag.boot_contrast(d["per_item"], p["per_item"], d["items"], p["items"])
                    rows.append({"grid": gname, "author": a, "receiver": r, "length": n, "B": b["F"],
                                 "placebo": p["F"], "desc": d["F"] if d else None,
                                 "placebo_minus_B": c_pb, "desc_minus_placebo": c_dp})
        if not rows:
            print(f"  length {n}: no complete cells yet")
            continue
        print(f"\n--- length {n}: {len(rows)} pairs ---")
        print(f"{'grid':<6} {'author -> receiver':<34} {'B':>6} {'placebo':>8} {'desc':>6}  {'plac-B':>22}  {'desc-plac':>22}")
        for x in rows:
            pb, dp = x["placebo_minus_B"], x["desc_minus_placebo"]
            desc_s = f"{x['desc']:.3f}" if x["desc"] is not None else "n/a"
            pb_s = f"{sig(pb['ci'])}{pb['delta']:+.3f} [{pb['ci'][0]:+.3f},{pb['ci'][1]:+.3f}]"
            dp_s = (f"{sig(dp['ci'])}{dp['delta']:+.3f} [{dp['ci'][0]:+.3f},{dp['ci'][1]:+.3f}]"
                    if dp else "n/a")
            print(f"{x['grid']:<6} {x['author'] + ' -> ' + x['receiver']:<34} {x['B']:>6.3f} "
                  f"{x['placebo']:>8.3f} {desc_s:>6}  {pb_s:>22}  {dp_s:>22}")
        mB = float(np.mean([x["B"] for x in rows]))
        mP = float(np.mean([x["placebo"] for x in rows]))
        ds = [x["desc"] for x in rows if x["desc"] is not None]
        mD = float(np.mean(ds)) if ds else None
        pooled = {"n_pairs": len(rows), "B": mB, "placebo": mP, "desc": mD,
                  "placebo_minus_B": mP - mB, "desc_minus_placebo": (mD - mP) if mD is not None else None,
                  "pb_sig_pos": sum(1 for x in rows if x["placebo_minus_B"]["ci"][0] > 0),
                  "pb_sig_neg": sum(1 for x in rows if x["placebo_minus_B"]["ci"][1] < 0),
                  "dp_sig_pos": sum(1 for x in rows if x["desc_minus_placebo"] and x["desc_minus_placebo"]["ci"][0] > 0),
                  "dp_sig_neg": sum(1 for x in rows if x["desc_minus_placebo"] and x["desc_minus_placebo"]["ci"][1] < 0)}
        print(f"POOLED length {n}: B={mB:.3f} placebo={mP:.3f} desc={(f'{mD:.3f}' if mD else 'n/a')}  "
              f"placebo-B={mP-mB:+.3f} (sig +{pooled['pb_sig_pos']}/-{pooled['pb_sig_neg']})  "
              f"desc-placebo={(f'{mD-mP:+.3f}' if mD else 'n/a')} (sig +{pooled['dp_sig_pos']}/-{pooled['dp_sig_neg']})")
        out["table2"]["pairs"] += rows
        out["table2"]["pooled"][str(n)] = pooled

    print("\n=== Table-8 design: model scored against ITSELF ===")
    retest = {}
    for m, g in grid[grid["arm"] == "retest"].groupby("receiver"):
        fid = ag.fidelity(ref[m], tbl_of(g))
        if fid:
            retest[m] = fid["F"]
    for n in LENGTHS:
        rows = []
        for m in sorted(ref):
            if (m, n) not in plac:
                continue
            p = ag.fidelity(ref[m], plac[(m, n)])
            td = desc_tbl(m, m, n)
            d = ag.fidelity(ref[m], td) if td else None
            c = None
            if d:
                seed_rng(f"{m}|{n}|own-dp")
                c = ag.boot_contrast(d["per_item"], p["per_item"], d["items"], p["items"])
            rows.append({"model": m, "length": n, "retest": retest.get(m), "placebo": p["F"],
                         "own_desc": d["F"] if d else None, "own_desc_minus_placebo": c})
        if not rows:
            continue
        print(f"\n--- length {n} ---")
        print(f"{'model':<16} {'retest':>7} {'placebo':>8} {'own desc':>9}  {'owndesc-plac':>22}")
        for x in rows:
            c = x["own_desc_minus_placebo"]
            rt_s = f"{x['retest']:.3f}" if x["retest"] else "n/a"
            od_s = f"{x['own_desc']:.3f}" if x["own_desc"] is not None else "n/a"
            c_s = f"{sig(c['ci'])}{c['delta']:+.3f} [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}]" if c else "n/a"
            print(f"{x['model']:<16} {rt_s:>7} {x['placebo']:>8.3f} {od_s:>9}  {c_s:>22}")
        mP = float(np.mean([x["placebo"] for x in rows]))
        ds = [x["own_desc"] for x in rows if x["own_desc"] is not None]
        mD = float(np.mean(ds)) if ds else None
        mC = float(np.mean([x["retest"] for x in rows if x["retest"]]))
        pooled = {"n_models": len(rows), "retest": mC, "placebo": mP, "own_desc": mD,
                  "own_desc_minus_placebo": (mD - mP) if mD is not None else None,
                  "sig_neg": sum(1 for x in rows if x["own_desc_minus_placebo"] and x["own_desc_minus_placebo"]["ci"][1] < 0),
                  "sig_pos": sum(1 for x in rows if x["own_desc_minus_placebo"] and x["own_desc_minus_placebo"]["ci"][0] > 0)}
        print(f"POOLED length {n}: retest={mC:.3f} placebo={mP:.3f} own_desc={(f'{mD:.3f}' if mD else 'n/a')}  "
              f"owndesc-placebo={(f'{mD-mP:+.3f}' if mD else 'n/a')} (sig +{pooled['sig_pos']}/-{pooled['sig_neg']})")
        out["table8"]["models"] += rows
        out["table8"]["pooled"][str(n)] = pooled

    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
