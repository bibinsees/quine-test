"""Score the existing placebo-500 runs (unrelated prose) the Table-2 way: receiver R
running the placebo, compared with each author A of R's grid. Side by side with
B (R generic vs A), F_cross (R running A's neutral-500 self-description vs A),
and a third column: R running ANOTHER model C's description, scored against A
(the specificity comparison). Analysis only, no new calls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import analyze_grid as ag

ROOT = Path(__file__).resolve().parent.parent
MERGED = ROOT / "results" / "arxiv" / "grid_merged.jsonl"
OUT = ROOT / "results" / "arxiv" / "placebo_transfer.json"
GRIDS = {"A": ag.GRID_A, "B": ag.GRID_B, "local": ag.LOCALS}


def tbl_of(g):
    ok = g[g["choice"].notna()]
    return {i: (gg["choice"] == "option_a").mean() for i, gg in ok.groupby("item_id")}


def main() -> None:
    grid = ag.load_jsonl(MERGED if MERGED.exists() else ag.GRID)
    screen = ag.load_jsonl(ag.SCREEN)
    final_items = set(grid["item_id"].unique())
    ref = ag.p_hat_table(screen[screen["item_id"].isin(final_items)], "short")
    for model, t in ag.p_hat_table(grid[grid["arm"] == "baseline"], "receiver").items():
        ref[model] = t
    plac = {r: tbl_of(g) for r, g in grid[grid["arm"] == "placebo"].groupby("receiver")}

    def desc_tbl(r, a):
        cell = grid[(grid["receiver"] == r) & (grid["author"] == a) & (grid["arm"] == "neutral")
                    & (grid["length"] == 500) & (grid["regen"] == 0)]
        return tbl_of(cell) if len(cell) else None

    out = {"pairs": [], "pooled": {}}
    allrows = []
    print(f"{'grid':<6} {'author -> receiver':<34} {'B':>6} {'placebo':>8} {'A desc':>7} {'C desc':>7}")
    for gname, models in GRIDS.items():
        rows = []
        for a in models:
            for r in models:
                if a == r or r not in plac:
                    continue
                c = [m for m in models if m not in (a, r)][0]
                b = ag.fidelity(ref[a], ref[r])
                p = ag.fidelity(ref[a], plac[r])
                ta, tc = desc_tbl(r, a), desc_tbl(r, c)
                x = ag.fidelity(ref[a], ta)["F"] if ta else None
                y = ag.fidelity(ref[a], tc)["F"] if tc else None
                d = ag.boot_contrast(p["per_item"], b["per_item"], p["items"], b["items"])
                row = {"grid": gname, "author": a, "receiver": r, "third": c, "B": b["F"],
                       "placebo": p["F"], "own_author_desc": x, "third_desc": y,
                       "placebo_minus_B": d["delta"], "placebo_minus_B_ci": d["ci"]}
                rows.append(row)
                print(f"{gname:<6} {a + ' -> ' + r:<34} {b['F']:>6.3f} {p['F']:>8.3f} "
                      f"{(f'{x:.3f}' if x else 'n/a'):>7} {(f'{y:.3f}' if y else 'n/a'):>7}  "
                      f"plac-B {d['delta']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]")
        m = {k: float(np.mean([r[k] for r in rows if r[k] is not None]))
             for k in ("B", "placebo", "own_author_desc", "third_desc")}
        nsig_neg = sum(1 for r in rows if r["placebo_minus_B_ci"][1] < 0)
        nsig_pos = sum(1 for r in rows if r["placebo_minus_B_ci"][0] > 0)
        print(f"{gname:<6} {'MEAN':<34} {m['B']:>6.3f} {m['placebo']:>8.3f} {m['own_author_desc']:>7.3f} "
              f"{m['third_desc']:>7.3f}  placebo-B {m['placebo']-m['B']:+.3f} "
              f"(sig +{nsig_pos}/-{nsig_neg} of {len(rows)})\n")
        out["pooled"][gname] = {**m, "n_pairs": len(rows), "sig_pos": nsig_pos, "sig_neg": nsig_neg}
        allrows += rows
    m = {k: float(np.mean([r[k] for r in allrows if r[k] is not None]))
         for k in ("B", "placebo", "own_author_desc", "third_desc")}
    nsig_neg = sum(1 for r in allrows if r["placebo_minus_B_ci"][1] < 0)
    nsig_pos = sum(1 for r in allrows if r["placebo_minus_B_ci"][0] > 0)
    print(f"{'ALL':<6} {'MEAN over ' + str(len(allrows)) + ' pairs':<34} {m['B']:>6.3f} {m['placebo']:>8.3f} "
          f"{m['own_author_desc']:>7.3f} {m['third_desc']:>7.3f}  placebo-B {m['placebo']-m['B']:+.3f} "
          f"(sig +{nsig_pos}/-{nsig_neg})")
    out["pooled"]["all"] = {**m, "n_pairs": len(allrows), "sig_pos": nsig_pos, "sig_neg": nsig_neg}
    out["pairs"] = allrows
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
