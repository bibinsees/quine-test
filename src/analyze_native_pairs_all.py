"""Per-pair transfer contrasts for every main-grid self-description condition.

Same fidelity, baseline and paired item bootstrap as analyze_grid.py (which
tabulates neutral-500 only), applied to neutral-100/500/2000, framed-100/500/2000,
paraphrase-500, adopt-500 and third-person-500 on the merged grid log. Writes
results/arxiv/native_pairs_all.json and prints the significantly positive pairs,
which the Limitations section counts.
"""

from __future__ import annotations

import json

import numpy as np

import analyze_grid as ag

ag.GRID = ag.ROOT / "results" / "arxiv" / "grid_merged.jsonl"
OUT = ag.ROOT / "results" / "arxiv" / "native_pairs_all.json"


def main() -> None:
    ag.RNG = np.random.default_rng(14)
    grid = ag.load_jsonl(ag.GRID)
    screen = ag.load_jsonl(ag.SCREEN)
    final_items = set(grid["item_id"].unique())
    ref = ag.p_hat_table(screen[screen["item_id"].isin(final_items)], "short")
    for model, tbl in ag.p_hat_table(grid[grid["arm"] == "baseline"], "receiver").items():
        ref[model] = tbl
    base = {}
    for a in ref:
        for r in ref:
            if a != r:
                fid = ag.fidelity(ref[a], ref[r])
                if fid:
                    base[(a, r)] = fid

    cells = grid[(~grid["arm"].isin(["baseline", "retest", "placebo"])) & (grid["regen"] == 0)]
    todo = []
    for cell_id, g in cells.groupby("cell_id"):
        meta = g.iloc[0]
        if meta["receiver"] == meta["author"] or meta["author"] not in ref:
            continue
        ok = g[g["choice"].notna()]
        tbl = {i: (gg["choice"] == "option_a").mean() for i, gg in ok.groupby("item_id")}
        todo.append((f"{meta['arm']}-{meta['length']}", meta["author"], meta["receiver"], tbl))

    out: dict = {}
    for cond, a, r, tbl in sorted(todo, key=lambda x: (x[0], x[1], x[2])):
        fid = ag.fidelity(ref[a], tbl)
        if fid is None or (a, r) not in base:
            continue
        c = ag.boot_contrast(fid["per_item"], base[(a, r)]["per_item"], fid["items"], base[(a, r)]["items"])
        out.setdefault(cond, {})[f"{a}->{r}"] = {"F_cross": fid["F"], "B": base[(a, r)]["F"], **c}

    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for cond, pairs in out.items():
        pos = {k: v for k, v in pairs.items() if v["ci"][0] > 0}
        neg = sum(v["ci"][1] < 0 for v in pairs.values())
        print(f"{cond}: {len(pairs)} cross pairs, {len(pos)} significantly positive, {neg} negative")
        for k, v in sorted(pos.items(), key=lambda kv: -kv[1]["delta"]):
            print(f"   + {k}: {v['delta']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] (n={v['n']})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
