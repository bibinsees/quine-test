"""Per model: how far does the model drift from its OWN generic-prompt profile
under each prefix? Columns: test-retest (same prompt, fresh seeds), no prompt,
placebo prose, its own neutral-500 self-description, another grid member's
neutral-500 self-description (mean over partners). All fidelities are against
the model's own generic-prompt reference, so this is 'how much does the prefix
move the model', not transfer to an author."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import analyze_grid as ag

ROOT = Path(__file__).resolve().parent.parent
MERGED = ROOT / "results" / "arxiv" / "grid_merged.jsonl"
NOSYS = ROOT / "runs" / "arxiv_nosys.jsonl"
GRIDS = [ag.GRID_A, ag.GRID_B, ag.LOCALS, ["hermes3-70b", "llama3.3-70b"]]


def tbl_of(g):
    ok = g[g["choice"].notna()]
    return {i: (gg["choice"] == "option_a").mean() for i, gg in ok.groupby("item_id")}


def main() -> None:
    grid = ag.load_jsonl(MERGED if MERGED.exists() else ag.GRID)
    screen = ag.load_jsonl(ag.SCREEN)
    nosys = ag.load_jsonl(NOSYS)
    final_items = set(grid["item_id"].unique())
    ref = ag.p_hat_table(screen[screen["item_id"].isin(final_items)], "short")
    for model, t in ag.p_hat_table(grid[grid["arm"] == "baseline"], "receiver").items():
        ref[model] = t

    def F(model, df):
        fid = ag.fidelity(ref[model], tbl_of(df)) if len(df) else None
        return fid["F"] if fid else None

    rows = []
    print(f"{'model':<16} {'retest':>7} {'no-prompt':>9} {'placebo':>8} {'own desc':>9} {'other desc':>11}")
    for model in sorted(ref):
        if model not in set(grid["receiver"]):
            continue
        partners = [m for grp in GRIDS if model in grp for m in grp if m != model]
        own = grid[(grid["receiver"] == model) & (grid["author"] == model) & (grid["arm"] == "neutral")
                   & (grid["length"] == 500) & (grid["regen"] == 0)]
        others = [F(model, grid[(grid["receiver"] == model) & (grid["author"] == p) & (grid["arm"] == "neutral")
                                & (grid["length"] == 500) & (grid["regen"] == 0)]) for p in partners]
        others = [o for o in others if o is not None]
        row = {"model": model,
               "retest": F(model, grid[(grid["receiver"] == model) & (grid["arm"] == "retest")]),
               "no_prompt": F(model, nosys[nosys["receiver"] == model]),
               "placebo": F(model, grid[(grid["receiver"] == model) & (grid["arm"] == "placebo")]),
               "own_desc": F(model, own),
               "other_desc": float(np.mean(others)) if others else None, "n_other": len(others)}
        rows.append(row)
        fmt = lambda v: f"{v:.3f}" if v is not None else "  n/a"
        print(f"{model:<16} {fmt(row['retest']):>7} {fmt(row['no_prompt']):>9} {fmt(row['placebo']):>8} "
              f"{fmt(row['own_desc']):>9} {fmt(row['other_desc']):>11} (n={row['n_other']})")
    means = {k: float(np.mean([r[k] for r in rows if r[k] is not None]))
             for k in ("retest", "no_prompt", "placebo", "own_desc", "other_desc")}
    print(f"{'MEAN':<16} " + " ".join(f"{means[k]:>{w}.3f}" for k, w in
                                     (("retest", 7), ("no_prompt", 9), ("placebo", 8), ("own_desc", 9), ("other_desc", 11))))
    (ROOT / "results" / "arxiv" / "self_ladder.json").write_text(json.dumps({"rows": rows, "means": means}, indent=1),
                                                                 encoding="utf-8")


if __name__ == "__main__":
    main()
