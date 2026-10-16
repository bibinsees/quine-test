"""Compare each model's no-system-prompt profile (runs/arxiv_nosys.jsonl) with its
generic-prompt reference profile, in the same fidelity metric, next to the
model's test-retest ceiling. Prints a table and writes results/arxiv/nosys.json."""

from __future__ import annotations

import json
from pathlib import Path

import analyze_grid as ag

ROOT = Path(__file__).resolve().parent.parent
NOSYS = ROOT / "runs" / "arxiv_nosys.jsonl"
MERGED = ROOT / "results" / "arxiv" / "grid_merged.jsonl"
OUT = ROOT / "results" / "arxiv" / "nosys.json"


def main() -> None:
    grid = ag.load_jsonl(MERGED if MERGED.exists() else ag.GRID)
    screen = ag.load_jsonl(ag.SCREEN)
    final_items = set(grid["item_id"].unique())
    ref = ag.p_hat_table(screen[screen["item_id"].isin(final_items)], "short")
    for model, tbl in ag.p_hat_table(grid[grid["arm"] == "baseline"], "receiver").items():
        ref[model] = tbl
    retest = {}
    for receiver, g in grid[grid["arm"] == "retest"].groupby("receiver"):
        ok = g[g["choice"].notna()]
        tbl = {i: (gg["choice"] == "option_a").mean() for i, gg in ok.groupby("item_id")}
        fid = ag.fidelity(ref[receiver], tbl)
        if fid:
            retest[receiver] = fid["F"]

    nosys = ag.load_jsonl(NOSYS)
    rows = []
    print(f"{'model':<16} {'n_calls':>7} {'compl.':>6} {'F(no-sys vs generic)':>21} {'retest C':>9} {'agree':>6}")
    for receiver, g in nosys.groupby("receiver"):
        ok = g[g["choice"].notna()]
        tbl = {i: (gg["choice"] == "option_a").mean() for i, gg in ok.groupby("item_id")}
        fid = ag.fidelity(ref[receiver], tbl)
        if fid is None:
            print(f"{receiver:<16} {len(g):>7} partial")
            continue
        lo, hi = ag.boot_ci(fid["per_item"])
        c = retest.get(receiver)
        rows.append({"model": receiver, "n_calls": int(len(g)), "compliance": len(ok) / len(g),
                     "F_nosys_vs_generic": fid["F"], "F_ci": [lo, hi], "agree": fid["agree"],
                     "retest_C": c})
        print(f"{receiver:<16} {len(g):>7} {len(ok)/len(g):>6.3f} {fid['F']:>9.3f} [{lo:.3f},{hi:.3f}] "
              f"{(f'{c:.3f}' if c else '  n/a'):>9} {fid['agree']:>6.2f}")
    if rows:
        fs = [r["F_nosys_vs_generic"] for r in rows]
        cs = [r["retest_C"] for r in rows if r["retest_C"]]
        print(f"{'MEAN':<16} {'':>7} {'':>6} {sum(fs)/len(fs):>9.3f}            "
              f"{(sum(cs)/len(cs) if cs else float('nan')):>9.3f}")
    OUT.write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
