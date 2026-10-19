"""Independent checks of the quantitative claims in results/review_2026-10-19/review.md.
Read-only; prints one block per claim."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_grid as ag

ROOT = Path(__file__).resolve().parent.parent
MERGED = ROOT / "results" / "arxiv" / "grid_merged.jsonl"


def load(p):
    return ag.load_jsonl(p) if p.exists() else pd.DataFrame()


def main() -> None:
    grid = load(ag.GRID)
    screen = load(ag.SCREEN)
    pc = load(ROOT / "runs" / "arxiv_positive_controls.jsonl")
    gridb = load(ROOT / "runs" / "arxiv_gridb.jsonl")
    local = load(ROOT / "runs" / "arxiv_local_demos.jsonl")
    final = json.loads((ROOT / "battery" / "items_final.json").read_text(encoding="utf-8"))
    cand = json.loads((ROOT / "battery" / "items_candidates.json").read_text(encoding="utf-8"))
    final_ids = {it["id"] for it in final}

    print("=== Claim 2a: demonstrations shift answers toward the FIRST option (letter A) ===")
    def a_rate(df):
        ok = df[df["letter"].notna()]
        return (ok["letter"] == "A").mean(), len(ok)
    for name, df, sel in [
        ("Grid A retest", grid, (grid["arm"] == "retest") & grid["receiver"].isin(ag.GRID_A)),
        ("Grid A demos-30", pc, (pc["arm"] == "fewshot") & (pc["length"] == 30)),
        ("Grid A demos-110", pc, (pc["arm"] == "oracle")),
        ("Grid A neutral-500", grid, (grid["arm"] == "neutral") & (grid["length"] == 500) & grid["receiver"].isin(ag.GRID_A)),
        ("Grid B retest", grid, (grid["arm"] == "retest") & grid["receiver"].isin(ag.GRID_B)),
        ("Grid B demos-30", gridb, (gridb["arm"] == "fewshot")),
        ("Local retest", grid, (grid["arm"] == "retest") & grid["receiver"].isin(ag.LOCALS)),
        ("Local demos-30", local, (local["arm"] == "fewshot") & (local["length"] == 30)),
    ]:
        if len(df):
            r, n = a_rate(df[sel])
            print(f"  {name:<20} A-rate {r:.3f}  (n parsed {n})")

    print("\n=== Claim 2b: did exactly 110 items pass the gates (so all 90 held-out failed)? ===")
    rep = ROOT / "battery" / "selection_report.md"
    if rep.exists():
        txt = rep.read_text(encoding="utf-8")
        for line in txt.splitlines():
            if any(k in line.lower() for k in ("pass", "eligible", "stab", "compliance", "selected", "retained")):
                print("  " + line.strip()[:140])
    else:
        print("  no battery/selection_report.md; files:", [p.name for p in (ROOT / "battery").iterdir()])

    print("\n=== Claim 2b (ties): deepseek-v3 held-out items with tied majority ===")
    held = sorted({it["id"] for it in cand} - final_ids)
    for short in ag.GRID_A:
        s = screen[(screen["short"] == short) & screen["item_id"].isin(held) & screen["choice"].notna()]
        ties = sum(1 for _, g in s.groupby("item_id") if (g["choice"] == "option_a").sum() * 2 == len(g))
        print(f"  {short:<14} tied items: {ties}/{len(held)}  (samples per item: {s.groupby('item_id').size().mean():.1f})")

    print("\n=== Claim 3: parsed answers and items retained in the cells the review names ===")
    def cov(df, cell):
        g = df[df["cell_id"] == cell]
        ok = g[g["choice"].notna()]
        return len(ok), len(g), ok["item_id"].nunique()
    for label, df, cell in [
        ("haiku own neutral-500", grid, "haiku-4.5|haiku-4.5|neutral|500|r0"),
        ("glm own neutral-2000", grid, "glm-4.5-air|glm-4.5-air|neutral|2000|r0"),
        ("gemma own framed-500", grid, "gemma4-31b|gemma4-31b|framed|500|r0"),
        ("gpt5.2 <- haiku demos-30", gridb, "gpt5.2|haiku-4.5|fewshot|30|r0"),
    ]:
        p, n, items = cov(df, cell)
        print(f"  {label:<26} parsed {p}/{n}  items with >=1 parsed answer {items}/110")
    print("  overall parsed rate, sprint grid:", round(grid["choice"].notna().mean(), 3))
    low = []
    for cell, g in grid[~grid["arm"].isin(["baseline", "retest", "placebo"])].groupby("cell_id"):
        r = g["choice"].notna().mean()
        if r < 0.8:
            low.append((cell, round(r, 3)))
    print(f"  sprint cells with parsed rate < 0.80: {len(low)}")
    for c, r in sorted(low, key=lambda x: x[1])[:12]:
        print(f"    {c:<48} {r}")

    print("\n=== Claim 4: specificity gain with paired item bootstrap (neutral-500, 18 comparisons) ===")
    gm = load(MERGED) if MERGED.exists() else grid
    final_items = set(gm["item_id"].unique())
    ref = ag.p_hat_table(screen[screen["item_id"].isin(final_items)], "short")
    for m, t in ag.p_hat_table(gm[gm["arm"] == "baseline"], "receiver").items():
        ref[m] = t
    n500 = gm[(gm["arm"] == "neutral") & (gm["length"] == 500) & (gm["regen"] == 0)]
    def tbl(df):
        ok = df[df["choice"].notna()]
        return {i: (gg["choice"] == "option_a").mean() for i, gg in ok.groupby("item_id")}
    diffs = []
    for trio in (ag.GRID_A, ag.GRID_B, ag.LOCALS):
        for a in trio:
            for b in trio:
                if a == b:
                    continue
                own = ag.fidelity(ref[a], tbl(n500[(n500["author"] == a) & (n500["receiver"] == b)]))
                for c in trio:
                    if c in (a, b):
                        continue
                    oth = ag.fidelity(ref[a], tbl(n500[(n500["author"] == c) & (n500["receiver"] == b)]))
                    if own and oth:
                        common = sorted(set(own["items"]) & set(oth["items"]))
                        da, db = dict(zip(own["items"], own["per_item"])), dict(zip(oth["items"], oth["per_item"]))
                        diffs.append(np.array([da[i] - db[i] for i in common]))
    rng = np.random.default_rng(14)
    means = np.array([np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in diffs]) for _ in range(5000)])
    print(f"  n comparisons {len(diffs)}, mean gain {np.mean([d.mean() for d in diffs]):+.4f}, "
          f"95% CI [{np.percentile(means, 2.5):+.4f}, {np.percentile(means, 97.5):+.4f}]")

    print("\n=== Claim 8: same-base pair transfer gain vs others at neutral-500 ===")
    B = {}
    for a in ref:
        for r in ref:
            if a != r:
                B[(a, r)] = ag.fidelity(ref[a], ref[r])
    rows = []
    for (a, r), g in n500[n500["author"] != n500["receiver"]].groupby(["author", "receiver"]):
        if (a, r) in B:
            f = ag.fidelity(ref[a], tbl(g))
            if f:
                rows.append((a, r, f["F"], B[(a, r)]["F"], f["F"] - B[(a, r)]["F"]))
    sb = [x for x in rows if {x[0], x[1]} == {"hermes3-70b", "llama3.3-70b"}]
    ot = [x for x in rows if {x[0], x[1]} != {"hermes3-70b", "llama3.3-70b"}]
    print(f"  same-base pairs: T = {[round(x[4], 3) for x in sb]} (B = {[round(x[3], 3) for x in sb]})")
    print(f"  other pairs mean T = {np.mean([x[4] for x in ot]):+.3f}, mean B = {np.mean([x[3] for x in ot]):.3f}")

    print("\n=== Claim 3 (denominator): items answered by every model at neutral-500 self cells ===")
    sets = []
    for m in sorted(ref):
        g = gm[(gm["receiver"] == m) & (gm["author"] == m) & (gm["arm"] == "neutral") & (gm["length"] == 500) & (gm["regen"] == 0)]
        if len(g):
            sets.append(set(g[g["choice"].notna()]["item_id"]))
    print(f"  models {len(sets)}, common items {len(set.intersection(*sets))}")

    print("\n=== Claim 8: model configurations actually used ===")
    from run_grid import MODEL_DEFS
    for k, (prov, mid, extra, mt) in MODEL_DEFS.items():
        print(f"  {k:<16} {mid:<36} extra={extra} max_tokens={mt}")


if __name__ == "__main__":
    main()
