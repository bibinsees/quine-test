"""First-option (letter A) rate per condition, Grid A: the positional-regularity check
registered as H1 in pilot/protocol_arxiv_demos_balanced.md."""

from __future__ import annotations

from pathlib import Path

import analyze_grid as ag

ROOT = Path(__file__).resolve().parent.parent


def rate(df):
    ok = df[df["letter"].notna()]
    return (ok["letter"] == "A").mean(), len(ok)


def main() -> None:
    grid = ag.load_jsonl(ag.GRID)
    pc = ag.load_jsonl(ROOT / "runs" / "arxiv_positive_controls.jsonl")
    bal_p = ROOT / "runs" / "arxiv_demos_balanced.jsonl"
    bal = ag.load_jsonl(bal_p) if bal_p.exists() else None
    ga = grid["receiver"].isin(ag.GRID_A)
    rows = [("generic prompt (retest)", grid[(grid["arm"] == "retest") & ga]),
            ("neutral-500 self-description", grid[(grid["arm"] == "neutral") & (grid["length"] == 500) & ga]),
            ("demonstrations-30 (original)", pc[(pc["arm"] == "fewshot") & (pc["length"] == 30)]),
            ("demonstrations-110 (answer key)", pc[pc["arm"] == "oracle"])]
    if bal is not None:
        rows += [("demonstrations-30, balanced", bal[bal["arm"] == "fewshot_bal"]),
                 ("demonstrations-55, balanced, stable", bal[bal["arm"] == "demos_bal"])]
    print(f"{'condition':<38} {'A-rate':>7} {'n':>6}")
    for name, df in rows:
        if len(df):
            r, n = rate(df)
            print(f"{name:<38} {r:>7.3f} {n:>6}")


if __name__ == "__main__":
    main()
