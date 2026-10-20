"""arXiv: pooled decomposition of the added conditions on Grid A (exploratory)
and on the two preregistered replication grids. Reads
results/arxiv/positive_controls.json, writes arxiv/tab_replication.tex."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PC = ROOT / "results" / "arxiv" / "positive_controls.json"
OUT = ROOT / "arxiv" / "tab_replication.tex"

GRIDS = [("A", "Grid A"), ("B", "Grid B (preregistered)"),
         ("L", "Local trio (preregistered)")]
CONDS = [("oracle-110", "demonstrations-110"), ("fewshot-30", "demonstrations-30"),
         ("fewshot-90", "demonstrations-90"), ("selfobs-500", "self-observation-500"),
         ("3p", "third-person-500")]


def main() -> None:
    pc = json.loads(PC.read_text(encoding="utf-8"))
    pooled = pc["pooled"]
    lines = [r"\begin{table}[H]", r"\centering\small",
             r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{llccccccl}", r"\toprule",
             r"Grid & Condition & $\Fself$ & $\Fcross$ & $\B$ & $\Tscript$ & $\Rweight$ & "
             r"sig.\ pairs ($+$/$-$) & Cells (s/c) \\", r"\midrule"]
    for gid, gname in GRIDS:
        first = True
        for cond, cname in CONDS:
            key = cond if gid == "A" else f"{gid}:{cond}"
            p = pooled.get(key)
            if not p or p.get("F_cross") is None:
                continue
            if cond == "3p":  # third-person (observer-written): B taken from the grid's cross-pair pool
                b_key = "selfobs-500" if gid == "A" else f"{gid}:selfobs-500"
                b = pooled.get(b_key, {}).get("B")
                t_str = f"${p['F_cross'] - b:+.3f}$" if b is not None else ""
                r_str = f"${p['F_self'] - p['F_cross']:+.3f}$"
                lines.append(f"{gname if first else ''} & {cname} & {p['F_self']:.3f} & {p['F_cross']:.3f} & "
                             f"{(f'{b:.3f}' if b is not None else '')} & {t_str} & {r_str} & & \\\\")
                first = False
                continue
            t = p["T_script"]
            lines.append(
                f"{gname if first else ''} & {cname} & {p['F_self']:.3f} & {p['F_cross']:.3f} & "
                f"{p['B']:.3f} & ${t:+.3f}$ & ${(p['R_weight'] if abs(p['R_weight']) >= 0.0005 else 0.0):+.3f}$ & "
                f"{p['n_pairs_T_pos_sig']}/{p['n_pairs_T_neg_sig']} & "
                f"{p['n_self_cells']}/{p['n_cross_cells']} \\\\")
            first = False
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\caption{Positive-control and authorship conditions on Grid A and on "
              r"the two preregistered replication grids (predictions committed before running; "
              r"\texttt{pilot/protocol\_arxiv\_*\_replication.md} in the repository). sig.\ pairs "
              r"counts cross cells whose 95\% paired-bootstrap CI for $\Tscript$ lies entirely "
              r"above ($+$) or below ($-$) zero.}",
              r"\label{tab:replication}", r"\end{table}"]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    preds = pc.get("predictions", [])
    for p in preds:
        print(f"[{'PASS' if p['pass'] else 'FAIL'}] {p['grid']}: {p['prediction']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
