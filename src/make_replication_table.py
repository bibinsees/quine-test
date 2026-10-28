"""arXiv: pooled decomposition of the added conditions on Grid A (exploratory)
and on the two preregistered replication grids. Reads
results/arxiv/positive_controls.json, writes arxiv/tab_replication.tex."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PC = ROOT / "results" / "arxiv" / "positive_controls.json"
TP = ROOT / "results" / "arxiv" / "third_person_pairs.json"  # analyze_3p_pairs.py
OUT = ROOT / "arxiv" / "tab_replication.tex"

GRIDS = [("A", "Grid A"), ("B", "Grid B (preregistered)"),
         ("L", "Local trio (preregistered)")]
CONDS = [("oracle-110", "demonstrations-110"), ("fewshot-30", "demonstrations-30"),
         ("fewshot-90", "demonstrations-90"), ("selfobs-500", "self-observation-500"),
         ("3p", "third-person-500")]


def main() -> None:
    pc = json.loads(PC.read_text(encoding="utf-8"))
    pooled = pc["pooled"]
    tp = json.loads(TP.read_text(encoding="utf-8")) if TP.exists() else {}
    lines = [r"\begin{table}[H]", r"\centering\footnotesize\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lccccccc}", r"\toprule",
             r"Condition & $\Fself$ & $\Fcross$ & $\B$ & $\Tscript$ & $\Rweight$ & "
             r"sig.\ pairs ($+$/$-$) & Cells (s/c) \\", r"\midrule"]
    for gid, gname in GRIDS:
        lines.append(r"\multicolumn{8}{l}{\emph{" + gname + r"}} \\*")
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
                s = tp.get(gid, {})
                sig = f"{s['n_pairs_T_pos_sig']}/{s['n_pairs_T_neg_sig']}" if s else ""
                cells_str = f"{s['n_self_cells']}/{s['n_cross_cells']}" if s else ""
                lines.append(f"{cname} & {p['F_self']:.3f} & {p['F_cross']:.3f} & "
                             f"{(f'{b:.3f}' if b is not None else '')} & {t_str} & {r_str} & {sig} & {cells_str} \\\\")
                continue
            t = p["T_script"]
            lines.append(
                f"{cname} & {p['F_self']:.3f} & {p['F_cross']:.3f} & "
                f"{p['B']:.3f} & ${t:+.3f}$ & ${(p['R_weight'] if abs(p['R_weight']) >= 0.0005 else 0.0):+.3f}$ & "
                f"{p['n_pairs_T_pos_sig']}/{p['n_pairs_T_neg_sig']} & "
                f"{p['n_self_cells']}/{p['n_cross_cells']} \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Positive-control and authorship conditions on Grid A and on "
              r"the two preregistered replication grids (predictions committed before running; "
              r"\texttt{pilot/protocol\_arxiv\_*\_replication.md} in the repository). sig.\ pairs "
              r"counts cross cells whose 95\% paired-bootstrap CI for $\Tscript$ lies entirely "
              r"above ($+$) or below ($-$) zero (unadjusted for multiple comparisons). "
              r"Cells (s/c) counts self/cross cells. Entries and contrasts are rounded independently.}",
              r"\label{tab:replication}", r"\end{table}"]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    preds = pc.get("predictions", [])
    for p in preds:
        print(f"[{'PASS' if p['pass'] else 'FAIL'}] {p['grid']}: {p['prediction']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
