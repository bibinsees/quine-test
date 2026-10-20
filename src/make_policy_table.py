"""Appendix table for the known-semantic-persona experiment. Reads
results/arxiv/policy_persona.json (analyze_policy.py), writes arxiv/tab_policy.tex."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "arxiv" / "policy_persona.json"
OUT = ROOT / "arxiv" / "tab_policy.tex"


def ci(d):
    return f"${d['delta']:+.3f}$ $[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]$"


def main() -> None:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    readopt = {(t["source"], t["policy"]): t for t in d["targets"]}
    lines = [r"\begin{table}[H]", r"\centering\footnotesize", r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{lllccccc}", r"\toprule",
             r"Source & Policy & Receiver & $B'$ & $F_{\mathrm{policy}}$ & $T_{\mathrm{policy}}$ & "
             r"$F_{\mathrm{desc}}$ & $T_{\mathrm{desc}}$ \\", r"\midrule"]
    for p in d["pairs"]:
        lines.append(f"{p['source']} & {p['policy']} & {p['receiver']} & {p['B_prime']:.3f} & {p['F_policy']:.3f} & "
                     f"{ci(p['T_policy'])} & {p['F_desc']:.3f} & {ci(p['T_desc'])} \\\\")
    po = d["pooled"]
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\caption{Known-semantic-persona control (Grid A, 57 fold-1 evaluation items per cell). For target "
              r"$(S, P)$, the source $S$ running policy text $P$, and receiver $R \ne S$: $B'$ is the receiver's "
              r"generic-prompt fidelity to the target, $F_{\mathrm{policy}}$ its fidelity when running the policy text, "
              r"$F_{\mathrm{desc}}$ when running the 500-word self-description $S$ wrote while operating under $P$; "
              r"$T = F - B'$ with paired item bootstrap CIs. Pooled over the 18 pairs with joint item resampling: "
              f"$T_{{\\mathrm{{policy}}}} = {po['T_policy']['mean']:+.3f}$ $[{po['T_policy']['ci'][0]:+.3f}, {po['T_policy']['ci'][1]:+.3f}]$, "
              f"$T_{{\\mathrm{{desc}}}} = {po['T_desc']['mean']:+.3f}$ $[{po['T_desc']['ci'][0]:+.3f}, {po['T_desc']['ci'][1]:+.3f}]$, "
              f"difference ${po['T_policy_minus_T_desc']['mean']:+.3f}$ $[{po['T_policy_minus_T_desc']['ci'][0]:+.3f}, {po['T_policy_minus_T_desc']['ci'][1]:+.3f}]$; "
              f"the same receivers' native neutral-500 gain on the same items is ${po['T_native_same_items']['mean']:+.3f}$ "
              f"$[{po['T_native_same_items']['ci'][0]:+.3f}, {po['T_native_same_items']['ci'][1]:+.3f}]$."
              r"}", r"\label{tab:policy}", r"\end{table}",
              r"\begin{table}[H]", r"\centering\footnotesize", r"\begin{tabular}{llccc}", r"\toprule",
              r"Source & Policy & $F$(target, generic) & $F$(target, reads $D_{S,P}$) & "
              r"$F$(generic, reads $D_{S,P}$) \\", r"\midrule"]
    for (s, name), t in readopt.items():
        lines.append(f"{s} & {name} & {t['F_target_to_generic_eval']:.3f} & {t['F_self_readopt']:.3f} & "
                     f"{t['F_self_desc_to_native']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Targets of the known-persona control (fold-1 items). ``target'' is the source's profile "
              r"under the policy, ``generic'' its generic-prompt profile, ``reads $D_{S,P}$'' the source running the "
              r"self-description it wrote under the policy. Column 3: how far the policy moved the source. Columns 4 "
              r"and 5: reading its own policy-conditioned description returns the source to the policy profile, not "
              r"to its native one (H5e).}",
              r"\label{tab:policy-targets}", r"\end{table}"]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
