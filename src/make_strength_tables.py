"""Appendix tables from results/arxiv/strength.json (analyze_strength.py):
arxiv/tab_category.tex (per-category transfer gain at neutral-500) and
arxiv/tab_movement.tex (per-pair movement decomposition at neutral-500)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "arxiv" / "strength.json"
NAME = {"mistral32-local": "mistral-small-3.2", "kimi-k2.5-nr": "kimi-k2.5", "gpt5.2": "gpt-5.2"}
CAT = {"assistant_policy": "assistant policy", "moral_circle": "moral circle", "self_referential": "self-referential",
       "ai_governance": "AI governance", "epistemic": "epistemic", "culture": "culture", "ethics": "ethics",
       "aesthetic_control": "aesthetic (control)"}


def s(v: float) -> str:
    return f"{v:+.3f}"


def main() -> None:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    # ---- per-category T
    rows = sorted(d["per_category_T"].items(), key=lambda kv: -kv[1]["n_items"])
    lines = [r"\begin{table}[H]", r"\centering\small", r"\begin{tabular}{lccc}", r"\toprule",
             r"Category & Items & $\Tscript$ & 95\% CI \\", r"\midrule"]
    for cat, r in rows:
        lines.append(f"{CAT.get(cat, cat)} & {r['n_items']} & ${s(r['mean'])}$ & $[{s(r['ci'][0])}, {s(r['ci'][1])}]$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Transfer gain by item category at neutral-500, pooled over the 20 directed cross pairs "
              r"(joint item bootstrap within category). Exploratory: eight categories are tested and the intervals "
              r"are not adjusted for multiplicity. Item counts are the items with parsed answers in every pair.}",
              r"\label{tab:category}", r"\end{table}"]
    (ROOT / "arxiv" / "tab_category.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- per-pair movement
    mv = d["movement"]
    lines = [r"\begin{table}[H]", r"\centering\footnotesize", r"\begin{tabular}{llcccccc}", r"\toprule",
             r"Author & Receiver & $|t|$ & $|m|$ desc. & $|m|$ prose & slope desc. & slope prose & cos desc. \\",
             r"\midrule"]
    for p in sorted(mv["pairs"], key=lambda p: (p["grid"], p["receiver"], p["author"])):
        lines.append(f"{NAME.get(p['author'], p['author'])} & {NAME.get(p['receiver'], p['receiver'])} & "
                     f"{p['desc_target_size']:.3f} & {p['desc_magnitude']:.3f} & {p['prose_magnitude']:.3f} & "
                     f"${s(p['desc_slope'])}$ & ${s(p['prose_slope'])}$ & ${s(p['desc_cos'])}$ \\\\")
    sm = mv["summary"]
    lines += [r"\midrule",
              f"Mean & & {sm['target_size']:.3f} & {sm['desc_magnitude']:.3f} & {sm['prose_magnitude']:.3f} & "
              f"${s(sm['desc_slope'])}$ & ${s(sm['prose_slope'])}$ & ${s(sm['desc_cos'])}$ \\\\",
              r"\bottomrule", r"\end{tabular}",
              r"\caption{Movement decomposition at neutral-500 for every directed cross pair. $t$ is the per-item "
              r"author--receiver difference in the probability of choosing option~A under generic prompts "
              r"(the gap the text would have to close); $m$ is the receiver's per-item shift under the text, "
              r"measured from an independent baseline sample (the receiver's test-retest cell). $|\cdot|$ is the "
              r"mean absolute value over items; the slope is the least-squares coefficient of $m$ on $t$ through "
              r"the origin (the fraction of the gap closed on average); cos is the cosine between $m$ and $t$. "
              r"``prose'' is the mean over the two 500-word unrelated texts read by the same receiver, scored "
              r"against the same author; the retest-to-reference noise floor for $|m|$ is "
              f"{sm['retest_noise_magnitude']:.3f}" r".}",
              r"\label{tab:movement}", r"\end{table}"]
    (ROOT / "arxiv" / "tab_movement.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote tab_category.tex, tab_movement.tex")


if __name__ == "__main__":
    main()
