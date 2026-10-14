"""arXiv appendix: per-pair contrasts for the conditions added after review.
Reads results/arxiv/positive_controls.json (analyze_arxiv.py) and writes
arxiv/tab_added_pairs.tex, which main.tex \\input{}s. Numbers never typed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PC = ROOT / "results" / "arxiv" / "positive_controls.json"
OUT = ROOT / "arxiv" / "tab_added_pairs.tex"

LABEL = {
    "selfobs-500": "self-observation-500",
    "optimized-500": "optimized-500",
    "fewshot-30": "demonstrations-30",
    "fewshot-60": "demonstrations-60",
    "fewshot-90": "demonstrations-90",
    "oracle-110": "demonstrations-110",
}
SHORT = {"deepseek-v3": "deepseek", "glm-4.5-air": "glm", "grok-4.3": "grok"}


def fmt(x: float) -> str:
    return f"{x:+.3f}".replace("+0.", "+.").replace("-0.", "-.")


def main() -> None:
    pc = json.loads(PC.read_text(encoding="utf-8"))
    cells = [c for c in pc["cells"] if c["kind"] == "cross"]
    lines = [
        r"\begin{table}[H]",
        r"\centering\scriptsize",
        r"\begin{tabular}{llrlrrr}",
        r"\toprule",
        r"Condition & Pair (author$\to$receiver) & $F$ & $\B$ & $\Tscript$ & 95\% CI & agree \\",
        r"\midrule",
    ]
    for cond in LABEL:
        rows = sorted([c for c in cells if c.get("cond") == cond],
                      key=lambda c: -c["T"]["delta"])
        if not rows:
            continue
        for i, c in enumerate(rows):
            t = c["T"]
            sig = "$^{*}$" if (t["ci"][0] > 0 or t["ci"][1] < 0) else ""
            name = LABEL[cond] if i == 0 else ""
            lines.append(
                f"{name} & {SHORT[c['author']]}$\\to${SHORT[c['receiver']]}{sig} & "
                f"{c['F']:.3f} & {c['B']:.3f} & ${fmt(t['delta'])}$ & "
                f"$[{fmt(t['ci'][0])},{fmt(t['ci'][1])}]$ & {c['agree']:.2f} \\\\")
        pooled = pc["pooled"].get(cond, {})
        if pooled.get("T_script") is not None:
            lines.append(f" & \\emph{{pooled}} & {pooled['F_cross']:.3f} & {pooled['B']:.3f} & "
                         f"${fmt(pooled['T_script'])}$ & & \\\\")
        lines.append(r"\addlinespace")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Per-pair transfer contrasts for the Grid-A conditions added after "
        r"review (cross cells). $\Tscript = F - \B$ with item-level paired bootstrap "
        r"(10k resamples); $^{*}$ marks 95\% CIs excluding zero; agree is majority-vote "
        r"agreement with the author across items.}",
        r"\label{tab:added-pairs}",
        r"\end{table}",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({sum(1 for c in cells)} cross cells)")


if __name__ == "__main__":
    main()
