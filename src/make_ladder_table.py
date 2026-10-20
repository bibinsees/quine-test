"""Appendix table: per-model drift from the model's own generic-prompt profile
under each prefix (retest, no prompt, placebo, own description, another
model's description). Reads results/arxiv/self_ladder.json (analyze_self_ladder.py),
writes arxiv/tab_self_ladder.tex."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "arxiv" / "self_ladder.json"
OUT = ROOT / "arxiv" / "tab_self_ladder.tex"
NAME = {"mistral32-local": "mistral-small-3.2", "kimi-k2.5-nr": "kimi-k2.5", "gpt5.2": "gpt-5.2"}


def f(v):
    return f"{v:.3f}" if v is not None else "n/a"


def main() -> None:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    # unrelated prose at matched lengths, two styles (analyze_placebo_lengths.py)
    styles = [("expo", "Expository"), ("mixed", "Sensory")]
    unrel: dict = {}  # (style, model) -> {length: F}
    for key, _ in styles:
        p = ROOT / "results" / "arxiv" / ("placebo_lengths.json" if key == "mixed" else f"placebo_lengths_{key}.json")
        if p.exists():
            for row in json.loads(p.read_text(encoding="utf-8"))["table8"]["models"]:
                unrel.setdefault((key, row["model"]), {})[row["length"]] = row["placebo"]
    lens = [100, 500, 2000] if unrel else []
    cols = [(key, n) for key, _ in styles for n in lens]
    group = (r"Model & Retest & No prompt & "
             + " & ".join(f"\\multicolumn{{{len(lens)}}}{{c}}{{{label}}}" for key, label in styles)
             + r" & Own & Other's \\")
    first = 4
    cmid = " ".join(f"\\cmidrule(lr){{{first + i * len(lens)}-{first + (i + 1) * len(lens) - 1}}}" for i in range(len(styles)))
    head = (r" & & & " + " & ".join(str(n) for _ in styles for n in lens) + r" & desc. & desc. \\")
    lines = [r"\begin{table}[H]", r"\centering\scriptsize\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{l" + "c" * (4 + len(cols)) + "}", r"\toprule", group, cmid, head, r"\midrule"]
    for r in d["rows"]:
        u = "".join(f"{f(unrel.get((key, r['model']), {}).get(n))} & " for key, n in cols)
        lines.append(f"{NAME.get(r['model'], r['model'])} & {f(r['retest'])} & {f(r['no_prompt'])} & "
                     f"{u}{f(r['own_desc'])} & {f(r['other_desc'])} \\\\")
    m = d["means"]
    um = ""
    for key, n in cols:
        vals = [unrel[k][n] for k in unrel if k[0] == key and n in unrel[k]]
        um += f"{f(sum(vals) / len(vals)) if vals else 'n/a'} & "
    lines += [r"\midrule",
              f"Mean & {f(m['retest'])} & {f(m['no_prompt'])} & {um}{f(m['own_desc'])} & {f(m['other_desc'])} \\\\",
              r"\bottomrule", r"\end{tabular}",
              r"\caption{Per-model perturbation ladder: fidelity $F$ of the model's choice profile "
              r"under each prefix to its own generic-prompt profile (1 = identical). Retest: the "
              r"generic prompt with fresh seeds (the ceiling $C$). No prompt: question only. "
              r"Expository and sensory: the unrelated texts of Section~\ref{sec:poscontrol-methods} at "
              r"nominal 100, 500 and 2{,}000 words. Own desc.: the model's neutral-500 "
              r"self-description. Other's desc.: another grid member's, averaged over partners. "
              r"Unlike Table~\ref{tab:decomp}, these measure how far each prefix moves a model from "
              r"its own usual choices, not how close it comes to an author.}",
              r"\label{tab:self-ladder}", r"\end{table}"]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(d['rows'])} models)")


if __name__ == "__main__":
    main()
