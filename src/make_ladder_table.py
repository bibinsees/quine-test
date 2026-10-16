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
    lines = [r"\begin{table}[H]", r"\centering\small", r"\begin{tabular}{lccccc}", r"\toprule",
             r"Model & Test-retest & No prompt & Placebo & Own description & Other's description \\",
             r"\midrule"]
    for r in d["rows"]:
        lines.append(f"{NAME.get(r['model'], r['model'])} & {f(r['retest'])} & {f(r['no_prompt'])} & "
                     f"{f(r['placebo'])} & {f(r['own_desc'])} & {f(r['other_desc'])} \\\\")
    m = d["means"]
    lines += [r"\midrule",
              f"Mean & {f(m['retest'])} & {f(m['no_prompt'])} & {f(m['placebo'])} & {f(m['own_desc'])} & {f(m['other_desc'])} \\\\",
              r"\bottomrule", r"\end{tabular}",
              r"\caption{Per-model perturbation ladder. Each entry is the fidelity $F$ of the model's "
              r"choice profile under the given prefix to its own generic-prompt reference profile "
              r"(1 = identical). Test-retest: the generic prompt again with fresh seeds (the ceiling "
              r"$C$). No prompt: question only. Placebo: the generic prompt replaced by 529 words of "
              r"unrelated prose. Own description: the model's neutral-500 self-description. Other's "
              r"description: the neutral-500 self-description of another model of the same grid, "
              r"averaged over grid partners. These fidelities answer a different question from "
              r"Table~\ref{tab:decomp}: not how close a model comes to the author whose description "
              r"it reads, but how far each prefix moves the model from its own usual choices.}",
              r"\label{tab:self-ladder}", r"\end{table}"]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(d['rows'])} models)")


if __name__ == "__main__":
    main()
