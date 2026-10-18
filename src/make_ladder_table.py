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
    # unrelated prose at matched lengths (analyze_placebo_lengths.py), if available
    unrel: dict = {}
    pl_path = ROOT / "results" / "arxiv" / "placebo_lengths.json"
    if pl_path.exists():
        pl = json.loads(pl_path.read_text(encoding="utf-8"))
        for row in pl["table8"]["models"]:
            unrel.setdefault(row["model"], {})[row["length"]] = row["placebo"]
    lens = [100, 500, 2000] if unrel else []
    head = (r"Model & Test-retest & No prompt & Placebo & "
            + "".join(f"Unrelated {n} & " for n in lens)
            + r"Own description & Other's description \\")
    lines = [r"\begin{table}[H]", r"\centering\footnotesize", r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{l" + "c" * (5 + len(lens)) + "}", r"\toprule", head, r"\midrule"]
    for r in d["rows"]:
        u = "".join(f"{f(unrel.get(r['model'], {}).get(n))} & " for n in lens)
        lines.append(f"{NAME.get(r['model'], r['model'])} & {f(r['retest'])} & {f(r['no_prompt'])} & "
                     f"{f(r['placebo'])} & {u}{f(r['own_desc'])} & {f(r['other_desc'])} \\\\")
    m = d["means"]
    um = ""
    for n in lens:
        vals = [unrel[k][n] for k in unrel if n in unrel[k]]
        um += f"{f(sum(vals) / len(vals)) if vals else 'n/a'} & "
    lines += [r"\midrule",
              f"Mean & {f(m['retest'])} & {f(m['no_prompt'])} & {f(m['placebo'])} & {um}{f(m['own_desc'])} & {f(m['other_desc'])} \\\\",
              r"\bottomrule", r"\end{tabular}}",
              r"\caption{Per-model perturbation ladder. Each entry is the fidelity $F$ of the model's "
              r"choice profile under the given prefix to its own generic-prompt reference profile "
              r"(1 = identical). Test-retest: the generic prompt again with fresh seeds (the ceiling "
              r"$C$). No prompt: question only. Placebo: the generic prompt replaced by 529 words of "
              r"prose about ocean currents. Unrelated 100/500/2000: the everyday-scene texts of "
              r"102, 537 and 1{,}680 words. Own description: the model's neutral-500 self-description. "
              r"Other's description: the neutral-500 self-description of another model of the same "
              r"grid, averaged over grid partners. These fidelities answer a different question from "
              r"Table~\ref{tab:decomp}: not how close a model comes to the author whose description "
              r"it reads, but how far each prefix moves the model from its own usual choices.}",
              r"\label{tab:self-ladder}", r"\end{table}"]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(d['rows'])} models)")


if __name__ == "__main__":
    main()
