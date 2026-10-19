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
    head = (r"Model & Test-retest & No prompt & "
            + "".join(f"{label} {n} & " for key, label in styles for n in lens)
            + r"Own description & Other's description \\")
    lines = [r"\begin{table}[H]", r"\centering\footnotesize", r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{l" + "c" * (4 + len(cols)) + "}", r"\toprule", head, r"\midrule"]
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
              r"\bottomrule", r"\end{tabular}}",
              r"\caption{Per-model perturbation ladder. Each entry is the fidelity $F$ of the model's "
              r"choice profile under the given prefix to its own generic-prompt reference profile "
              r"(1 = identical). Test-retest: the generic prompt again with fresh seeds (the ceiling "
              r"$C$). No prompt: question only. Expository 100/500/2000: encyclopedia-style prose "
              r"about everyday physical processes, 109, 529 (the ocean-currents placebo) and 1{,}971 "
              r"words. Sensory 100/500/2000: descriptive prose about everyday scenes, 102, 537 and "
              r"1{,}680 words. Own description: the model's neutral-500 self-description. "
              r"Other's description: the neutral-500 self-description of another model of the same "
              r"grid, averaged over grid partners. These fidelities answer a different question from "
              r"Table~\ref{tab:decomp}: not how close a model comes to the author whose description "
              r"it reads, but how far each prefix moves the model from its own usual choices.}",
              r"\label{tab:self-ladder}", r"\end{table}"]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(d['rows'])} models)")


if __name__ == "__main__":
    main()
