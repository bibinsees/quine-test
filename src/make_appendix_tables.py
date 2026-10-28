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
LABEL["selfobs_stable-500"] = "self-observation-500, stable items"
LABEL["fewshot_bal-30"] = "demos-30, balanced"
LABEL["demos_bal-55"] = "demos-55, balanced"
LABEL["neutral_user-500"] = "neutral-500, user msg."
LABEL["oracle_user-110"] = "demonstrations-110, user msg."
LABEL["3p-500"] = "observer-written-500"
SHORT = {"deepseek-v3": "deepseek", "glm-4.5-air": "glm", "grok-4.3": "grok",
         "gpt5.2": "gpt5.2", "kimi-k2.5-nr": "kimi", "haiku-4.5": "haiku",
         "llama3.3-70b": "llama3.3", "gemma4-31b": "gemma", "mistral32-local": "mistral3.2"}
OUT_REPL = ROOT / "arxiv" / "tab_added_pairs_repl.tex"


def fmt(x: float) -> str:
    return f"{x:+.3f}".replace("+0.", "+.").replace("-0.", "-.")


def table(pc: dict, grids: list[tuple[str, str]], caption: str, label: str, placement: str = "H") -> str:
    """A longtable: breaks across pages, caption on top, header repeated on each page."""
    header = r"Grid & Condition & Pair (author$\to$receiver) & $F$ & $\B$ & $\Tscript$ & 95\% CI & agree \\"
    lines = [
        r"{\scriptsize\renewcommand{\arraystretch}{0.9}",
        r"\begin{longtable}{lllrlrrr}",
        r"\caption{" + caption + r"}\label{" + label + r"} \\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        r"\multicolumn{8}{l}{\footnotesize Table~\ref{" + label + r"}, continued} \\",
        r"\toprule", header, r"\midrule", r"\endhead",
        r"\midrule", r"\multicolumn{8}{r}{\footnotesize continued on the next page} \\", r"\endfoot",
        r"\bottomrule", r"\endlastfoot",
    ]
    n = 0
    for gid, gname in grids:
        cells = [c for c in pc["cells"] if c["kind"] == "cross" and c.get("grid", "A") == gid]
        first_grid = True
        for cond in LABEL:
            rows = sorted([c for c in cells if c.get("cond") == cond],
                          key=lambda c: -c["T"]["delta"])
            if not rows:
                continue
            for i, c in enumerate(rows):
                t = c["T"]
                sig = "$^{*}$" if (t["ci"][0] > 0 or t["ci"][1] < 0) else ""
                lines.append(
                    f"{gname if first_grid else ''} & {LABEL[cond] if i == 0 else ''} & "
                    f"{SHORT[c['author']]}$\\to${SHORT[c['receiver']]}{sig} & "
                    f"{c['F']:.3f} & {c['B']:.3f} & ${fmt(t['delta'])}$ & "
                    f"$[{fmt(t['ci'][0])},{fmt(t['ci'][1])}]$ & {c['agree']:.2f} \\\\")
                first_grid = False
                n += 1
            key = cond if gid == "A" else f"{gid}:{cond}"
            pooled = pc["pooled"].get(key, {})
            if pooled.get("T_script") is not None:
                lines.append(f" & & \\emph{{pooled}} & {pooled['F_cross']:.3f} & {pooled['B']:.3f} & "
                             f"${fmt(pooled['T_script'])}$ & & \\\\")
            lines.append(r"\addlinespace")
    if lines[-1] == r"\addlinespace":
        lines.pop()
    lines += [r"\end{longtable}}"]
    print(f"{label}: {n} cross cells")
    return "\n".join(lines) + "\n"


def main() -> None:
    pc = json.loads(PC.read_text(encoding="utf-8"))
    OUT.write_text(table(
        pc, [("A", "A")],
        r"Per-pair transfer contrasts for the positive-control, authorship and placement "
        r"conditions, Grid A (cross cells). $\Tscript = F - \B$ with item-level paired bootstrap (10k resamples); $^{*}$ "
        r"marks 95\% CIs excluding zero; agree is majority-vote agreement with the author. $\Tscript$ and its "
        r"interval are computed on the items the cell covers; $\B$ is the pair's full-battery baseline, so "
        r"$F - \B$ can differ from $\Tscript$ by up to $0.003$ in cells with reduced coverage.",
        "tab:added-pairs", placement="p"), encoding="utf-8")
    OUT_REPL.write_text(table(
        pc, [("B", "B"), ("L", "local")],
        r"Per-pair transfer contrasts for the preregistered replications on Grid B and the "
        r"local trio (cross cells); same conventions as Table~\ref{tab:added-pairs}.",
        "tab:added-pairs-repl"), encoding="utf-8")
    print(f"wrote {OUT} and {OUT_REPL}")


if __name__ == "__main__":
    main()
