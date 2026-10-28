"""arXiv appendix: per-pair contrasts for the positive-control, authorship and
placement conditions on every grid, one table grouped by grid.
Reads results/arxiv/positive_controls.json (analyze_arxiv.py) and writes
arxiv/tab_control_pairs.tex, which main.tex \\input{}s. Numbers never typed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PC = ROOT / "results" / "arxiv" / "positive_controls.json"
OUT = ROOT / "arxiv" / "tab_control_pairs.tex"

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
LABEL["3p-500"] = "third-person-500"
SHORT = {"deepseek-v3": "deepseek", "glm-4.5-air": "glm", "grok-4.3": "grok",
         "gpt5.2": "gpt5.2", "kimi-k2.5-nr": "kimi", "haiku-4.5": "haiku",
         "llama3.3-70b": "llama3.3", "gemma4-31b": "gemma", "mistral32-local": "mistral3.2"}
GRIDS = [("A", "Grid A"), ("B", "Grid B (preregistered)"), ("L", "Local trio (preregistered)")]
NCOL = 7


def fmt(x: float) -> str:
    return f"{x:+.3f}".replace("+0.", "+.").replace("-0.", "-.")


def table(pc: dict, grids: list[tuple[str, str]], caption: str, label: str) -> str:
    """A longtable: breaks across pages, caption on top, header repeated on each page,
    one group heading per grid (as in the replication table of the main text)."""
    header = r"Condition & Pair (author$\to$receiver) & $F$ & $\B$ & $\Tscript$ & 95\% CI & agree \\"
    lines = [
        r"{\scriptsize\renewcommand{\arraystretch}{0.9}",
        r"\begin{longtable}{llrlrrr}",
        r"\caption{" + caption + r"}\label{" + label + r"} \\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        r"\multicolumn{%d}{l}{\footnotesize Table~\ref{" % NCOL + label + r"}, continued} \\",
        r"\toprule", header, r"\midrule", r"\endhead",
        r"\midrule", r"\multicolumn{%d}{r}{\footnotesize continued on the next page} \\" % NCOL, r"\endfoot",
        r"\bottomrule", r"\endlastfoot",
    ]
    n = 0
    for gid, gname in grids:
        cells = [c for c in pc["cells"] if c["kind"] == "cross" and c.get("grid", "A") == gid]
        if not cells:
            continue
        lines.append(r"\multicolumn{%d}{l}{\emph{%s}} \\*" % (NCOL, gname))
        for cond in LABEL:
            rows = sorted([c for c in cells if c.get("cond") == cond],
                          key=lambda c: -c["T"]["delta"])
            if not rows:
                continue
            key = cond if gid == "A" else f"{gid}:{cond}"
            pooled = pc["pooled"].get(key, {})
            has_pooled = pooled.get("T_script") is not None
            for i, c in enumerate(rows):
                t = c["T"]
                sig = "$^{*}$" if (t["ci"][0] > 0 or t["ci"][1] < 0) else ""
                lines.append(
                    f"{LABEL[cond] if i == 0 else ''} & "
                    f"{SHORT[c['author']]}$\\to${SHORT[c['receiver']]}{sig} & "
                    f"{c['F']:.3f} & {c['B']:.3f} & ${fmt(t['delta'])}$ & "
                    f"$[{fmt(t['ci'][0])},{fmt(t['ci'][1])}]$ & {c['agree']:.2f} "
                    + (r"\\*" if i < len(rows) - 1 or has_pooled else r"\\"))
                n += 1
            if has_pooled:
                lines.append(f" & \\emph{{pooled}} & {pooled['F_cross']:.3f} & {pooled['B']:.3f} & "
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
        pc, GRIDS,
        r"Per-pair transfer contrasts for the positive-control, authorship and placement "
        r"conditions on all three grids (cross cells). $\Tscript = F - \B$ with item-level paired bootstrap "
        r"(10k resamples); $^{*}$ marks 95\% CIs excluding zero; agree is majority-vote agreement with the "
        r"author. $\Tscript$ and its interval are computed on the items the cell covers; $\B$ is the pair's "
        r"full-battery baseline, so $F - \B$ can differ from $\Tscript$ by up to $0.003$ in cells with "
        r"reduced coverage.",
        "tab:control-pairs"), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
