"""Generate the neutral-500 appendix table from the current paired bootstraps."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "arxiv/tab_native_pairs.tex"
SHORT = {"deepseek-v3": "deepseek", "glm-4.5-air": "glm", "grok-4.3": "grok",
         "gpt5.2": "gpt5.2", "kimi-k2.5-nr": "kimi", "haiku-4.5": "haiku",
         "llama3.3-70b": "llama3.3", "gemma4-31b": "gemma", "mistral32-local": "mistral3.2",
         "hermes3-70b": "hermes3"}


def main():
    data = json.loads((ROOT / "results/arxiv/grid_summary.json").read_text(encoding="utf-8"))
    pairs = sorted(data["T_script_pairs_n500"].items(), key=lambda kv: -kv[1]["delta"])

    def cells(pair):
        name, v = pair
        a, r = name.split("->")
        lo, hi = v["ci"]
        star = r"$^{*}$" if lo > 0 or hi < 0 else ""
        return (f"{SHORT[a]}$\\to${SHORT[r]}{star} & ${v['delta']:+.3f}$ & "
                f"$[{lo:+.3f},{hi:+.3f}]$")

    lines = [r"\begin{table}[H]", r"\centering\scriptsize",
             r"\begin{tabular}{lrl@{\qquad}lrl}", r"\toprule",
             r"Pair (author$\to$receiver) & $\Tscript$ & 95\% CI & Pair & $\Tscript$ & 95\% CI \\", r"\midrule"]
    half = (len(pairs) + 1) // 2
    for i in range(half):
        right = cells(pairs[i + half]) if i + half < len(pairs) else " & & "
        lines.append(cells(pairs[i]) + " & " + right + r" \\")
    negative = sum(v["ci"][1] < 0 for _, v in pairs)
    positive = sum(v["ci"][0] > 0 for _, v in pairs)
    average = sum(v["delta"] for _, v in pairs) / len(pairs)
    lines += [r"\bottomrule", r"\end{tabular}",
              f"\\caption{{Neutral-500 self-descriptions: mean $\\Tscript = {average:+.3f}$ over {len(pairs)} directed pairs; "
              f"{negative} significantly negative, {positive} significantly positive "
              r"under unadjusted 95\% paired-bootstrap intervals. $^{*}$ marks intervals excluding zero.}",
              r"\label{tab:native-pairs}", r"\end{table}"]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
