"""Two checks prompted by the final review.

1. Parser sensitivity. The preregistered parser takes the first standalone A/B token in a reply,
   so a reply such as "I cannot select A or B" is scored as A. Here the six neutral/framed pooled
   gains are recomputed under (a) the original rule, (b) dropping replies that contain both
   standalone letters, (c) a strict rule accepting only a bare A or B (optional final period or
   exclamation mark). Counts of affected replies are reported.
2. Self-fidelity across lengths: paired joint-item-bootstrap contrasts of F_self between the
   neutral 100/500/2000-word descriptions over the ten self cells.

Writes results/arxiv/parser_sensitivity.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

import analyze_grid as ag
from analyze_strength import joint_pooled

ROOT = Path(__file__).resolve().parent.parent
MERGED = ROOT / "results" / "arxiv" / "grid_merged.jsonl"
OUT = ROOT / "results" / "arxiv" / "parser_sensitivity.json"
CONDS = [("neutral", 100), ("neutral", 500), ("neutral", 2000), ("framed", 100), ("framed", 500), ("framed", 2000)]
THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
LETTER_RE = re.compile(r"(?<![A-Za-z])([AB])(?![A-Za-z])")
STRICT_RE = re.compile(r"^\s*\**\s*([AB])\s*\**\s*[.!]?\s*$")


def letters(text):
    return LETTER_RE.findall(THINK_RE.sub(" ", text or "").strip())


def choice_from_letter(letter, order):
    if letter is None:
        return None
    if order == "orig":
        return "option_a" if letter == "A" else "option_b"
    return "option_b" if letter == "A" else "option_a"


def rechoice(row, rule):
    if not row["ok"] or row["choice"] is None:
        return None
    text = row["response"] or ""
    if rule == "original":
        return row["choice"]
    ls = letters(text)
    if rule == "drop_both":
        return None if ("A" in ls and "B" in ls) else row["choice"]
    if rule == "strict":
        m = STRICT_RE.match(THINK_RE.sub(" ", text).strip())
        return choice_from_letter(m.group(1), row["order"]) if m else None
    raise ValueError(rule)


def tbl_from(g, rule):
    ch = g.apply(lambda r: rechoice(r, rule), axis=1)
    ok = g.assign(_c=ch)
    ok = ok[ok["_c"].notna()]
    return {i: (gg["_c"] == "option_a").mean() for i, gg in ok.groupby("item_id")}


def main() -> None:
    grid = ag.load_jsonl(MERGED)
    screen = ag.load_jsonl(ag.SCREEN)
    final_items = set(grid["item_id"].unique())
    ref = ag.p_hat_table(screen[screen["item_id"].isin(final_items)], "short")
    for m, t in ag.p_hat_table(grid[grid["arm"] == "baseline"], "receiver").items():
        ref[m] = t
    parsed = grid[grid["choice"].notna()]
    both = parsed["response"].fillna("").map(lambda s: (lambda L: "A" in L and "B" in L)(letters(s)))
    strict_ok = parsed["response"].fillna("").map(lambda s: bool(STRICT_RE.match(THINK_RE.sub(" ", s).strip())))
    out = {"n_parsed": int(len(parsed)), "n_both_letters": int(both.sum()),
           "n_not_strict": int((~strict_ok).sum()), "pooled": {}}
    print(f"parsed replies {len(parsed)}; containing both letters {int(both.sum())}; not a bare A/B {int((~strict_ok).sum())}")
    print(f"{'condition':<14}{'rule':<12}{'pairs':>6}{'mean T':>9}{'95% CI':>22}{'upper95':>9}")
    for arm, n in CONDS:
        df = grid[(grid["arm"] == arm) & (grid["length"] == n) & (grid["regen"] == 0)]
        for rule in ("original", "drop_both", "strict"):
            pairs = []
            for (r, a), g in df.groupby(["receiver", "author"]):
                if r == a or a not in ref or len(g) < 400:
                    continue
                fc, b = ag.fidelity(ref[a], tbl_from(g, rule)), ag.fidelity(ref[a], ref[r])
                if fc is None or b is None:
                    continue
                dc, db = dict(zip(fc["items"], fc["per_item"])), dict(zip(b["items"], b["per_item"]))
                pairs.append({i: dc[i] - db[i] for i in set(dc) & set(db)})
            res = joint_pooled(pairs)
            if res:
                out["pooled"][f"{arm}-{n}|{rule}"] = res
                print(f"{arm + '-' + str(n):<14}{rule:<12}{res['n_pairs']:>6}{res['mean']:>+9.3f}   "
                      f"[{res['ci'][0]:+.3f}, {res['ci'][1]:+.3f}]{res['upper95']:>+9.3f}")

    # self-fidelity across lengths (neutral arm, ten self cells)
    def self_items(n):
        cells = {}
        df = grid[(grid["arm"] == "neutral") & (grid["length"] == n) & (grid["regen"] == 0)]
        for (r, a), g in df.groupby(["receiver", "author"]):
            if r != a or len(g) < 400:
                continue
            fid = ag.fidelity(ref[a], ag.p_hat_table(g, "receiver")[r])
            if fid:
                cells[a] = dict(zip(fid["items"], fid["per_item"]))
        return cells

    s = {n: self_items(n) for n in (100, 500, 2000)}
    out["self_fidelity_contrasts"] = {}
    print("\nself-fidelity contrasts (paired over models, joint item bootstrap):")
    for n1, n0 in ((500, 100), (2000, 100), (2000, 500)):
        pairs = []
        for m in s[n1]:
            if m in s[n0]:
                pairs.append({i: s[n1][m][i] - s[n0][m][i] for i in set(s[n1][m]) & set(s[n0][m])})
        res = joint_pooled(pairs)
        out["self_fidelity_contrasts"][f"{n1}-{n0}"] = res
        print(f"  {n1} minus {n0}: {res['mean']:+.4f} [{res['ci'][0]:+.4f}, {res['ci'][1]:+.4f}]  models {res['n_pairs']} items {res['n_items']}")
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
