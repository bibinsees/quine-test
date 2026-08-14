"""Pilot analyzer: eligibility gates, pairwise divergence, and the KILL-TEST verdict.

Reads runs/pilot.jsonl, computes the preregistered metrics from pilot/protocol.md,
and prints:
  1. per-model eligibility (compliance, order robustness, decisiveness)
  2. pairwise baseline agreement matrix B-hat (+ JSD, Pearson r)
  3. KILL-TEST VERDICT per our_idea.md (GO / MARGINAL / NO-GO at 0.75 / 0.85)
  4. recommended triple + most discriminative items for the full battery
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUN_PATH = ROOT / "runs" / "pilot.jsonl"

GATE_COMPLIANCE = 0.95
GATE_ORDER_ROBUST = 0.70
GATE_DECISIVE = 0.30
KILL_GO = 0.75
KILL_CEILING = 0.85

EXPLORATORY = {"qwen-uncensored"}  # scored, but excluded from the core triple
AXES = {
    "qwen3.6-35b": "cn-open", "qwen3.5-122b": "cn-open", "deepseek-v3": "cn-closed",
    "gemma4-31b": "us-open", "llama3.1-8b": "us-open",
    "gpt4o-mini": "us-closed", "haiku-4.5": "us-closed",
    "mistral-small": "eu", "mistral-local": "eu", "mistral32-local": "eu",
    "qwen-uncensored": "uncensored",
    "sonnet-4.6": "us-closed", "gpt5.2": "us-closed", "gpt5-mini": "us-closed",
    "kimi-k2.5-nr": "cn-closed", "llama3.3-70b": "us-open",
    "grok-4.3": "us-xai", "kimi-k2.5": "cn-closed", "glm-4.5-air": "cn-closed",
    "hermes4-70b": "us-open-minimal", "phi-4": "us-synthetic", "gemma4-12b": "us-open",
}


def load() -> pd.DataFrame:
    rows = [json.loads(l) for l in RUN_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    # keep the latest record per call_id (reruns append)
    df = df.sort_values("ts").drop_duplicates("call_id", keep="last")
    return df


def jsd(p: float, q: float) -> float:
    """Jensen-Shannon divergence between Bernoulli(p) and Bernoulli(q), base 2."""
    def h(x):
        x = np.clip(x, 1e-12, 1 - 1e-12)
        return -(x * np.log2(x) + (1 - x) * np.log2(1 - x))
    m = (p + q) / 2
    return h(m) - (h(p) + h(q)) / 2


def main() -> None:
    df = load()
    ok = df[df["ok"] == True]  # noqa: E712
    models = sorted(df["short"].unique())

    # ---------------------------------------------------- 1. eligibility gates
    print("=" * 78)
    print("1. MODEL ELIGIBILITY (gates: compliance>=0.95, order-robust>=0.70, decisive>=0.30)")
    print("=" * 78)
    stats = {}
    for m in models:
        sub = df[df["short"] == m]
        sub_ok = sub[sub["ok"] == True]  # noqa: E712
        compliance = (sub_ok["letter"].notna().sum()) / max(len(sub), 1)
        parsed = sub_ok[sub_ok["choice"].notna()]

        # p-hat per item (pooled over orders), plus per-order majorities
        p_item, orob, dec = {}, [], []
        for item_id, g in parsed.groupby("item_id"):
            p = (g["choice"] == "option_a").mean()
            p_item[item_id] = p
            dec.append(abs(p - 0.5) * 2)
            maj = {}
            for order, go in g.groupby("order"):
                if len(go):
                    maj[order] = (go["choice"] == "option_a").mean() >= 0.5
            if len(maj) == 2:
                orob.append(maj["orig"] == maj["flip"])
        order_robust = float(np.mean(orob)) if orob else np.nan
        decisive = float(np.mean(dec)) if dec else np.nan
        err_rate = 1 - len(sub_ok) / max(len(sub), 1)

        eligible = (compliance >= GATE_COMPLIANCE and order_robust >= GATE_ORDER_ROBUST
                    and decisive >= GATE_DECISIVE)
        stats[m] = {"compliance": compliance, "order_robust": order_robust,
                    "decisive": decisive, "p_item": p_item, "eligible": eligible}
        flag = "PASS" if eligible else "FAIL"
        print(f"  {m:<16} compliance={compliance:.3f}  order_robust={order_robust:.3f}  "
              f"decisive={decisive:.3f}  api_err={err_rate:.3f}  [{flag}]")

    eligible_models = [m for m in models if stats[m]["eligible"]]
    core_models = [m for m in eligible_models if m not in EXPLORATORY]

    # ------------------------------------------------- 2. pairwise divergence
    print()
    print("=" * 78)
    print("2. PAIRWISE BASELINE AGREEMENT B-hat (majority vote) | JSD | Pearson r")
    print("=" * 78)
    pair_stats = {}
    for m1, m2 in combinations(models, 2):
        common = sorted(set(stats[m1]["p_item"]) & set(stats[m2]["p_item"]))
        if len(common) < 10:
            continue
        p1 = np.array([stats[m1]["p_item"][i] for i in common])
        p2 = np.array([stats[m2]["p_item"][i] for i in common])
        agree = float(np.mean((p1 >= 0.5) == (p2 >= 0.5)))
        mjsd = float(np.mean([jsd(a, b) for a, b in zip(p1, p2)]))
        r = float(np.corrcoef(p1, p2)[0, 1]) if p1.std() > 0 and p2.std() > 0 else np.nan
        pair_stats[(m1, m2)] = {"agree": agree, "jsd": mjsd, "r": r, "n": len(common)}

    for (m1, m2), s in sorted(pair_stats.items(), key=lambda kv: kv[1]["agree"]):
        print(f"  {m1:<16} vs {m2:<16} B={s['agree']:.3f}  JSD={s['jsd']:.3f}  "
              f"r={s['r']:.3f}  (n={s['n']})")

    # --------------------------------------------------------- 3. kill verdict
    print()
    print("=" * 78)
    print("3. KILL-TEST VERDICT (our_idea.md day-1 gate)")
    print("=" * 78)
    best_triple, best_score = None, None
    for triple in combinations(core_models, 3):
        axes = {AXES.get(m, "?") for m in triple}
        if len(axes) < 2:
            continue
        pairs = [tuple(sorted(p)) for p in combinations(triple, 2)]
        if not all(p in pair_stats for p in pairs):
            continue
        max_b = max(pair_stats[p]["agree"] for p in pairs)
        mean_div = float(np.mean([1 - pair_stats[p]["agree"] for p in pairs]))
        if best_score is None or (max_b, -mean_div) < best_score:
            best_score, best_triple = (max_b, -mean_div), triple

    if best_triple is None:
        print("  NO ELIGIBLE TRIPLE - too few models passed the gates. Fix compliance first.")
    else:
        max_b = best_score[0]
        print(f"  best triple : {best_triple}")
        for p in combinations(best_triple, 2):
            p = tuple(sorted(p))
            print(f"    {p[0]} vs {p[1]}: B={pair_stats[p]['agree']:.3f}")
        print(f"  worst pairwise B in triple = {max_b:.3f}")
        if max_b <= KILL_GO:
            print(f"  >>> VERDICT: GO (all pairwise B <= {KILL_GO}). Proceed to full grid.")
        elif max_b <= KILL_CEILING:
            print(f"  >>> VERDICT: MARGINAL ({KILL_GO} < B <= {KILL_CEILING}). "
                  "Rebuild full battery from discriminative items below, then re-verify.")
        else:
            print(f"  >>> VERDICT: NO-GO (B > {KILL_CEILING}, ceiling effect). "
                  "Pivot per strategic plan section 6.")

    # -------------------------------------------- 4. discriminative items
    print()
    print("=" * 78)
    print("4. MOST DISCRIMINATIVE ITEMS (cross-model variance of p-hat, core models)")
    print("=" * 78)
    ref = core_models if core_models else models
    item_ids = sorted(set().union(*[set(stats[m]["p_item"]) for m in ref]))
    rows = []
    for i in item_ids:
        ps = [stats[m]["p_item"][i] for m in ref if i in stats[m]["p_item"]]
        if len(ps) >= 3:
            rows.append((i, float(np.var(ps)), float(np.mean(ps))))
    rows.sort(key=lambda r: -r[1])
    cat = pd.DataFrame(rows, columns=["item", "var", "mean_p"])
    cat["category"] = cat["item"].str.rsplit("-", n=1).str[0]
    for i, v, mp in rows[:15]:
        print(f"  {i:<10} var={v:.3f}  mean_p={mp:.2f}")
    print()
    print("  variance by category (control 'aes' should be LOW-signal):")
    for c, g in cat.groupby("category"):
        print(f"    {c:<6} mean_var={g['var'].mean():.3f}")


if __name__ == "__main__":
    main()
