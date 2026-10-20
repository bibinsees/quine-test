"""Numbers for the reporting revisions requested by the 2026-10-19 review.

1. Coverage: parsed-answer rate and items covered per cell; list of cells below 0.90.
2. Prediction ledger: every registered criterion (pilot/protocol_arxiv_*_replication.md)
   evaluated explicitly, pass/fail.
3. Framed vs neutral at 500 words restricted to CROSS cells (transfer question).
4. Length differences with paired item bootstrap CIs (cross cells, neutral arm):
   F(2000) - F(100), F(500) - F(100) per pair, pooled.
5. Baseline-adjusted self/cross comparison: (F_self - C) vs (F_cross - B) per condition.
Writes results/arxiv/reporting_fixes.json and arxiv/tab_coverage.tex, arxiv/tab_ledger.tex."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_grid as ag

ROOT = Path(__file__).resolve().parent.parent
MERGED = ROOT / "results" / "arxiv" / "grid_merged.jsonl"
PC = ROOT / "results" / "arxiv" / "positive_controls.json"
OUT = ROOT / "results" / "arxiv" / "reporting_fixes.json"
RUN_FILES = ["grid.jsonl", "arxiv_positive_controls.jsonl", "arxiv_optimized.jsonl", "arxiv_demos_dose.jsonl",
             "arxiv_gridb.jsonl", "arxiv_local_demos.jsonl", "arxiv_usermsg.jsonl", "arxiv_nosys.jsonl",
             "arxiv_placebo_lengths.jsonl", "arxiv_placebo_expo.jsonl", "arxiv_demos_balanced.jsonl",
             "arxiv_obs_stable.jsonl", "arxiv_policy.jsonl"]
NAME = {"mistral32-local": "mistral-small-3.2", "kimi-k2.5-nr": "kimi-k2.5", "gpt5.2": "gpt-5.2"}


def tbl_of(g):
    ok = g[g["choice"].notna()]
    return {i: (gg["choice"] == "option_a").mean() for i, gg in ok.groupby("item_id")}


def seed(key):
    ag.RNG = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))


def main() -> None:
    out = {}
    # ---------- 1. coverage ----------
    frames = []
    for f in RUN_FILES:
        p = ROOT / "runs" / f
        if p.exists():
            df = ag.load_jsonl(p)
            if "phase" in df.columns:
                df = df[df["phase"] != "select"]
            frames.append(df)
    allruns = pd.concat(frames, ignore_index=True)
    battery = allruns[allruns["item_id"].isin(set(ag.load_jsonl(ag.GRID)["item_id"]))]
    cov = []
    for cell, g in battery.groupby("cell_id"):
        if len(g) < 200 or "heldout" in cell or "optselect" in cell:  # 228-call half-battery cells included
            continue
        ok = g[g["choice"].notna()]
        cov.append({"cell": cell, "n_calls": int(len(g)), "parsed": int(len(ok)),
                    "rate": len(ok) / len(g), "items": int(ok["item_id"].nunique()),
                    "empty": int(((g["ok"] == True) & (g["response"].fillna("") == "")).sum())})
    cov_df = pd.DataFrame(cov)
    print(f"battery cells: {len(cov_df)}; overall parsed rate {cov_df['parsed'].sum() / cov_df['n_calls'].sum():.4f}; "
          f"cells with rate >= 0.95: {(cov_df['rate'] >= 0.95).sum()}, 0.90-0.95: "
          f"{((cov_df['rate'] >= 0.90) & (cov_df['rate'] < 0.95)).sum()}, < 0.90: {(cov_df['rate'] < 0.90).sum()}")
    low = cov_df[cov_df["rate"] < 0.90].sort_values("rate")
    for _, r in low.iterrows():
        print(f"  {r['cell']:<52} parsed {r['parsed']:>3}/{r['n_calls']} items {r['items']:>3}  empty {r['empty']}")
    out["coverage"] = {"n_cells": int(len(cov_df)), "overall_rate": float(cov_df["parsed"].sum() / cov_df["n_calls"].sum()),
                       "n_ge_095": int((cov_df["rate"] >= 0.95).sum()), "n_lt_090": int((cov_df["rate"] < 0.90).sum()),
                       "low": low.to_dict("records")}
    lines = [r"\begin{table}[H]", r"\centering\footnotesize", r"\begin{tabular}{lrrr}", r"\toprule",
             r"Cell (receiver $\mid$ author $\mid$ condition) & Parsed / calls & Items covered & Empty replies \\", r"\midrule"]
    for _, r in low.iterrows():
        parts = r["cell"].split("|")
        label = f"{NAME.get(parts[0], parts[0])} $\\mid$ {NAME.get(parts[1], parts[1])} $\\mid$ " + "-".join(parts[2:4]).replace("_", "\\_")
        lines.append(f"{label} & {r['parsed']} / {r['n_calls']} & {r['items']} & {r['empty']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Battery cells with fewer than 90\% parsed answers (all other cells are at or above "
              r"90\%; overall parsed rate " + f"{out['coverage']['overall_rate']:.1%}".replace("%", r"\%") +
              r"). A call is unparsed when the reply contains no standalone A or B: either an empty reply "
              r"(reasoning consumed the output budget) or a refusal to state a preference. Fidelity is "
              r"computed over the items with at least one parsed answer; cells covering fewer than 60 items "
              r"are excluded.}",
              r"\label{tab:coverage}", r"\end{table}"]
    (ROOT / "arxiv" / "tab_coverage.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------- 2. prediction ledger ----------
    pc = json.loads(PC.read_text(encoding="utf-8"))
    P = pc["pooled"]
    cells = pc["cells"]

    def n_sig_pos(grid, cond):
        return sum(1 for c in cells if c.get("grid") == grid and c["cond"] == cond and c["kind"] == "cross"
                   and c["T"]["ci"][0] > 0)

    def fself(grid, cond):
        return P.get(f"{grid}:{cond}", {}).get("F_self")

    neutral_self = {}
    gm = ag.load_jsonl(MERGED)
    screen = ag.load_jsonl(ag.SCREEN)
    final_items = set(gm["item_id"].unique())
    ref = ag.p_hat_table(screen[screen["item_id"].isin(final_items)], "short")
    for m, t in ag.p_hat_table(gm[gm["arm"] == "baseline"], "receiver").items():
        ref[m] = t
    for grid_id, models in (("B", ag.GRID_B), ("L", ag.LOCALS)):
        vals = []
        for m in models:
            g = gm[(gm["receiver"] == m) & (gm["author"] == m) & (gm["arm"] == "neutral") & (gm["length"] == 500) & (gm["regen"] == 0)]
            fid = ag.fidelity(ref[m], tbl_of(g))
            if fid:
                vals.append(fid["F"])
        neutral_self[grid_id] = float(np.mean(vals))

    ledger = []
    for grid_id, gname in (("B", "Grid B"), ("L", "local trio")):
        p110, p30 = P[f"{grid_id}:oracle-110"], P[f"{grid_id}:fewshot-30"]
        ledger += [
            (gname, "P1a demonstrations-110: pooled T > 0", p110["T_script"] > 0, f"T={p110['T_script']:+.3f}"),
            (gname, "P1b demonstrations-110: all 6 cross pairs significantly positive", n_sig_pos(grid_id, "oracle-110") == 6,
             f"{n_sig_pos(grid_id, 'oracle-110')}/6"),
            (gname, "P1c demonstrations-110: pooled F_cross >= 0.93", p110["F_cross"] >= 0.93, f"F_cross={p110['F_cross']:.3f}"),
            (gname, "P1d demonstrations-110: |R| <= 0.02", abs(p110["R_weight"]) <= 0.02, f"R={p110['R_weight']:+.3f}"),
            (gname, "P2a demonstrations-30: pooled T <= 0", p30["T_script"] <= 0, f"T={p30['T_script']:+.3f}"),
            (gname, "P2b demonstrations-30: no cross pair significantly positive", n_sig_pos(grid_id, "fewshot-30") == 0,
             f"{n_sig_pos(grid_id, 'fewshot-30')} positive"),
        ]
        if grid_id == "L":
            p90 = P["L:fewshot-90"]
            ledger += [
                (gname, "P3a demonstrations-90: pooled T <= 0", p90["T_script"] <= 0, f"T={p90['T_script']:+.3f}"),
                (gname, "P3b demonstrations-90: no cross pair significantly positive", n_sig_pos("L", "fewshot-90") == 0,
                 f"{n_sig_pos('L', 'fewshot-90')} positive"),
                (gname, "P3c no monotone increase in F_cross from k=30 to k=90",
                 not (P["L:fewshot-30"]["F_cross"] < P["L:fewshot-60"]["F_cross"] < p90["F_cross"]) if "L:fewshot-60" in P
                 else p90["F_cross"] <= p30["F_cross"] + 0.0,
                 f"F_cross 30={p30['F_cross']:.3f}, 90={p90['F_cross']:.3f}"),
                (gname, "P4a F_self(demos-110) > F_self(demos-30)", fself("L", "oracle-110") > fself("L", "fewshot-30"),
                 f"{fself('L', 'oracle-110'):.3f} vs {fself('L', 'fewshot-30'):.3f}"),
                (gname, "P4b F_self(demos-30) < F_self(neutral-500 self-description)", fself("L", "fewshot-30") < neutral_self["L"],
                 f"{fself('L', 'fewshot-30'):.3f} vs {neutral_self['L']:.3f}"),
            ]
        else:
            pso, p3 = P["B:selfobs-500"], P["B:3p"]
            dcross, dself = pso["F_cross"] - p3["F_cross"], pso["F_self"] - p3["F_self"]
            signs = {np.sign(t["d"]) for t in pc["authorship"]["B"]}
            ledger += [
                (gname, "P3 self-observation-500: pooled T < 0", pso["T_script"] < 0, f"T={pso['T_script']:+.3f}"),
                (gname, "P4a authorship: |dF_cross| < 0.03", abs(dcross) < 0.03, f"{dcross:+.3f}"),
                (gname, "P4b authorship: |dF_self| < 0.03", abs(dself) < 0.03, f"{dself:+.3f}"),
                (gname, "P4c authorship: per-target differences not all of one sign", len(signs) > 1,
                 "both signs present" if len(signs) > 1 else "one sign only"),
                (gname, "P5 F_self(demos-30) < F_self(neutral-500 self-description)", fself("B", "fewshot-30") < neutral_self["B"],
                 f"{fself('B', 'fewshot-30'):.3f} vs {neutral_self['B']:.3f}"),
            ]
    # balanced-demonstration protocol (pilot/protocol_arxiv_demos_balanced.md), if run
    bal_p = ROOT / "runs" / "arxiv_demos_balanced.jsonl"
    if bal_p.exists() and "fewshot_bal-30" in P and "demos_bal-55" in P:
        bal = ag.load_jsonl(bal_p)
        retest_a = grid_retest = ag.load_jsonl(ag.GRID)
        ga = grid_retest[(grid_retest["arm"] == "retest") & grid_retest["receiver"].isin(ag.GRID_A)]
        a_ref = (ga[ga["letter"].notna()]["letter"] == "A").mean()
        def a_rate(arm):
            ok = bal[(bal["arm"] == arm) & bal["letter"].notna()]
            return (ok["letter"] == "A").mean()
        a30, a55 = a_rate("fewshot_bal"), a_rate("demos_bal")
        p30b, p55 = P["fewshot_bal-30"], P["demos_bal-55"]
        ledger += [
            ("A, balanced demos", "H1 first-option rate within 3 points of generic prompt (both conditions)",
             abs(a30 - a_ref) <= 0.03 and abs(a55 - a_ref) <= 0.03,
             f"generic {a_ref:.3f}; bal.-30 {a30:.3f}; bal.-55 {a55:.3f}"),
            ("A, balanced demos", "H2 balanced-30 pooled T closer to zero than original -0.061",
             abs(p30b["T_script"]) < 0.061, f"T={p30b['T_script']:+.3f}"),
            ("A, balanced demos", "H3 (open) balanced-55 generalisation",
             None, f"T={p55['T_script']:+.3f}; {n_sig_pos('A', 'demos_bal-55')} pair sig. +, "
             f"{sum(1 for c in cells if c.get('grid') == 'A' and c['cond'] == 'demos_bal-55' and c['kind'] == 'cross' and c['T']['ci'][1] < 0)} sig. -"),
        ]
    # stable-item observation-derived descriptions (pilot/protocol_arxiv_obs_stable.md), if run
    if "3p_stable-500" in P and "selfobs_stable-500" in P:
        p3s, pss = P["3p_stable-500"], P["selfobs_stable-500"]
        ledger += [
            ("A, stable items", "H4a observer-written from stable items: pooled F_self > 0.771",
             p3s["F_self"] > 0.771, f"F_self={p3s['F_self']:.3f}"),
            ("A, stable items", "H4a self-observation from stable items: pooled F_self > 0.781",
             pss["F_self"] > 0.781, f"F_self={pss['F_self']:.3f}"),
            ("A, stable items", "H4b authorship: |F_self(self-obs) - F_self(observer)| < 0.03",
             abs(pss["F_self"] - p3s["F_self"]) < 0.03, f"{pss['F_self'] - p3s['F_self']:+.3f}"),
            ("A, stable items", "H4c (open) self-observation cross T",
             None, f"T={pss['T_script']:+.3f}; {pss['n_pairs_T_pos_sig']} sig. +, {pss['n_pairs_T_neg_sig']} sig. -"),
        ]
    # known semantic persona (pilot/protocol_arxiv_policy_persona.md), if run
    pp_path = ROOT / "results" / "arxiv" / "policy_persona.json"
    if pp_path.exists():
        import json as _json
        pp = _json.loads(pp_path.read_text(encoding="utf-8"))
        po, L = pp["pooled"], pp["ledger"]
        def ci_s(v):
            return f"{v['mean']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]"
        re_ = [t["F_self_readopt"] for t in pp["targets"] if t.get("F_self_readopt") is not None]
        ledger += [
            ("A, known persona", "H5a policy text transfers: pooled T_policy CI > 0, >= 5 of 6 pairs sig. +",
             L["H5a"]["pass"], f"T={ci_s(po['T_policy'])}; {L['H5a']['pairs_sig_positive']}/6"),
            ("A, known persona", "H5b policy-conditioned self-description: pooled T_desc CI > 0",
             L["H5b"]["pass"], f"T={ci_s(po['T_desc'])}"),
            ("A, known persona", "H5c (open) T_policy - T_desc",
             None, f"{ci_s(po['T_policy_minus_T_desc'])}"),
            ("A, known persona", "H5d T_desc > T_native, same receivers and items: CI > 0",
             L["H5d"]["pass"], f"{ci_s(po['T_desc_minus_T_native'])}"),
            ("A, known persona", "H5e (open) self re-adoption F(target, source reading own policy description)",
             None, f"{min(re_):.3f} to {max(re_):.3f} over {len(re_)} targets"),
        ]
    preds = [x for x in ledger if x[2] is not None]
    n_pass = sum(1 for x in preds if x[2])
    n_open = len(ledger) - len(preds)
    print(f"\nprediction ledger: {n_pass}/{len(preds)} predicted criteria met; {n_open} registered open questions reported, not scored")
    for g, text, ok, ev in ledger:
        tag = "OPEN" if ok is None else ("PASS" if ok else "FAIL")
        print(f"  [{tag}] {g:<10} {text:<62} {ev}")
    out["ledger"] = [{"grid": g, "criterion": t, "pass": (None if o is None else bool(o)), "evidence": e}
                     for g, t, o, e in ledger]
    out["ledger_summary"] = {"predicted": len(preds), "met": n_pass, "open": n_open}
    head = r"Grid & Registered criterion & Met & Observed \\"
    cap = (r"\caption{Every criterion registered in the five preregistered protocols (the timestamped "
           r"\texttt{protocol\_arxiv\_*.md} files in \texttt{pilot/} of the repository) and its outcome. Predicted criteria: "
           + f"{n_pass} of {len(preds)} met. " + f"{n_open} registered open questions (marked ``open'') "
           r"carried no directional prediction; their rows record the observed outcome and are not counted.}"
           r"\label{tab:ledger} \\")
    lines = [r"{\scriptsize\renewcommand{\arraystretch}{0.88}",
             r"\begin{longtable}{>{\raggedright\arraybackslash}p{1.9cm}>{\raggedright\arraybackslash}p{7.4cm}>{\raggedright\arraybackslash}p{0.8cm}>{\raggedright\arraybackslash}p{4.3cm}}",
             cap, r"\toprule", head, r"\midrule", r"\endfirsthead",
             r"\multicolumn{4}{l}{\footnotesize Table~\ref{tab:ledger}, continued} \\", r"\toprule", head, r"\midrule", r"\endhead",
             r"\midrule", r"\multicolumn{4}{r}{\footnotesize continued on the next page} \\", r"\endfoot",
             r"\bottomrule", r"\endlastfoot"]
    def tex(s):
        return (s.replace("_", "-").replace(">=", "$\\geq$").replace("<=", "$\\leq$").replace(">", "$>$")
                .replace("<", "$<$").replace("|", "$|$").replace("[", "(").replace("]", ")"))
    for g, t, o, e in ledger:
        met = "open" if o is None else ("yes" if o else "no")
        lines.append(f"{g} & {tex(t)} & {met} & {tex(e)} \\\\")
    lines += [r"\end{longtable}}"]
    (ROOT / "arxiv" / "tab_ledger.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------- 3. framed vs neutral at 500, cross cells only ----------
    def cell_F(df):
        res = {}
        for cid, g in df.groupby("cell_id"):
            a = g.iloc[0]["author"]
            fid = ag.fidelity(ref[a], tbl_of(g))
            if fid:
                res[(g.iloc[0]["receiver"], a)] = fid
        return res
    f500 = cell_F(gm[(gm["arm"] == "framed") & (gm["length"] == 500) & (gm["regen"] == 0)])
    n500 = cell_F(gm[(gm["arm"] == "neutral") & (gm["length"] == 500) & (gm["regen"] == 0)])
    d_cross = [f500[k]["F"] - n500[k]["F"] for k in f500 if k in n500 and k[0] != k[1]]
    d_self = [f500[k]["F"] - n500[k]["F"] for k in f500 if k in n500 and k[0] == k[1]]
    print(f"\nframed - neutral at 500: cross cells mean {np.mean(d_cross):+.3f} (n={len(d_cross)}), "
          f"self cells mean {np.mean(d_self):+.3f} (n={len(d_self)})")
    out["framed_minus_neutral_500"] = {"cross_mean": float(np.mean(d_cross)), "n_cross": len(d_cross),
                                       "self_mean": float(np.mean(d_self)), "n_self": len(d_self)}

    # ---------- 4. length differences with CIs (cross cells, neutral) ----------
    res = {}
    for a, b in ((2000, 100), (500, 100), (2000, 500)):
        fa = cell_F(gm[(gm["arm"] == "neutral") & (gm["length"] == a) & (gm["regen"] == 0)])
        fb = cell_F(gm[(gm["arm"] == "neutral") & (gm["length"] == b) & (gm["regen"] == 0)])
        pair_diffs = []  # one {item: difference} per cross pair
        for k in fa:
            if k in fb and k[0] != k[1]:
                da, db = dict(zip(fa[k]["items"], fa[k]["per_item"])), dict(zip(fb[k]["items"], fb[k]["per_item"]))
                pair_diffs.append({i: da[i] - db[i] for i in set(da) & set(db)})
        # joint item resampling: one item resample per replicate, applied to every pair
        common = sorted(set.intersection(*(set(p) for p in pair_diffs)))
        D = np.array([[p[i] for i in common] for p in pair_diffs])  # pairs x items
        rng = np.random.default_rng(14)
        means = np.array([D[:, rng.integers(0, D.shape[1], D.shape[1])].mean() for _ in range(10000)])
        res[f"{a}-{b}"] = {"mean": float(D.mean()), "ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
                           "n_pairs": len(pair_diffs), "n_items": len(common), "resampling": "joint over items"}
        print(f"length {a} - {b} (cross, neutral, joint): {res[f'{a}-{b}']['mean']:+.4f} "
              f"CI [{res[f'{a}-{b}']['ci'][0]:+.4f}, {res[f'{a}-{b}']['ci'][1]:+.4f}] n={len(pair_diffs)} items={len(common)}")
    out["length_diffs"] = res

    # ---------- 5. baseline-adjusted self vs cross ----------
    retest = {}
    for m, g in gm[gm["arm"] == "retest"].groupby("receiver"):
        fid = ag.fidelity(ref[m], tbl_of(g))
        if fid:
            retest[m] = fid["F"]
    adj = {}
    for (arm, length), g in gm[gm["arm"].isin(["neutral", "framed"]) & (gm["regen"] == 0)].groupby(["arm", "length"]):
        F = cell_F(g)
        self_loss = [F[k]["F"] - retest[k[0]] for k in F if k[0] == k[1] and k[0] in retest]
        cross_gain = [F[k]["F"] - ag.fidelity(ref[k[1]], ref[k[0]])["F"] for k in F if k[0] != k[1]]
        adj[f"{arm}-{int(length)}"] = {"self_minus_C": float(np.mean(self_loss)), "cross_minus_B": float(np.mean(cross_gain))}
        print(f"baseline-adjusted {arm}-{int(length)}: F_self - C = {np.mean(self_loss):+.3f}, F_cross - B = {np.mean(cross_gain):+.3f}")
    out["baseline_adjusted"] = adj
    OUT.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"\nwritten {OUT}, arxiv/tab_coverage.tex, arxiv/tab_ledger.tex")


if __name__ == "__main__":
    main()
