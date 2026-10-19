"""arXiv-revision analysis. Leaves the sprint pipeline untouched.

1. Merge runs/grid.jsonl with the arXiv re-runs:
     - the four hermes3-authored framed-500/2000 cells whose r0 description was
       a refusal are replaced by the r1 re-run (runs/arxiv_hermes_r1.jsonl)
       when that file is complete for a cell; otherwise the sprint cell is kept
       and flagged.
   Writes results/arxiv/grid_merged.jsonl + manifest_merged.jsonl and reruns
   the full sprint analysis (analyze_grid.main) on them.
2. Positive controls (runs/arxiv_positive_controls.jsonl): few-shot-30,
   oracle-110, matched-information self-observation descriptions. Per-cell F,
   B, T_script with paired bootstrap CIs, pooled per arm, and the
   self-observation vs observer-written comparison.
Writes results/arxiv/analysis.txt and results/arxiv/positive_controls.json.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_grid as ag

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "arxiv"
GRID = ROOT / "runs" / "grid.jsonl"
HERMES_R1 = ROOT / "runs" / "arxiv_hermes_r1.jsonl"
POSCTRL = ROOT / "runs" / "arxiv_positive_controls.jsonl"
MANIFEST = ROOT / "descriptions" / "manifest.jsonl"
MANIFEST_ARXIV = ROOT / "descriptions" / "manifest_arxiv.jsonl"
OPTIMIZED = ROOT / "runs" / "arxiv_optimized.jsonl"
DEMOS_DOSE = ROOT / "runs" / "arxiv_demos_dose.jsonl"
GRIDB = ROOT / "runs" / "arxiv_gridb.jsonl"
LOCAL_DEMOS = ROOT / "runs" / "arxiv_local_demos.jsonl"
USERMSG = ROOT / "runs" / "arxiv_usermsg.jsonl"
DEMOS_BAL = ROOT / "runs" / "arxiv_demos_balanced.jsonl"
OBS_STABLE = ROOT / "runs" / "arxiv_obs_stable.jsonl"
GRID_OF = {**{m: "A" for m in ag.GRID_A}, **{m: "B" for m in ag.GRID_B},
           **{m: "L" for m in ag.LOCALS}}
GRID_NAME = {"A": "Grid A", "B": "Grid B (preregistered replication)",
             "L": "local trio (preregistered replication)"}
ADDED_CONDS = ["selfobs-500", "optimized-500", "fewshot-30", "fewshot-60", "fewshot-90",
               "oracle-110", "neutral_user-500", "oracle_user-110", "fewshot_bal-30", "demos_bal-55",
               "3p_stable-500", "selfobs_stable-500"]

REFUSAL_CELLS = {  # sprint cell -> r1 replacement cell
    "hermes3-70b|hermes3-70b|framed|500|r0": "hermes3-70b|hermes3-70b|framed|500|r1",
    "hermes3-70b|hermes3-70b|framed|2000|r0": "hermes3-70b|hermes3-70b|framed|2000|r1",
    "llama3.3-70b|hermes3-70b|framed|500|r0": "llama3.3-70b|hermes3-70b|framed|500|r1",
    "llama3.3-70b|hermes3-70b|framed|2000|r0": "llama3.3-70b|hermes3-70b|framed|2000|r1",
}
CALLS_PER_CELL = 440


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def merge() -> tuple[Path, Path, dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = read_rows(GRID)
    r1 = read_rows(HERMES_R1)
    r1_ok = {}
    for r in r1:
        if r.get("ok"):
            r1_ok.setdefault(r["cell_id"], {})[r["call_id"]] = r
    status = {}
    replaced_cells = set()
    merged = []
    for sprint_cell, r1_cell in REFUSAL_CELLS.items():
        n = len(r1_ok.get(r1_cell, {}))
        complete = n >= CALLS_PER_CELL
        status[sprint_cell] = {"r1_calls_ok": n, "replaced": complete}
        if complete:
            replaced_cells.add(sprint_cell)
            for rec in r1_ok[r1_cell].values():
                rec = dict(rec)
                rec["substituted_from_regen"] = 1
                rec["regen"] = 0
                rec["cell_id"] = sprint_cell
                rec["call_id"] = rec["call_id"].replace(f"|{r1_cell}|", f"|{sprint_cell}|")
                merged.append(rec)
    merged = [r for r in grid if r.get("cell_id") not in replaced_cells] + merged
    gpath = OUT_DIR / "grid_merged.jsonl"
    gpath.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n",
                     encoding="utf-8")

    # manifest: point the two replaced r0 description ids at the r1 word counts
    man = read_rows(MANIFEST)
    words = {m["desc_id"]: m["words"] for m in man if m.get("ok")}
    out_man = []
    for m in man:
        m = dict(m)
        did = m.get("desc_id", "")
        if m.get("ok") and did in ("hermes3-70b|framed|500|0", "hermes3-70b|framed|2000|0"):
            cell = f"hermes3-70b|hermes3-70b|framed|{did.split('|')[2]}|r0"
            if cell in replaced_cells:
                m["words"] = words[did.replace("|0", "|1")]
                m["substituted_from_regen"] = 1
        out_man.append(m)
    mpath = OUT_DIR / "manifest_merged.jsonl"
    mpath.write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in out_man) + "\n",
                     encoding="utf-8")
    return gpath, mpath, status


def run_base_analysis(gpath: Path, mpath: Path) -> str:
    ag.GRID, ag.MANIFEST, ag.OUT = gpath, mpath, OUT_DIR / "grid_summary.json"
    ag.RNG = np.random.default_rng(14)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ag.main()
    return buf.getvalue()


def positive_controls(gpath: Path) -> tuple[str, dict]:
    lines: list[str] = []
    out: dict = {"cells": [], "pooled": {}, "pairs_B": {}}
    pc = read_rows(POSCTRL)
    if not pc:
        return "positive controls: no data yet\n", out
    pcdf = ag.load_jsonl(POSCTRL)
    if OPTIMIZED.exists():  # battery test cells of the optimized-description experiment
        odf = ag.load_jsonl(OPTIMIZED)
        odf = odf[odf["phase"] == "test"]
        pcdf = pd.concat([pcdf, odf[pcdf.columns.intersection(odf.columns)]], ignore_index=True)
    for extra in (DEMOS_DOSE, GRIDB, LOCAL_DEMOS, USERMSG, DEMOS_BAL, OBS_STABLE):  # k = 60/90 dose; Grid B and local-trio replications
        if extra.exists():
            edf = ag.load_jsonl(extra)
            pcdf = pd.concat([pcdf, edf[pcdf.columns.intersection(edf.columns)]], ignore_index=True)
    grid = ag.load_jsonl(gpath)
    screen = ag.load_jsonl(ag.SCREEN)
    final_items = set(grid["item_id"].unique())
    ref = ag.p_hat_table(screen[screen["item_id"].isin(final_items)], "short")
    for model, tbl in ag.p_hat_table(grid[grid["arm"] == "baseline"], "receiver").items():
        ref[model] = tbl

    def cell_tables(df: pd.DataFrame) -> dict:
        res = {}
        for cell_id, g in df.groupby("cell_id"):
            ok = g[g["choice"].notna()]
            tbl = {i: (gg["choice"] == "option_a").mean() for i, gg in ok.groupby("item_id")}
            res[cell_id] = (g.iloc[0], tbl, len(g), int(g["choice"].notna().sum()))
        return res

    B = {}
    for a in ref:
        for r in ref:
            if a != r and GRID_OF.get(a) == GRID_OF.get(r):
                B[(a, r)] = ag.fidelity(ref[a], ref[r])
                if GRID_OF.get(a) == "A":
                    out["pairs_B"][f"{a}->{r}"] = B[(a, r)]["F"]

    lines.append("=" * 78)
    lines.append("8. CONDITIONS ADDED AFTER REVIEW: Grid A, then preregistered replications")
    lines.append("   F vs the AUTHOR's generic-prompt reference; T = F_cross - B (paired bootstrap)")
    lines.append("=" * 78)
    rows = []
    for cell_id, (meta, tbl, n_calls, n_ok) in cell_tables(pcdf).items():
        if n_calls < CALLS_PER_CELL:
            lines.append(f"  {cell_id:<48} incomplete ({n_calls} calls), skipped")
            continue
        fid = ag.fidelity(ref[meta["author"]], tbl)
        if fid is None:
            lines.append(f"  {cell_id:<48} partial ({n_calls} calls), skipped")
            continue
        # per-cell bootstrap seed, so a cell's CI does not depend on which other cells exist
        ag.RNG = np.random.default_rng(int(hashlib.sha256(cell_id.encode()).hexdigest()[:8], 16))
        lo, hi = ag.boot_ci(fid["per_item"])
        row = {"cell_id": cell_id, "receiver": meta["receiver"], "author": meta["author"],
               "grid": GRID_OF[meta["receiver"]],
               "arm": meta["arm"], "length": int(meta["length"]),
               "cond": f"{meta['arm']}-{int(meta['length'])}",
               "kind": "self" if meta["receiver"] == meta["author"] else "cross",
               "F": fid["F"], "F_ci": [lo, hi], "r": fid["r"], "agree": fid["agree"],
               "n_items": fid["n_items"], "n_calls": n_calls, "compliance": n_ok / n_calls}
        if row["kind"] == "cross":
            b = B[(meta["author"], meta["receiver"])]
            row["B"] = b["F"]
            row["T"] = ag.boot_contrast(fid["per_item"], b["per_item"], fid["items"], b["items"])
        rows.append(row)
        out["cells"].append(row)
    rdf = pd.DataFrame(rows)
    for gid, cond in [(g, c) for g in ("A", "B", "L") for c in ADDED_CONDS + ["3p-500"]]:
        if cond == "3p-500" and gid == "A":
            continue  # Grid A observer-written cells live in the sprint grid (handled below)
        arm = cond if gid == "A" else f"{gid}:{cond}"  # key: plain for Grid A (figure/table scripts)
        g = rdf[(rdf["cond"] == cond) & (rdf["grid"] == gid)] if len(rdf) else rdf
        if not len(g):
            continue
        lines.append(f"\n  --- {GRID_NAME[gid]}: {cond} ---")
        for _, c in g.sort_values(["kind", "author", "receiver"]).iterrows():
            if c["kind"] == "cross":
                t = c["T"]
                sig = "*" if (t["ci"][0] > 0 or t["ci"][1] < 0) else " "
                lines.append(f"  {sig} cross {c['author']:>12} -> {c['receiver']:<12} "
                             f"F={c['F']:.3f} B={c['B']:.3f} T={t['delta']:+.3f} "
                             f"CI[{t['ci'][0]:+.3f},{t['ci'][1]:+.3f}] agree={c['agree']:.2f} "
                             f"comp={c['compliance']:.3f} ({c['n_calls']} calls)")
            else:
                lines.append(f"    self  {c['author']:>12} -> {c['receiver']:<12} "
                             f"F={c['F']:.3f} CI[{c['F_ci'][0]:.3f},{c['F_ci'][1]:.3f}] "
                             f"agree={c['agree']:.2f} comp={c['compliance']:.3f} ({c['n_calls']} calls)")
        selfs, crosses = g[g["kind"] == "self"], g[g["kind"] == "cross"]
        pooled = {"F_self": float(selfs["F"].mean()) if len(selfs) else None,
                  "F_cross": float(crosses["F"].mean()) if len(crosses) else None,
                  "B": float(crosses["B"].mean()) if len(crosses) else None,
                  "n_self_cells": int(len(selfs)), "n_cross_cells": int(len(crosses))}
        if pooled["F_cross"] is not None:
            pooled["T_script"] = pooled["F_cross"] - pooled["B"]
            pooled["n_pairs_T_pos_sig"] = int(sum(1 for t in crosses["T"] if t["ci"][0] > 0))
            pooled["n_pairs_T_neg_sig"] = int(sum(1 for t in crosses["T"] if t["ci"][1] < 0))
        if pooled["F_self"] is not None and pooled["F_cross"] is not None:
            pooled["R_weight"] = pooled["F_self"] - pooled["F_cross"]
        out["pooled"][arm] = pooled
        lines.append(f"  POOLED {arm}: " + "  ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in pooled.items()))

    # self-observation vs observer-written (3p) descriptions, same information + prompt.
    # Grid A observer-written cells are sprint cells (grid.jsonl); Grid B ones were run here.
    comp: dict = {}
    for cell_id, (meta, tbl, n_calls, _) in cell_tables(grid[grid["arm"] == "3p"]).items():
        fid = ag.fidelity(ref[meta["author"]], tbl)
        if fid:
            kind = "self" if meta["receiver"] == meta["author"] else "cross"
            comp.setdefault((GRID_OF[meta["receiver"]], meta["author"], kind), []).append(fid["F"])
    if len(rdf):
        for _, c in rdf[rdf["cond"] == "3p-500"].iterrows():
            comp.setdefault((c["grid"], c["author"], c["kind"]), []).append(c["F"])
    out["authorship"] = {}
    for gid in ("A", "B"):
        so = rdf[(rdf["cond"] == "selfobs-500") & (rdf["grid"] == gid)] if len(rdf) else rdf
        have3 = [k for k in comp if k[0] == gid]
        if not len(so) or not have3:
            continue
        lines.append(f"\n  --- {GRID_NAME[gid]}: authorship control, self-observation vs observer-written "
                     f"(same 90 observations, same prompt) ---")
        per_target = []
        for author in sorted(so["author"].unique()):
            for kind in ("self", "cross"):
                s = so[(so["author"] == author) & (so["kind"] == kind)]["F"]
                o = comp.get((gid, author, kind), [])
                if len(s) and o:
                    d = float(s.mean() - np.mean(o))
                    per_target.append({"author": author, "kind": kind, "selfobs_F": float(s.mean()),
                                       "observer_F": float(np.mean(o)), "d": d})
                    lines.append(f"  {kind:<5} target {author:<12} self-written-from-obs F={s.mean():.3f}  "
                                 f"observer-written F={np.mean(o):.3f}  d={d:+.3f}")
        pooled3 = {k: float(np.mean([f for (gg, a, kk), fs in comp.items() if gg == gid and kk == k for f in fs]))
                   for k in ("self", "cross")}
        key = "3p" if gid == "A" else f"{gid}:3p"
        out["pooled"][key] = {"F_self": pooled3["self"], "F_cross": pooled3["cross"]}
        out["authorship"][gid] = per_target
        lines.append(f"  POOLED observer-written: F_self={pooled3['self']:.3f} F_cross={pooled3['cross']:.3f}")

    # preregistered predictions for the replications (pilot/protocol_arxiv_*_replication.md)
    def pooled_of(gid, cond):
        p = out["pooled"].get(f"{gid}:{cond}")
        return p if p and "T_script" in p and "R_weight" in p else None
    checks = []
    for gid in ("B", "L"):
        p110, p30, p90 = pooled_of(gid, "oracle-110"), pooled_of(gid, "fewshot-30"), pooled_of(gid, "fewshot-90")
        if p110:
            checks.append((gid, "P1 demos-110: T>0, 6/6 cross pairs sig. positive, F_cross>=0.93, |R|<=0.02",
                           p110["T_script"] > 0 and p110["n_pairs_T_pos_sig"] == 6
                           and p110["F_cross"] >= 0.93 and abs(p110["R_weight"]) <= 0.02))
        if p30:
            checks.append((gid, "P2 demos-30: T<=0 and no cross pair sig. positive",
                           p30["T_script"] <= 0 and p30["n_pairs_T_pos_sig"] == 0))
        if p90 and gid == "L":
            checks.append((gid, "P3 demos-90: T<=0, no cross pair sig. positive",
                           p90["T_script"] <= 0 and p90["n_pairs_T_pos_sig"] == 0))
        pso = pooled_of(gid, "selfobs-500")
        if pso and gid == "B":
            checks.append((gid, "P3 self-observation: T<0", pso["T_script"] < 0))
            p3 = out["pooled"].get("B:3p")
            if p3:
                dself, dcross = pso["F_self"] - p3["F_self"], pso["F_cross"] - p3["F_cross"]
                signs = {np.sign(t["d"]) for t in out["authorship"].get("B", [])}
                checks.append((gid, f"P4 authorship: |dF_cross|={abs(dcross):.3f}<0.03, |dF_self|={abs(dself):.3f}<0.03, mixed signs",
                               abs(dcross) < 0.03 and abs(dself) < 0.03 and len(signs) > 1))
    if checks:
        lines.append("\n  --- preregistered predictions ---")
        for gid, text, ok in checks:
            lines.append(f"  [{'PASS' if ok else 'FAIL'}] {GRID_NAME[gid]}: {text}")
        out["predictions"] = [{"grid": g, "prediction": t, "pass": bool(o)} for g, t, o in checks]

    lines.append("\n  Grid A pairwise B (author->receiver): " + ", ".join(
        f"{k} {v:.3f}" for k, v in out["pairs_B"].items()))

    # optimized-description selection history (held-out F per round)
    sel = [m for m in read_rows(MANIFEST_ARXIV) if m.get("desc_id", "").endswith("|selected")]
    if sel:
        lines.append("\n  --- optimized descriptions: held-out F by round (round 0 = natural neutral-500) ---")
        out["optimized_selection"] = []
        for m in sel:
            hist = " ".join(f"r{r}={f:.3f}" for r, f in m["history"])
            lines.append(f"  {m['desc_id']:<40} selected r{m['selected_round']} "
                         f"({m['words']} words): {hist}")
            out["optimized_selection"].append(m)
        n_improved = sum(1 for m in sel if m["selected_round"] > 0)
        lines.append(f"  pairs where any rewrite beat the natural description on held-out items: "
                     f"{n_improved}/{len(sel)}")

    # per-cell F of the Grid A sprint conditions the figure compares against
    out["grid_a_reference"] = []
    ga = grid[grid["author"].isin(ag.GRID_A) & grid["receiver"].isin(ag.GRID_A)
              & grid["arm"].isin(["neutral", "framed", "3p", "adopt", "para"])
              & (grid["length"] == 500) & (grid["regen"] == 0)]
    for cell_id, (meta, tbl, _, _) in cell_tables(ga).items():
        fid = ag.fidelity(ref[meta["author"]], tbl)
        if fid:
            out["grid_a_reference"].append({
                "cell_id": cell_id, "receiver": meta["receiver"], "author": meta["author"],
                "arm": meta["arm"], "kind": "self" if meta["receiver"] == meta["author"] else "cross",
                "F": fid["F"]})
    return "\n".join(lines) + "\n", out


def main() -> None:
    gpath, mpath, status = merge()
    text = run_base_analysis(gpath, mpath)
    text += "\nhermes3 refusal-cell substitution status:\n"
    for cell, st in status.items():
        text += f"  {cell:<44} r1 ok calls={st['r1_calls_ok']:>4}  replaced={st['replaced']}\n"
    pc_text, pc = positive_controls(gpath)
    text += "\n" + pc_text
    (OUT_DIR / "analysis.txt").write_text(text, encoding="utf-8")
    (OUT_DIR / "positive_controls.json").write_text(
        json.dumps(pc, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
        encoding="utf-8")
    print(text)
    print(f"written to {OUT_DIR}")


if __name__ == "__main__":
    main()
