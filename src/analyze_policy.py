"""Analysis for the known-semantic-persona experiment (pilot/protocol_arxiv_policy_persona.md).

For each passing target (S, P) and receiver R != S, on fold-1 items:
  B'       = F(p_{S|P}, p_R)
  T_policy = F(p_{S|P}, p_{R|P}) - B'
  T_desc   = F(p_{S|P}, p_{R|D_{S,P}}) - B'
  T_native = F(p_S, p_{R|D_S}) - F(p_S, p_R)   (existing neutral-500 cells, same items)
Pooled values with joint item resampling; per directed receiver pair with a paired item
bootstrap (H5a). Secondary: T_policy on all 110 items. Writes results/arxiv/policy_persona.json.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

import analyze_grid as ag
from run_arxiv_demos_balanced import SEED, folds
from run_arxiv_policy import GATE_PATH, OUT_PATH, POLICIES

ROOT = Path(__file__).resolve().parent.parent
MERGED = ROOT / "results" / "arxiv" / "grid_merged.jsonl"
ITEMS_FINAL = ROOT / "battery" / "items_final.json"
OUT = ROOT / "results" / "arxiv" / "policy_persona.json"
N_BOOT = 10_000
rng = np.random.default_rng(14)


def tbl_of(g):
    ok = g[g["choice"].notna()]
    return {i: (gg["choice"] == "option_a").mean() for i, gg in ok.groupby("item_id")}


def per_item_F(p: dict, q: dict, items) -> dict:
    common = [i for i in items if i in p and i in q]
    a, b = np.array([p[i] for i in common]), np.array([q[i] for i in common])
    return dict(zip(common, 1 - ag.jsd(a, b)))


def joint(pairs: list[dict]) -> dict | None:
    if not pairs:
        return None
    common = sorted(set.intersection(*(set(p) for p in pairs)))
    D = np.array([[p[i] for i in common] for p in pairs])
    idx = rng.integers(0, D.shape[1], size=(N_BOOT, D.shape[1]))
    means = np.array([D[:, ix].mean() for ix in idx])
    return {"n_pairs": len(pairs), "n_items": len(common), "mean": float(D.mean()),
            "ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


def paired(d: dict) -> dict:
    v = np.array(list(d.values()))
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    means = v[idx].mean(axis=1)
    return {"delta": float(v.mean()), "ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "n": len(v)}


def main() -> None:
    final = json.loads(ITEMS_FINAL.read_text(encoding="utf-8"))
    fold = folds(final, random.Random(SEED))
    eval_items, all_items = sorted(fold[1]), sorted(it["id"] for it in final)
    rows = ag.load_jsonl(OUT_PATH)
    grid = ag.load_jsonl(MERGED)
    screen = ag.load_jsonl(ag.SCREEN)
    ref = ag.p_hat_table(screen[screen["item_id"].isin(set(all_items))], "short")
    gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))

    def policy_cell(r, name):
        return tbl_of(rows[(rows["arm"] == "policy") & (rows["receiver"] == r) & (rows["policy"] == name)])

    def desc_cell(r, s, name):
        return tbl_of(rows[(rows["arm"] == "policy_desc") & (rows["receiver"] == r) & (rows["author"] == s)
                           & (rows["policy"] == name)])

    def native_cell(r, s):
        return tbl_of(grid[(grid["receiver"] == r) & (grid["author"] == s) & (grid["arm"] == "neutral")
                           & (grid["length"] == 500) & (grid["regen"] == 0)])

    out = {"gate": gate, "targets": [], "pairs": []}
    T_pol, T_desc, T_nat, T_pol_all = [], [], [], []
    T_pol_d, T_nat_d = [], []  # aligned with T_desc (pairs whose description cell is complete)
    by_dir = defaultdict(list)
    for s, name in gate["passing"]:
        target = policy_cell(s, name)
        self_desc = desc_cell(s, s, name)
        if len([i for i in eval_items if i in self_desc]) < 50:
            print(f"  WARNING incomplete self cell {s}/{name}: {len(self_desc)} items; excluded")
            self_desc = {}
        trow = {"source": s, "policy": name,
                "F_target_to_generic_eval": float(np.mean(list(per_item_F(ref[s], target, eval_items).values()))),
                "n_items_target_eval": len([i for i in eval_items if i in target])}
        if self_desc:
            trow["F_self_readopt"] = float(np.mean(list(per_item_F(target, self_desc, eval_items).values())))
            trow["F_self_desc_to_native"] = float(np.mean(list(per_item_F(ref[s], self_desc, eval_items).values())))
        out["targets"].append(trow)
        for r in ag.GRID_A:
            if r == s:
                continue
            Bp = per_item_F(target, ref[r], eval_items)
            Fp = per_item_F(target, policy_cell(r, name), eval_items)
            d_pol = {i: Fp[i] - Bp[i] for i in Fp if i in Bp}
            Bp_all = per_item_F(target, ref[r], all_items)
            Fp_all = per_item_F(target, policy_cell(r, name), all_items)
            T_pol_all.append({i: Fp_all[i] - Bp_all[i] for i in Fp_all if i in Bp_all})
            dc = desc_cell(r, s, name)
            d_desc = None
            if len([i for i in eval_items if i in dc]) < 50:
                print(f"  WARNING incomplete description cell {r} <- {s}/{name}: {len(dc)} items; excluded")
                dc = {}
            if dc:
                Fd = per_item_F(target, dc, eval_items)
                d_desc = {i: Fd[i] - Bp[i] for i in Fd if i in Bp}
            Bn = per_item_F(ref[s], ref[r], eval_items)
            Fn = per_item_F(ref[s], native_cell(r, s), eval_items)
            d_nat = {i: Fn[i] - Bn[i] for i in Fn if i in Bn}
            prow = {"source": s, "policy": name, "receiver": r, "B_prime": float(np.mean(list(Bp.values()))),
                    "F_policy": float(np.mean(list(Fp.values()))), "T_policy": paired(d_pol),
                    "F_desc": float(np.mean(list(Fd.values()))) if d_desc else None,
                    "T_desc": paired(d_desc) if d_desc else None, "T_native": paired(d_nat)}
            out["pairs"].append(prow)
            T_pol.append(d_pol)
            by_dir[(r, s)].append(d_pol)
            T_nat.append(d_nat)
            if d_desc:
                T_desc.append(d_desc)
                T_pol_d.append(d_pol)
                T_nat_d.append(d_nat)

    pooled = {"T_policy": joint(T_pol), "T_desc": joint(T_desc), "T_native_same_items": joint(T_nat),
              "T_policy_all_items": joint(T_pol_all), "T_native_desc_pairs": joint(T_nat_d)}
    if T_desc:
        pooled["T_policy_minus_T_desc"] = joint([{i: a[i] - b[i] for i in a if i in b} for a, b in zip(T_pol_d, T_desc)])
        pooled["T_desc_minus_T_native"] = joint([{i: a[i] - b[i] for i in a if i in b} for a, b in zip(T_desc, T_nat_d)])
    out["pooled"] = pooled
    # H5a per directed receiver pair, pooled over policies
    dirs = {}
    for (r, s), ds in by_dir.items():
        common = sorted(set.intersection(*(set(d) for d in ds)))
        dirs[f"{s}->{r}"] = paired({i: float(np.mean([d[i] for d in ds])) for i in common})
    out["T_policy_by_directed_pair"] = dirs
    n_pos = sum(v["ci"][0] > 0 for v in dirs.values())
    out["ledger"] = {
        "H5a": {"pooled_ci_excludes_zero": bool(pooled["T_policy"] and pooled["T_policy"]["ci"][0] > 0),
                "pairs_sig_positive": n_pos, "n_pairs": len(dirs), "pass": bool(pooled["T_policy"] and pooled["T_policy"]["ci"][0] > 0 and n_pos >= 5)},
        "H5b": {"pass": bool(pooled["T_desc"] and pooled["T_desc"]["ci"][0] > 0)} if pooled["T_desc"] else None,
        "H5c_open": pooled.get("T_policy_minus_T_desc"),
        "H5d": {"pass": bool(pooled.get("T_desc_minus_T_native") and pooled["T_desc_minus_T_native"]["ci"][0] > 0)}
        if pooled.get("T_desc_minus_T_native") else None,
        "H5e_open": [{k: t.get(k) for k in ("source", "policy", "F_self_readopt", "F_self_desc_to_native")} for t in out["targets"]],
    }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")

    print("TARGETS (fold-1 items)")
    for t in out["targets"]:
        print(f"  {t['source']:<12} {t['policy']:<9} F(target vs generic)={t['F_target_to_generic_eval']:.3f}"
              + (f"  self re-adopt F={t['F_self_readopt']:.3f}  self-desc vs native={t['F_self_desc_to_native']:.3f}"
                 if "F_self_readopt" in t else ""))
    print("PAIRS")
    for p in out["pairs"]:
        td = p["T_desc"]
        print(f"  {p['source']:<12} {p['policy']:<9} -> {p['receiver']:<12} B'={p['B_prime']:.3f} "
              f"F_policy={p['F_policy']:.3f} T_policy={p['T_policy']['delta']:+.3f} [{p['T_policy']['ci'][0]:+.3f},{p['T_policy']['ci'][1]:+.3f}]"
              + (f"  F_desc={p['F_desc']:.3f} T_desc={td['delta']:+.3f} [{td['ci'][0]:+.3f},{td['ci'][1]:+.3f}]" if td else "")
              + f"  T_native={p['T_native']['delta']:+.3f}")
    print("POOLED (joint item bootstrap)")
    for k, v in pooled.items():
        if v:
            print(f"  {k:<24} {v['mean']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]  pairs {v['n_pairs']} items {v['n_items']}")
    print("T_policy by directed pair:", {k: f"{v['delta']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]" for k, v in dirs.items()})
    print("LEDGER", json.dumps(out["ledger"], indent=1)[:1500])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
