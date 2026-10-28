"""Third-person-500 per-pair transfer contrasts for the sig.-pairs and cell-count
columns of the replication table. Grid A cells live in results/arxiv/grid_merged.jsonl
(paired item bootstrap, 10k resamples, as in analyze_arxiv.py); Grid B cells are read
from results/arxiv/positive_controls.json. Writes results/arxiv/third_person_pairs.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_grid as ag  # noqa: E402
from analyze_strength import tbl_of  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PC = ROOT / "results" / "arxiv" / "positive_controls.json"
OUT = ROOT / "results" / "arxiv" / "third_person_pairs.json"
N_BOOT = 10_000


def grid_a() -> dict:
    grid = ag.load_jsonl(ROOT / "results" / "arxiv" / "grid_merged.jsonl")
    screen = ag.load_jsonl(ag.SCREEN)
    final_items = set(grid["item_id"].unique())
    ref = ag.p_hat_table(screen[screen["item_id"].isin(final_items)], "short")
    for m, t in ag.p_hat_table(grid[grid["arm"] == "baseline"], "receiver").items():
        ref[m] = t
    rng = np.random.default_rng(0)
    pairs, n_self = [], 0
    for _, g in grid[grid["arm"] == "3p"].groupby("cell_id"):
        author, receiver = g.iloc[0]["author"], g.iloc[0]["receiver"]
        if author == receiver:
            n_self += 1
            continue
        f = ag.fidelity(ref[author], tbl_of(g))
        b = ag.fidelity(ref[author], ref[receiver])
        fd = dict(zip(f["items"], f["per_item"]))
        bd = dict(zip(b["items"], b["per_item"]))
        items = sorted(set(fd) & set(bd))
        d = np.array([fd[i] - bd[i] for i in items])
        boots = rng.choice(d, size=(N_BOOT, len(d)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        pairs.append({"author": author, "receiver": receiver, "F": f["F"], "B": b["F"],
                      "T": float(d.mean()), "ci": [float(lo), float(hi)], "n_items": len(d),
                      "agree": f["agree"]})
        print(f"A {author}->{receiver}: F={f['F']:.3f} B={b['F']:.3f} T={d.mean():+.3f} "
              f"[{lo:+.3f},{hi:+.3f}] n={len(d)}")
    return summarize(pairs, n_self)


def grid_b() -> dict:
    pc = json.loads(PC.read_text(encoding="utf-8"))
    cells = [c for c in pc["cells"] if c.get("cond") == "3p-500" and c.get("grid") == "B"]
    pairs = [{"author": c["author"], "receiver": c["receiver"], "F": c["F"], "B": c["B"],
              "T": c["T"]["delta"], "ci": c["T"]["ci"], "agree": c["agree"]}
             for c in cells if c["kind"] == "cross"]
    return summarize(pairs, sum(1 for c in cells if c["kind"] == "self"))


def summarize(pairs: list, n_self: int) -> dict:
    return {"pairs": pairs, "n_self_cells": n_self, "n_cross_cells": len(pairs),
            "n_pairs_T_pos_sig": sum(1 for p in pairs if p["ci"][0] > 0),
            "n_pairs_T_neg_sig": sum(1 for p in pairs if p["ci"][1] < 0)}


def main() -> None:
    out = {"A": grid_a(), "B": grid_b()}
    for g, r in out.items():
        print(f"Grid {g}: sig +/- {r['n_pairs_T_pos_sig']}/{r['n_pairs_T_neg_sig']}, "
              f"cells {r['n_self_cells']}/{r['n_cross_cells']}")
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
