"""One-off: r0 vs r1 regeneration robustness at neutral-500 (Grid A)."""
import numpy as np

import analyze_grid as ag

ag.GRID = ag.ROOT / "runs" / "grid_snapshot.jsonl"

grid = ag.load_jsonl(ag.GRID)
screen = ag.load_jsonl(ag.SCREEN)
final_items = set(grid["item_id"].unique())
ref = ag.p_hat_table(screen[screen["item_id"].isin(final_items)], "short")
for model, tbl in ag.p_hat_table(grid[grid["arm"] == "baseline"], "receiver").items():
    ref[model] = tbl

cells = grid[(grid["arm"] == "neutral") & (grid["length"] == 500)]
pairs = {}
for cell_id, g in cells.groupby("cell_id"):
    meta = g.iloc[0]
    ok = g[g["choice"].notna()]
    tbl = {i: (gg["choice"] == "option_a").mean() for i, gg in ok.groupby("item_id")}
    fid = ag.fidelity(ref.get(meta["author"], {}), tbl)
    if fid:
        key = (meta["author"], meta["receiver"])
        pairs.setdefault(key, {})[int(meta["regen"])] = fid["F"]

both = {k: v for k, v in pairs.items() if 0 in v and 1 in v}
r0 = np.array([v[0] for v in both.values()])
r1 = np.array([v[1] for v in both.values()])
print(f"cells with both regens: {len(both)}")
for (a, r), v in sorted(both.items()):
    print(f"  {a:>14} -> {r:<14} r0={v[0]:.3f} r1={v[1]:.3f} d={v[1]-v[0]:+.3f}")
print(f"mean |r1-r0| = {np.abs(r1-r0).mean():.3f}; r0 mean {r0.mean():.3f}, "
      f"r1 mean {r1.mean():.3f}; corr={np.corrcoef(r0, r1)[0,1]:.3f}")
