"""Figure 1 (hero): dose-response scatter with decomposition bands.

x = actual self-description length (words, log2), y = fidelity F vs author.
Series: F_self (blue circles), F_cross (orange triangles); baseline B as a
neutral gray band. Flat clouds + persistent gap = the whole result in one plot.
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_grid as ag

# palette (validated: see dataviz skill run)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, SEC, MUTED, GRID_C, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#ffffff"

ag.GRID = ag.ROOT / "runs" / "grid_snapshot.jsonl"


def collect():
    import json
    grid = ag.load_jsonl(ag.GRID)
    screen = ag.load_jsonl(ag.SCREEN)
    words = {}
    for line in ag.MANIFEST.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("ok") and "words" in rec:
            words[rec["desc_id"]] = rec["words"]

    final_items = set(grid["item_id"].unique())
    ref = ag.p_hat_table(screen[screen["item_id"].isin(final_items)], "short")
    for model, tbl in ag.p_hat_table(grid[grid["arm"] == "baseline"], "receiver").items():
        ref[model] = tbl

    pts = {"self": [], "cross": []}
    b_vals = []
    cells = grid[(grid["arm"] == "neutral") & (grid["regen"] == 0)]
    for cell_id, g in cells.groupby("cell_id"):
        meta = g.iloc[0]
        ok = g[g["choice"].notna()]
        tbl = {i: (gg["choice"] == "option_a").mean() for i, gg in ok.groupby("item_id")}
        author = meta["author"]
        if author not in ref:
            continue
        fid = ag.fidelity(ref[author], tbl)
        if fid is None:
            continue
        w = words.get(f"{author}|neutral|{meta['length']}|0")
        if w is None:
            continue
        kind = "self" if meta["receiver"] == author else "cross"
        pts[kind].append((w, fid["F"]))
    for a in ref:
        for r in ref:
            if a != r:
                fid = ag.fidelity(ref[a], ref[r])
                if fid:
                    b_vals.append(fid["F"])
    return pts, float(np.mean(b_vals)), float(np.std(b_vals) / np.sqrt(len(b_vals)))


def main() -> None:
    pts, b_mean, b_se = collect()
    self_pts = np.array(pts["self"])
    cross_pts = np.array(pts["cross"])

    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # baseline band (B +/- 2 SE) — neutral gray, recessive
    ax.axhspan(b_mean - 2 * b_se, b_mean + 2 * b_se, color="#f0efec", zorder=1)
    ax.axhline(b_mean, color=BASE, lw=1.2, zorder=2)

    ax.scatter(cross_pts[:, 0], cross_pts[:, 1], s=34, marker="^",
               facecolor=ORANGE, edgecolor=SURFACE, linewidth=0.8,
               zorder=4, label=r"$F_{\rm cross}$ (other model + description)")
    ax.scatter(self_pts[:, 0], self_pts[:, 1], s=34, marker="o",
               facecolor=BLUE, edgecolor=SURFACE, linewidth=0.8,
               zorder=5, label=r"$F_{\rm self}$ (same weights + description)")

    # per-series mean lines (dashed, series color, thin)
    for arr, color in ((self_pts, BLUE), (cross_pts, ORANGE)):
        ax.axhline(arr[:, 1].mean(), color=color, lw=1.2, ls=(0, (4, 3)),
                   alpha=0.85, zorder=3)

    # direct labels at right margin (clear of the data cloud)
    x_lab = 3100
    ax.text(x_lab, self_pts[:, 1].mean() + 0.002, r"$F_{\rm self}$", color=BLUE,
            fontsize=10, va="bottom", fontweight="bold")
    ax.text(x_lab, cross_pts[:, 1].mean() - 0.006, r"$F_{\rm cross}$",
            color=ORANGE, fontsize=10, va="top", fontweight="bold")
    ax.text(x_lab, b_mean + 0.002, r"$B$ (generic prompt)", color=SEC,
            fontsize=8.5, va="bottom", fontweight="bold")

    # annotate the two contrasts with a bracket-free note
    ax.annotate(
        r"$R_{\rm weight} = F_{\rm self} - F_{\rm cross} > 0$: only the author's"
        "\nown weights recover the profile",
        xy=(0.02, 0.97), xycoords="axes fraction", fontsize=8.5, color=INK,
        va="top")
    ax.annotate(
        r"$T_{\rm script} = F_{\rm cross} - B \leq 0$: the description transports"
        "\nnothing another model can execute",
        xy=(0.02, 0.13), xycoords="axes fraction", fontsize=8.5, color=INK,
        va="top")

    ax.set_xscale("log", base=2)
    ax.set_xticks([100, 200, 500, 1000, 2000])
    ax.set_xticklabels(["100", "200", "500", "1000", "2000"])
    ax.set_xlim(70, 5600)
    ax.set_ylim(0.78, 1.012)
    ax.set_xlabel("Self-description length (actual words, log scale)",
                  fontsize=9.5, color=SEC)
    ax.set_ylabel(r"Reconstruction fidelity  $F = 1 - \overline{\rm JSD}$",
                  fontsize=9.5, color=SEC)

    ax.grid(True, axis="y", color=GRID_C, lw=0.6, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=8.5)

    leg = ax.legend(loc="lower right", fontsize=8.5, frameon=False,
                    borderaxespad=0.2)
    for t in leg.get_texts():
        t.set_color(SEC)

    fig.tight_layout()
    out = ag.ROOT / "report" / "fig1_dose_decomposition.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"self n={len(self_pts)} mean={self_pts[:,1].mean():.3f} | "
          f"cross n={len(cross_pts)} mean={cross_pts[:,1].mean():.3f} | "
          f"B={b_mean:.3f}+-{2*b_se:.3f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
