"""One-off: verify the Ivan-round-2 additions in final_report.tex."""
import json
from pathlib import Path

C = 0.979
rows = {
    "neutral-100": (0.935, 0.878, 0.892), "neutral-500": (0.918, 0.875, 0.892),
    "neutral-2000": (0.917, 0.880, 0.892), "framed-100": (0.912, 0.858, 0.884),
    "framed-500": (0.924, 0.876, 0.892), "framed-2000": (0.920, 0.870, 0.884),
    "paraphrase-500": (0.912, 0.855, 0.863), "adopt-500": (0.897, 0.850, 0.863),
    "third-person-500": (0.771, 0.739, 0.863),
}
fracs = []
print(f"{'condition':<18} {'C-B':>6} {'R% of range':>12} {'Fcross>B?':>10}")
for k, (fs, fc, b) in rows.items():
    rng = C - b
    rpct = (fs - fc) / rng * 100
    if k != "third-person-500":
        fracs.append(rpct)
    print(f"{k:<18} {rng:6.3f} {rpct:11.0f}% {'YES' if fc > b else 'no':>10}")
print(f"\nself-authored R% span: {min(fracs):.0f}% .. {max(fracs):.0f}%")
print(f"range span: {C - 0.892:.3f} .. {C - 0.863:.3f}")

# actual lengths of r0 framed/neutral descriptions (the length-ablation set)
recs = []
for line in (Path(__file__).resolve().parent.parent / "descriptions" /
             "manifest.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    if r.get("ok") and "words" in r and r["desc_id"].count("|") == 3:
        short, arm, length, regen = r["desc_id"].split("|")
        if regen == "0" and arm in ("neutral", "framed"):
            recs.append((r["words"], r["desc_id"]))
recs.sort()
print(f"\nshortest r0 descriptions: {recs[:3]}")
print(f"longest r0 descriptions: {recs[-3:]}")
