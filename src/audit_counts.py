"""One-off: verify every gate-table row is backed by a full 240-call run."""
import json
from collections import Counter
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent / "runs" / "pilot.jsonl"
latest = {}
for line in RUN.read_text(encoding="utf-8").splitlines():
    if line.strip():
        rec = json.loads(line)
        latest[rec["call_id"]] = rec

ok_counts = Counter(r["short"] for r in latest.values() if r.get("ok"))
item_counts = {}
for short in ok_counts:
    items = {r["item_id"] for r in latest.values()
             if r.get("ok") and r["short"] == short and r.get("choice")}
    item_counts[short] = len(items)

for short in sorted(ok_counts):
    flag = "  <-- PARTIAL" if ok_counts[short] < 240 else ""
    print(f"{short:<18} ok_calls={ok_counts[short]:>3}  items_covered={item_counts[short]:>2}{flag}")
