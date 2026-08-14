"""One-off: summarize failed calls in runs/pilot.jsonl by model and error."""
import json
from collections import Counter
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent / "runs" / "pilot.jsonl"

latest = {}
for line in RUN.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    rec = json.loads(line)
    latest[rec["call_id"]] = rec  # last write wins

fails = [r for r in latest.values() if not r.get("ok")]
print(f"failed (latest per call_id): {len(fails)} / {len(latest)}")
print()
by_model = Counter(r["short"] for r in fails)
for m, n in by_model.most_common():
    print(f"  {m:<16} {n}")
print()
errs = Counter((r["short"], (r.get("error") or "")[:120]) for r in fails)
for (m, e), n in errs.most_common(12):
    print(f"  [{n:>3}x] {m}: {e}")
