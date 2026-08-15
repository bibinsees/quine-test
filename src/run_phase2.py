"""Phase 2b: transfer cells for third-person and paraphrase descriptions.

Grid A only (cheap, full-control trio). Appends to runs/grid.jsonl (run ONLY
after the main grid completes - single writer at a time).

Cells:
  3p self-reconstruction: receiver = target, desc = desc3p_{target}_by_{describer}
    -> compare vs F_self(target): does self-authorship beat observation?  (6)
  3p cross: receiver = third model, same desc -> compare vs F_cross        (6)
  paraphrase: desc = desc_{author}_para_500_r0, self + cross               (9)

Sampling identical to main grid: 2 orders x 2 samples, temp 1.0.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_client import LLMClient
from run_grid import MODEL_DEFS, GRID_A, N_SAMPLES, TEMPERATURE, OUT_PATH, run_job
from run_pilot import PROMPT_TEMPLATE, parse_choice, stable_seed  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
DESC_DIR = ROOT / "descriptions"
ITEMS_PATH = ROOT / "battery" / "items_final.json"

CREDIT_FLOOR = 5.0
MAX_WORKERS = 10


def build_cells() -> list[dict]:
    cells = []
    for target in GRID_A:
        others = [m for m in GRID_A if m != target]
        for describer in others:
            third = [m for m in others if m != describer][0]
            path = DESC_DIR / f"desc3p_{target}_by_{describer}_500_r0.md"
            system = path.read_text(encoding="utf-8").strip()
            for receiver, kind in ((target, "3p_self"), (third, "3p_cross")):
                cells.append({
                    "cell_id": f"{receiver}|{target}|3p_by_{describer}|500|r0",
                    "receiver": receiver, "author": target,
                    "arm": "3p", "length": 500, "regen": 0, "system": system,
                })
    for author in GRID_A:
        system = (DESC_DIR / f"desc_{author}_para_500_r0.md").read_text(
            encoding="utf-8").strip()
        for receiver in GRID_A:
            cells.append({
                "cell_id": f"{receiver}|{author}|para|500|r0",
                "receiver": receiver, "author": author,
                "arm": "para", "length": 500, "regen": 0, "system": system,
            })
    return cells


def main() -> None:
    items = json.loads(ITEMS_PATH.read_text(encoding="utf-8"))
    done = set()
    if OUT_PATH.exists():
        for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                if rec.get("ok"):
                    done.add(rec["call_id"])
            except json.JSONDecodeError:
                continue

    jobs = []
    for cell in build_cells():
        provider, model_id, extra, max_tok = MODEL_DEFS[cell["receiver"]]
        for item in items:
            for order in ("orig", "flip"):
                a, b = item["option_a"], item["option_b"]
                if order == "flip":
                    a, b = b, a
                prompt = PROMPT_TEMPLATE.format(opt_a=a, opt_b=b)
                for sample in range(N_SAMPLES):
                    cid = f"grid|{cell['cell_id']}|{item['id']}|{order}|{sample}"
                    if cid in done:
                        continue
                    jobs.append({
                        "call_id": cid, "cell_id": cell["cell_id"],
                        "receiver": cell["receiver"], "author": cell["author"],
                        "arm": cell["arm"], "length": cell["length"],
                        "regen": cell["regen"], "provider": provider,
                        "model": model_id, "extra_body": extra,
                        "max_tokens": max_tok, "system": cell["system"],
                        "prompt": prompt, "item_id": item["id"],
                        "category": item["category"], "order": order,
                        "sample": sample,
                        "seed": stable_seed("grid", cell["cell_id"], item["id"],
                                            order, str(sample)),
                    })
    print(f"phase2: {len(jobs)} calls to run")
    if not jobs:
        return

    client = LLMClient(timeout=240.0)
    print(f"credits ok: ${client.ensure_credits(CREDIT_FLOOR):.2f}")
    write_lock = threading.Lock()
    counters = {"done": 0, "fail": 0}
    start = time.time()

    def worker(job: dict) -> None:
        rec = run_job(client, job)
        with write_lock:
            with OUT_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counters["done" if rec["ok"] else "fail"] += 1
            n = counters["done"] + counters["fail"]
            if n % 500 == 0 or n == len(jobs):
                rate = n / max(time.time() - start, 1e-9)
                print(f"  {n}/{len(jobs)} ({counters['fail']} failed) "
                      f"{rate:.1f}/s", flush=True)

    pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures = [pool.submit(worker, j) for j in jobs]
    for f in as_completed(futures):
        f.result()
    pool.shutdown()
    print(f"finished: {counters['done']} ok, {counters['fail']} failed; "
          f"credits ${client.credits_remaining():.2f}")


if __name__ == "__main__":
    main()
