"""arXiv revision: rerun the 4 hermes3-authored framed-500/2000 cells with the
r1 descriptions. The r0 texts at these lengths were refusals ("I will not write
a description of myself...", 57 and 46 words), noticed only during the arXiv
revision; the sprint analysis ran these four cells on the refusal text. The
revision substitutes the second independent draw (r1, 278 and 555 words) and
reports the refusal itself as a finding. Writes to a NEW log so the sprint
data stay untouched.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_client import LLMClient
from run_grid import MODEL_DEFS, N_SAMPLES, TEMPERATURE, desc_text, run_job
from run_pilot import PROMPT_TEMPLATE, stable_seed

ROOT = Path(__file__).resolve().parent.parent
ITEMS_PATH = ROOT / "battery" / "items_final.json"
OUT_PATH = ROOT / "runs" / "arxiv_hermes_r1.jsonl"

CELLS = [  # (receiver, author, arm, length, regen)
    ("hermes3-70b", "hermes3-70b", "framed", 500, 1),
    ("hermes3-70b", "hermes3-70b", "framed", 2000, 1),
    ("llama3.3-70b", "hermes3-70b", "framed", 500, 1),
    ("llama3.3-70b", "hermes3-70b", "framed", 2000, 1),
]


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
                pass

    jobs = []
    for receiver, author, arm, length, regen in CELLS:
        provider, model_id, extra, max_tok = MODEL_DEFS[receiver]
        system = desc_text(author, arm, length, regen)
        assert len(system.split()) > 150, f"unexpectedly short desc for {author} {arm} {length} r{regen}"
        cell_id = f"{receiver}|{author}|{arm}|{length}|r{regen}"
        for item in items:
            for order in ("orig", "flip"):
                a, b = item["option_a"], item["option_b"]
                if order == "flip":
                    a, b = b, a
                prompt = PROMPT_TEMPLATE.format(opt_a=a, opt_b=b)
                for sample in range(N_SAMPLES):
                    cid = f"grid|{cell_id}|{item['id']}|{order}|{sample}"
                    if cid in done:
                        continue
                    jobs.append({
                        "call_id": cid, "cell_id": cell_id, "receiver": receiver,
                        "author": author, "arm": arm, "length": length,
                        "regen": regen, "provider": provider, "model": model_id,
                        "extra_body": extra, "max_tokens": max_tok,
                        "system": system, "prompt": prompt,
                        "item_id": item["id"], "category": item["category"],
                        "order": order, "sample": sample,
                        "seed": stable_seed("grid", cell_id, item["id"], order,
                                            str(sample)),
                    })
    print(f"hermes3 r1 rerun: {len(jobs)} calls to run")
    if not jobs:
        return

    client = LLMClient(timeout=240.0)
    lock = threading.Lock()
    counters = {"done": 0, "fail": 0}
    start = time.time()

    def worker(job):
        rec = run_job(client, job)
        with lock:
            with OUT_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counters["done" if rec["ok"] else "fail"] += 1
            n = counters["done"] + counters["fail"]
            if n % 200 == 0 or n == len(jobs):
                rate = n / max(time.time() - start, 1e-9)
                print(f"  {n}/{len(jobs)} ({counters['fail']} failed) {rate:.1f}/s",
                      flush=True)

    pool = ThreadPoolExecutor(max_workers=4)
    futures = [pool.submit(worker, j) for j in jobs]
    for f in as_completed(futures):
        f.result()
    pool.shutdown()
    print(f"finished: {counters['done']} ok, {counters['fail']} failed")


if __name__ == "__main__":
    main()
