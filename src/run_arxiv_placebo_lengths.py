"""Unrelated-prose ablation: every model runs the battery with the mixed
placebo text (100, 500, 2000 words) as system prompt. One run per model per
length serves both the Table-2 scoring (against each author of its grid) and
the Table-8 scoring (against itself). 30 cells. Writes
runs/arxiv_placebo_lengths.jsonl (new file). Resumable."""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_client import LLMClient
from run_arxiv_positive_controls import MAX_TOKENS_OVERRIDE, load_jsonl
from run_grid import MODEL_DEFS, N_SAMPLES, run_job
from run_pilot import PROMPT_TEMPLATE, stable_seed

ROOT = Path(__file__).resolve().parent.parent
ITEMS_FINAL = ROOT / "battery" / "items_final.json"
DESC_DIR = ROOT / "descriptions"
OUT_PATH = ROOT / "runs" / "arxiv_placebo_lengths.jsonl"
CREDIT_FLOOR = 2.0
WORKERS = {"openrouter": 24, "ollama": 4}


def main() -> None:
    lengths = [int(x) for x in sys.argv[1:]] or [100, 500, 2000]
    items = json.loads(ITEMS_FINAL.read_text(encoding="utf-8"))
    done = {r["call_id"] for r in load_jsonl(OUT_PATH) if r.get("ok")} if OUT_PATH.exists() else set()
    jobs = []
    for n in lengths:
        text = (DESC_DIR / f"placebo_mixed_{n}.md").read_text(encoding="utf-8").strip()
        for short, (provider, model_id, extra, max_tok) in MODEL_DEFS.items():
            max_tok = MAX_TOKENS_OVERRIDE.get(short, max_tok)
            cell_id = f"{short}|placebo_mixed|{n}"
            for item in items:
                for order in ("orig", "flip"):
                    a, b = item["option_a"], item["option_b"]
                    if order == "flip":
                        a, b = b, a
                    for sample in range(N_SAMPLES):
                        cid = f"grid|{cell_id}|{item['id']}|{order}|{sample}"
                        if cid in done:
                            continue
                        jobs.append({"call_id": cid, "cell_id": cell_id, "receiver": short,
                                     "author": None, "arm": "placebo_mixed", "length": n, "regen": 0,
                                     "provider": provider, "model": model_id, "extra_body": extra,
                                     "max_tokens": max_tok, "system": text,
                                     "prompt": PROMPT_TEMPLATE.format(opt_a=a, opt_b=b),
                                     "item_id": item["id"], "category": item["category"],
                                     "order": order, "sample": sample,
                                     "seed": stable_seed("grid", cell_id, item["id"], order, str(sample))})
    print(f"placebo lengths {lengths}: {len({j['cell_id'] for j in jobs})} cells, {len(jobs)} calls",
          flush=True)
    if not jobs:
        return
    client = LLMClient(timeout=240.0)
    print(f"credits ok: ${client.ensure_credits(CREDIT_FLOOR):.2f}", flush=True)
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
            if n % 1000 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} ({counters['fail']} failed) {n/(time.time()-start):.1f}/s",
                      flush=True)
            if counters["done"] % 2000 == 0 and counters["done"] > 0:
                client.ensure_credits(CREDIT_FLOOR)

    pools = []
    for provider in ("openrouter", "ollama"):
        pj = [j for j in jobs if j["provider"] == provider]
        if pj:
            pool = ThreadPoolExecutor(max_workers=WORKERS[provider])
            pools.append((pool, [pool.submit(worker, j) for j in pj]))
    for pool, futures in pools:
        for f in as_completed(futures):
            f.result()
        pool.shutdown()
    print(f"finished: {counters['done']} ok, {counters['fail']} failed, {int(time.time() - start)}s; "
          f"credits ${client.credits_remaining():.2f}", flush=True)


if __name__ == "__main__":
    main()
