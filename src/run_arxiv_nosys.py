"""arXiv revision: no-system-prompt baseline. Every model answers the 110
battery items with ONLY the question in the user message (no system prompt at
all), 2 orders x 2 samples. Compared afterwards with the generic-prompt
reference profile to check that the seven-word generic prompt is inert.
Writes runs/arxiv_nosys.jsonl (new file)."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_client import LLMClient
from run_arxiv_positive_controls import load_jsonl
from run_grid import MODEL_DEFS, N_SAMPLES, TEMPERATURE
from run_pilot import PROMPT_TEMPLATE, parse_choice, stable_seed

ROOT = Path(__file__).resolve().parent.parent
ITEMS_FINAL = ROOT / "battery" / "items_final.json"
OUT_PATH = ROOT / "runs" / "arxiv_nosys.jsonl"
WORKERS = {"openrouter": 24, "ollama": 4}


def run_job(client: LLMClient, job: dict) -> dict:
    rec = {k: job[k] for k in ("call_id", "cell_id", "receiver", "provider", "model",
                                "item_id", "category", "order", "sample", "seed")}
    rec.update(author=None, arm="nosys", length=0, regen=0, ts=time.time())
    try:
        r = client.chat(provider=job["provider"], model=job["model"],
                        messages=[{"role": "user", "content": job["prompt"]}],
                        temperature=TEMPERATURE, max_tokens=job["max_tokens"],
                        seed=job["seed"], extra_body=job["extra_body"])
        letter = parse_choice(r.text)
        choice = None
        if letter is not None:
            choice = ("option_a" if letter == "A" else "option_b") if job["order"] == "orig" \
                else ("option_b" if letter == "A" else "option_a")
        rec.update(ok=True, response=r.text[:500], letter=letter, choice=choice,
                   latency_s=round(r.latency_s, 3))
    except Exception as exc:
        rec.update(ok=False, error=str(exc)[:400], response=None, letter=None, choice=None)
    return rec


def main() -> None:
    items = json.loads(ITEMS_FINAL.read_text(encoding="utf-8"))
    done = {r["call_id"] for r in load_jsonl(OUT_PATH) if r.get("ok")} if OUT_PATH.exists() else set()
    jobs = []
    for short, (provider, model_id, extra, max_tok) in MODEL_DEFS.items():
        cell_id = f"{short}|nosys"
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
                                 "provider": provider, "model": model_id, "extra_body": extra,
                                 "max_tokens": max_tok,
                                 "prompt": PROMPT_TEMPLATE.format(opt_a=a, opt_b=b),
                                 "item_id": item["id"], "category": item["category"],
                                 "order": order, "sample": sample,
                                 "seed": stable_seed("grid", cell_id, item["id"], order, str(sample))})
    print(f"no-system-prompt baseline: {len({j['cell_id'] for j in jobs})} cells, {len(jobs)} calls",
          flush=True)
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
            if n % 500 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} ({counters['fail']} failed)", flush=True)

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
    print(f"finished: {counters['done']} ok, {counters['fail']} failed, {int(time.time() - start)}s",
          flush=True)


if __name__ == "__main__":
    main()
