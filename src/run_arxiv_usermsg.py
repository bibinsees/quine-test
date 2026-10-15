"""arXiv revision: placement control. Same texts as before, placed in the USER
message (before the question) instead of the system prompt; system prompt is
the generic assistant prompt used for the baseline B. Grid A, 9 cells each:

  neutral_user-500   the author's neutral-500 r0 self-description (sprint text)
  oracle_user-110    the author's 110-item answer key (identical text to oracle-110)

Writes runs/arxiv_usermsg.jsonl (new file).
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_client import LLMClient
from run_arxiv_positive_controls import MAX_TOKENS_OVERRIDE, build_system_prompts, load_jsonl
from run_grid import GRID_A, MODEL_DEFS, N_SAMPLES, desc_text, run_job
from run_pilot import PROMPT_TEMPLATE, SYSTEM_PROMPT, stable_seed

ROOT = Path(__file__).resolve().parent.parent
ITEMS_FINAL = ROOT / "battery" / "items_final.json"
OUT_PATH = ROOT / "runs" / "arxiv_usermsg.jsonl"
CREDIT_FLOOR = 2.0
MAX_WORKERS = 32


def main() -> None:
    final = json.loads(ITEMS_FINAL.read_text(encoding="utf-8"))
    client = LLMClient(timeout=240.0)
    print(f"credits ok: ${client.ensure_credits(CREDIT_FLOOR):.2f}", flush=True)
    prompts = build_system_prompts(client)  # existing files only, no new calls
    texts = {}
    for author in GRID_A:
        texts[(author, "neutral_user", 500)] = desc_text(author, "neutral", 500, 0)
        texts[(author, "oracle_user", 110)] = prompts[(author, "oracle", 110)]

    done = {r["call_id"] for r in load_jsonl(OUT_PATH) if r.get("ok")} if OUT_PATH.exists() else set()
    jobs = []
    for (author, arm, length), text in texts.items():
        for receiver in GRID_A:
            provider, model_id, extra, max_tok = MODEL_DEFS[receiver]
            max_tok = MAX_TOKENS_OVERRIDE.get(receiver, max_tok)
            cell_id = f"{receiver}|{author}|{arm}|{length}|r0"
            for item in final:
                for order in ("orig", "flip"):
                    a, b = item["option_a"], item["option_b"]
                    if order == "flip":
                        a, b = b, a
                    for sample in range(N_SAMPLES):
                        cid = f"grid|{cell_id}|{item['id']}|{order}|{sample}"
                        if cid in done:
                            continue
                        jobs.append({
                            "call_id": cid, "cell_id": cell_id, "receiver": receiver,
                            "author": author, "arm": arm, "length": length, "regen": 0,
                            "provider": provider, "model": model_id, "extra_body": extra,
                            "max_tokens": max_tok, "system": SYSTEM_PROMPT,
                            "prompt": text.strip() + "\n\n" + PROMPT_TEMPLATE.format(opt_a=a, opt_b=b),
                            "item_id": item["id"], "category": item["category"],
                            "order": order, "sample": sample,
                            "seed": stable_seed("grid", cell_id, item["id"], order, str(sample)),
                        })
    print(f"user-message placement: {len({j['cell_id'] for j in jobs})} cells, {len(jobs)} calls",
          flush=True)
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
                print(f"  {n}/{len(jobs)} ({counters['fail']} failed)", flush=True)
            if counters["done"] % 1000 == 0 and counters["done"] > 0:
                client.ensure_credits(CREDIT_FLOOR)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for f in as_completed([pool.submit(worker, j) for j in jobs]):
            f.result()
    print(f"finished: {counters['done']} ok, {counters['fail']} failed, "
          f"{int(time.time() - start)}s; credits ${client.credits_remaining():.2f}", flush=True)


if __name__ == "__main__":
    main()
