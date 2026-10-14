"""arXiv revision: demonstration dose curve. Same format as demonstrations-30
(run_arxiv_positive_controls.py) with k = 60 and k = 90 held-out items; the
item sets are nested (the 30 used before are contained in the 60, the 60 in
the 90 = all held-out items). Grid A, all 9 receiver x author cells per k.
Writes to runs/arxiv_demos_dose.jsonl (new file).
"""

from __future__ import annotations

import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_client import LLMClient
from run_arxiv_positive_controls import (DEMO_PROMPT, FEWSHOT_SEED, K_FEWSHOT,
                                         MAX_TOKENS_OVERRIDE, demo_lines, load_jsonl,
                                         majority_choices)
from run_grid import GRID_A, MODEL_DEFS, N_SAMPLES, run_job
from run_pilot import PROMPT_TEMPLATE, stable_seed

ROOT = Path(__file__).resolve().parent.parent
ITEMS_FINAL = ROOT / "battery" / "items_final.json"
ITEMS_CAND = ROOT / "battery" / "items_candidates.json"
SCREEN = ROOT / "runs" / "screening.jsonl"
OUT_PATH = ROOT / "runs" / "arxiv_demos_dose.jsonl"

KS = [60, 90]
CREDIT_FLOOR = 2.0
MAX_WORKERS = 32


def nested_sets(heldout_ids: list[str]) -> dict[int, list[str]]:
    rng = random.Random(FEWSHOT_SEED)
    k30 = sorted(rng.sample(heldout_ids, K_FEWSHOT))          # identical to demonstrations-30
    rest = [i for i in heldout_ids if i not in k30]
    k60 = sorted(k30 + random.Random(FEWSHOT_SEED + 1).sample(rest, 30))
    return {30: k30, 60: k60, 90: sorted(heldout_ids)}


def main() -> None:
    final = json.loads(ITEMS_FINAL.read_text(encoding="utf-8"))
    cand = {it["id"]: it for it in json.loads(ITEMS_CAND.read_text(encoding="utf-8"))}
    final_ids = {it["id"] for it in final}
    heldout_ids = sorted(set(cand) - final_ids)
    sets = nested_sets(heldout_ids)
    assert set(sets[30]) <= set(sets[60]) <= set(sets[90])
    screen = load_jsonl(SCREEN)

    done = set()
    if OUT_PATH.exists():
        for rec in load_jsonl(OUT_PATH):
            if rec.get("ok"):
                done.add(rec["call_id"])

    jobs = []
    for k in KS:
        for receiver in ["glm-4.5-air", "deepseek-v3", "grok-4.3"]:
            provider, model_id, extra, max_tok = MODEL_DEFS[receiver]
            max_tok = MAX_TOKENS_OVERRIDE.get(receiver, max_tok)
            for author in GRID_A:
                maj = majority_choices(screen, author, set(heldout_ids), "short", "item_id")
                lines = demo_lines(cand, maj, sets[k])
                system = DEMO_PROMPT.format(n=len(lines), demos="\n".join(lines))
                cell_id = f"{receiver}|{author}|fewshot|{k}|r0"
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
                                "author": author, "arm": "fewshot", "length": k, "regen": 0,
                                "provider": provider, "model": model_id, "extra_body": extra,
                                "max_tokens": max_tok, "system": system,
                                "prompt": PROMPT_TEMPLATE.format(opt_a=a, opt_b=b),
                                "item_id": item["id"], "category": item["category"],
                                "order": order, "sample": sample,
                                "seed": stable_seed("grid", cell_id, item["id"], order,
                                                    str(sample)),
                            })
    n_cells = len({j["cell_id"] for j in jobs})
    print(f"demo dose: {n_cells} cells, {len(jobs)} calls to run ({len(done)} cached)")
    if not jobs:
        return
    client = LLMClient(timeout=240.0)
    print(f"credits ok: ${client.ensure_credits(CREDIT_FLOOR):.2f}")
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
                rate = n / max(time.time() - start, 1e-9)
                print(f"  {n}/{len(jobs)} ({counters['fail']} failed) {rate:.1f}/s", flush=True)
            if counters["done"] % 1000 == 0 and counters["done"] > 0:
                client.ensure_credits(CREDIT_FLOOR)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for f in as_completed([pool.submit(worker, j) for j in jobs]):
            f.result()
    print(f"finished: {counters['done']} ok, {counters['fail']} failed, "
          f"{int(time.time() - start)}s; credits ${client.credits_remaining():.2f}")


if __name__ == "__main__":
    main()
