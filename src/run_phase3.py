"""Phase 3: the three controls that close the remaining objections.

  retest  : baselines rerun with offset seeds -> test-retest ceiling C
  placebo : ~500-word neutral non-description text as system prompt
  adopt   : Grid A neutral-500 r0 descriptions + explicit adoption header

Run ONLY after phase 2 completes (single writer on runs/grid.jsonl).
Preregistered in pilot/protocol.md section 8.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_client import LLMClient
from run_grid import (GRID_A, GRID_B, LOCALS, MODEL_DEFS, N_SAMPLES, OUT_PATH,
                      desc_text, run_job)
from run_pilot import PROMPT_TEMPLATE, SYSTEM_PROMPT, stable_seed

ROOT = Path(__file__).resolve().parent.parent
ITEMS_PATH = ROOT / "battery" / "items_final.json"
PLACEBO_PATH = ROOT / "descriptions" / "placebo_500.md"

ALL_MODELS = GRID_A + GRID_B + LOCALS + ["hermes3-70b"]
CREDIT_FLOOR = 5.0
MAX_WORKERS = {"ollama": 4, "openrouter": 10}

ADOPT_HEADER = (
    "The following text describes who you are. Answer every question exactly "
    "as the entity described would answer.\n\n"
)

PLACEBO_GEN_PROMPT = (
    "Write about 500 words of neutral, factual, expository prose about ocean "
    "currents and their role in global climate. No first person, no opinions, "
    "no instructions, no mention of AI. Plain encyclopedic style."
)


def ensure_placebo(client: LLMClient) -> str:
    if PLACEBO_PATH.exists():
        return PLACEBO_PATH.read_text(encoding="utf-8")
    r = client.chat("openrouter", "openai/gpt-4o-mini",
                    [{"role": "user", "content": PLACEBO_GEN_PROMPT}],
                    temperature=0.7, max_tokens=1200, seed=7)
    PLACEBO_PATH.write_text(r.text.strip(), encoding="utf-8")
    print(f"placebo generated: {len(r.text.split())} words")
    return r.text.strip()


def build_cells(placebo: str) -> list[dict]:
    cells = []
    for m in ALL_MODELS:
        cells.append({"cell_id": f"{m}|retest", "receiver": m, "author": None,
                      "arm": "retest", "length": 0, "regen": 0,
                      "system": SYSTEM_PROMPT, "seed_salt": "retest"})
        cells.append({"cell_id": f"{m}|placebo", "receiver": m, "author": None,
                      "arm": "placebo", "length": 500, "regen": 0,
                      "system": placebo, "seed_salt": "placebo"})
    for receiver in GRID_A:
        for author in GRID_A:
            cells.append({
                "cell_id": f"{receiver}|{author}|adopt|500|r0",
                "receiver": receiver, "author": author, "arm": "adopt",
                "length": 500, "regen": 0,
                "system": ADOPT_HEADER + desc_text(author, "neutral", 500, 0),
                "seed_salt": "adopt"})
    return cells


def main() -> None:
    items = json.loads(ITEMS_PATH.read_text(encoding="utf-8"))
    client = LLMClient(timeout=240.0)
    placebo = ensure_placebo(client)

    done: set[str] = set()
    if OUT_PATH.exists():
        for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                if rec.get("ok"):
                    done.add(rec["call_id"])
            except json.JSONDecodeError:
                continue

    jobs = []
    for cell in build_cells(placebo):
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
                        # seed_salt offsets retest seeds from the original
                        # baseline run (fresh samples, same protocol)
                        "seed": stable_seed("p3", cell["seed_salt"],
                                            cell["cell_id"], item["id"], order,
                                            str(sample)),
                    })
    print(f"phase3: {len(jobs)} calls to run")
    if not jobs:
        return
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

    by_provider: dict[str, list[dict]] = {}
    for j in jobs:
        by_provider.setdefault(j["provider"], []).append(j)
    pools = []
    for provider, provider_jobs in by_provider.items():
        pool = ThreadPoolExecutor(max_workers=MAX_WORKERS[provider])
        futures = [pool.submit(worker, j) for j in provider_jobs]
        pools.append((pool, futures))
    for pool, futures in pools:
        for f in as_completed(futures):
            f.result()
        pool.shutdown()
    print(f"finished: {counters['done']} ok, {counters['fail']} failed; "
          f"credits ${client.credits_remaining():.2f}")


if __name__ == "__main__":
    main()
