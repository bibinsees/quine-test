"""Screening run: 200 candidate items x 6 core grid models x (2 orders x 3 samples).

Same task/prompt/hygiene as the pilot (generic system prompt, temp 1.0,
deterministic seeds, content-coded choices). Output: runs/screening.jsonl.
Selection into the final battery happens in select_items.py.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_client import LLMClient
from run_pilot import (PROMPT_TEMPLATE, SYSTEM_PROMPT, TEMPERATURE, parse_choice,
                       stable_seed)

ROOT = Path(__file__).resolve().parent.parent
ITEMS_PATH = ROOT / "battery" / "items_candidates.json"
OUT_PATH = ROOT / "runs" / "screening.jsonl"

N_SAMPLES = 3
CREDIT_FLOOR = 20.0  # hard-stop: never let screening take balance below this

MODELS = [
    ("deepseek-v3", "openrouter", "deepseek/deepseek-chat-v3-0324", None, 8),
    ("glm-4.5-air", "openrouter", "z-ai/glm-4.5-air", None, 512),
    ("grok-4.3", "openrouter", "x-ai/grok-4.3", None, 512),
    ("gpt5.2", "openrouter", "openai/gpt-5.2", {"reasoning": {"effort": "minimal"}}, 64),
    ("kimi-k2.5-nr", "openrouter", "moonshotai/kimi-k2.5", {"reasoning": {"enabled": False}}, 16),
    ("haiku-4.5", "openrouter", "anthropic/claude-haiku-4.5", None, 16),
]

MAX_WORKERS = 10


def call_id(short: str, item_id: str, order: str, sample: int) -> str:
    return f"scr|{short}|{item_id}|{order}|{sample}"


def build_jobs(items: list[dict]) -> list[dict]:
    jobs = []
    for short, provider, model_id, extra, max_tok in MODELS:
        for item in items:
            for order in ("orig", "flip"):
                a, b = item["option_a"], item["option_b"]
                if order == "flip":
                    a, b = b, a
                prompt = PROMPT_TEMPLATE.format(opt_a=a, opt_b=b)
                for sample in range(N_SAMPLES):
                    jobs.append({
                        "call_id": call_id(short, item["id"], order, sample),
                        "short": short, "provider": provider, "model": model_id,
                        "extra_body": extra, "max_tokens": max_tok,
                        "item_id": item["id"], "category": item["category"],
                        "order": order, "sample": sample,
                        "seed": stable_seed("scr", item["id"], order, str(sample)),
                        "prompt": prompt,
                    })
    return jobs


def run_job(client: LLMClient, job: dict) -> dict:
    record = {k: job[k] for k in
              ("call_id", "short", "provider", "model", "item_id", "category",
               "order", "sample", "seed")}
    record["ts"] = time.time()
    try:
        result = client.chat(
            provider=job["provider"], model=job["model"],
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": job["prompt"]}],
            temperature=TEMPERATURE, max_tokens=job["max_tokens"],
            seed=job["seed"], extra_body=job["extra_body"],
        )
        letter = parse_choice(result.text)
        choice = None
        if letter is not None:
            if job["order"] == "orig":
                choice = "option_a" if letter == "A" else "option_b"
            else:
                choice = "option_b" if letter == "A" else "option_a"
        record.update(ok=True, response=result.text[:2000], letter=letter,
                      choice=choice, latency_s=round(result.latency_s, 3))
    except Exception as exc:
        record.update(ok=False, error=str(exc)[:500], response=None,
                      letter=None, choice=None)
    return record


def main() -> None:
    items = json.loads(ITEMS_PATH.read_text(encoding="utf-8"))
    OUT_PATH.parent.mkdir(exist_ok=True)

    done: set[str] = set()
    if OUT_PATH.exists():
        for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                if rec.get("ok"):
                    done.add(rec["call_id"])
            except json.JSONDecodeError:
                continue

    all_jobs = build_jobs(items)
    jobs = [j for j in all_jobs if j["call_id"] not in done]
    print(f"jobs: {len(all_jobs)} total, {len(done)} cached, {len(jobs)} to run")
    if not jobs:
        print("nothing to do")
        return

    client = LLMClient(timeout=180.0)
    remaining = client.ensure_credits(CREDIT_FLOOR)
    print(f"credits ok: ${remaining:.2f} remaining (floor ${CREDIT_FLOOR:.2f})")

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
            if n % 200 == 0 or n == len(jobs):
                rate = n / max(time.time() - start, 1e-9)
                print(f"  {n}/{len(jobs)} ({counters['fail']} failed) "
                      f"{rate:.1f}/s eta {int((len(jobs) - n) / max(rate, 1e-9))}s",
                      flush=True)
        if counters["done"] % 1000 == 0 and counters["done"] > 0:
            client.ensure_credits(CREDIT_FLOOR)  # periodic hard-stop check

    pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures = [pool.submit(worker, j) for j in jobs]
    for f in as_completed(futures):
        f.result()
    pool.shutdown()

    print(f"finished: {counters['done']} ok, {counters['fail']} failed, "
          f"{int(time.time() - start)}s; credits ${client.credits_remaining():.2f}")


if __name__ == "__main__":
    main()
