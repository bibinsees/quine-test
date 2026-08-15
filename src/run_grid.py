"""MAIN TRANSFER GRID.

Cells (preregistered in pilot/protocol.md section 7 before launch):

  Grid A (deepseek-v3, glm-4.5-air, grok-4.3) - API:
    conditions: neutral-100/500/2000 + framed-500, regen r0
    F_self  : each receiver + its own description        (3 x 4 cells)
    F_cross : all 6 directed pairs                       (6 x 4 cells)
    robustness: neutral-500 r1, self + cross             (9 cells)
    baseline B: reused from runs/screening.jsonl (same items, generic prompt)

  Grid B (gpt5.2, kimi-k2.5-nr, haiku-4.5) - API, frontier:
    conditions: neutral-500 + framed-500, r0 only
    F_self + F_cross                                     (9 x 2 cells)
    baseline B: reused from screening.

  Local trio (llama3.3-70b, gemma4-31b, mistral32-local) - free replication:
    conditions: neutral-100/500/2000 + framed-500, r0
    F_self + F_cross + generic-prompt baseline           (9 x 4 + 3 cells)

Sampling: 160 final items x 2 orders x 2 samples = 640 calls per cell.
Specificity control needs no extra cells: within each trio, fidelity(A, B+descA)
vs fidelity(A, B+descC) reuses the cross cells.
Third-person and paraphrase cells run in a later phase (separate script).
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_client import LLMClient
from run_pilot import PROMPT_TEMPLATE, SYSTEM_PROMPT, parse_choice, stable_seed

ROOT = Path(__file__).resolve().parent.parent
ITEMS_PATH = ROOT / "battery" / "items_final.json"
DESC_DIR = ROOT / "descriptions"
OUT_PATH = ROOT / "runs" / "grid.jsonl"

N_SAMPLES = 2            # per order -> 4 per item per cell
TEMPERATURE = 1.0
CREDIT_FLOOR = 4.0  # lowered for phase 4 (user can top up; resume from cache)

MODEL_DEFS = {
    "deepseek-v3": ("openrouter", "deepseek/deepseek-chat-v3-0324", None, 8),
    "glm-4.5-air": ("openrouter", "z-ai/glm-4.5-air", None, 512),
    "grok-4.3": ("openrouter", "x-ai/grok-4.3", None, 512),
    "gpt5.2": ("openrouter", "openai/gpt-5.2", {"reasoning": {"effort": "minimal"}}, 64),
    "kimi-k2.5-nr": ("openrouter", "moonshotai/kimi-k2.5", {"reasoning": {"enabled": False}}, 16),
    "haiku-4.5": ("openrouter", "anthropic/claude-haiku-4.5", None, 16),
    "llama3.3-70b": ("ollama", "llama3.3:70b", None, 512),
    "hermes3-70b": ("ollama", "hermes3:70b", None, 512),
    "gemma4-31b": ("ollama", "gemma4:31b", {"think": False}, 512),
    "mistral32-local": ("ollama", "mistral-small3.2:24b", None, 512),
}

GRID_A = ["deepseek-v3", "glm-4.5-air", "grok-4.3"]
GRID_B = ["gpt5.2", "kimi-k2.5-nr", "haiku-4.5"]
LOCALS = ["llama3.3-70b", "gemma4-31b", "mistral32-local"]

FULL_CONDITIONS = [("neutral", 100), ("neutral", 500), ("neutral", 2000),
                   ("framed", 500)]
CORE_CONDITIONS = [("neutral", 500), ("framed", 500)]
# framed 100/2000 reactivated by user decision Sat evening (after phase-1
# results); Grid A + locals + same-base pair, Grid B excluded (cost).
FRAMED_EXT = [("framed", 100), ("framed", 2000)]

MAX_WORKERS = {"ollama": 4, "openrouter": 12}


def desc_text(author: str, arm: str, length: int, regen: int) -> str:
    path = DESC_DIR / f"desc_{author}_{arm}_{length}_r{regen}.md"
    return path.read_text(encoding="utf-8").strip()


def build_cells() -> list[dict]:
    """One dict per cell: receiver, author (None = generic baseline), arm,
    length, regen."""
    cells = []

    def trio_cells(trio, conditions, regens=(0,)):
        out = []
        for receiver in trio:
            for author in trio:  # author == receiver -> F_self
                for arm, length in conditions:
                    for regen in regens:
                        out.append({"receiver": receiver, "author": author,
                                    "arm": arm, "length": length, "regen": regen})
        return out

    cells += trio_cells(GRID_A, FULL_CONDITIONS)
    cells += trio_cells(GRID_A, [("neutral", 500)], regens=(1,))  # robustness
    cells += trio_cells(GRID_B, FULL_CONDITIONS)  # extended per user 2026-08-15
    cells += trio_cells(LOCALS, FULL_CONDITIONS)
    for receiver in LOCALS:  # locals need explicit generic baselines
        cells.append({"receiver": receiver, "author": None,
                      "arm": "baseline", "length": 0, "regen": 0})
    cells += trio_cells(GRID_A, FRAMED_EXT)
    cells += trio_cells(LOCALS, FRAMED_EXT)
    # same-base transfer test: hermes3:70b and llama3.3:70b share Llama base
    # lineage but differ in post-training. F_cross between them vs F_cross to
    # unrelated models isolates whether the residual is weight-bound.
    for arm, length in FULL_CONDITIONS + FRAMED_EXT:
        cells.append({"receiver": "hermes3-70b", "author": "hermes3-70b",
                      "arm": arm, "length": length, "regen": 0})
        cells.append({"receiver": "hermes3-70b", "author": "llama3.3-70b",
                      "arm": arm, "length": length, "regen": 0})
        cells.append({"receiver": "llama3.3-70b", "author": "hermes3-70b",
                      "arm": arm, "length": length, "regen": 0})
    cells.append({"receiver": "hermes3-70b", "author": None,
                  "arm": "baseline", "length": 0, "regen": 0})
    return cells


def build_jobs(items: list[dict]) -> list[dict]:
    jobs = []
    for cell in build_cells():
        receiver = cell["receiver"]
        provider, model_id, extra, max_tok = MODEL_DEFS[receiver]
        if cell["author"] is None:
            system = SYSTEM_PROMPT
            cell_id = f"{receiver}|baseline"
        else:
            system = desc_text(cell["author"], cell["arm"], cell["length"],
                               cell["regen"])
            cell_id = (f"{receiver}|{cell['author']}|{cell['arm']}|"
                       f"{cell['length']}|r{cell['regen']}")
        for item in items:
            for order in ("orig", "flip"):
                a, b = item["option_a"], item["option_b"]
                if order == "flip":
                    a, b = b, a
                prompt = PROMPT_TEMPLATE.format(opt_a=a, opt_b=b)
                for sample in range(N_SAMPLES):
                    jobs.append({
                        "call_id": f"grid|{cell_id}|{item['id']}|{order}|{sample}",
                        "cell_id": cell_id, "receiver": receiver,
                        "author": cell["author"], "arm": cell["arm"],
                        "length": cell["length"], "regen": cell["regen"],
                        "provider": provider, "model": model_id,
                        "extra_body": extra, "max_tokens": max_tok,
                        "system": system, "prompt": prompt,
                        "item_id": item["id"], "category": item["category"],
                        "order": order, "sample": sample,
                        "seed": stable_seed("grid", cell_id, item["id"], order,
                                            str(sample)),
                    })
    return jobs


def run_job(client: LLMClient, job: dict) -> dict:
    record = {k: job[k] for k in
              ("call_id", "cell_id", "receiver", "author", "arm", "length",
               "regen", "provider", "model", "item_id", "category", "order",
               "sample", "seed")}
    record["ts"] = time.time()
    try:
        result = client.chat(
            provider=job["provider"], model=job["model"],
            messages=[{"role": "system", "content": job["system"]},
                      {"role": "user", "content": job["prompt"]}],
            temperature=TEMPERATURE, max_tokens=job["max_tokens"],
            seed=job["seed"], extra_body=job["extra_body"],
            cache_system=(job["provider"] == "openrouter"),
        )
        letter = parse_choice(result.text)
        choice = None
        if letter is not None:
            if job["order"] == "orig":
                choice = "option_a" if letter == "A" else "option_b"
            else:
                choice = "option_b" if letter == "A" else "option_a"
        record.update(ok=True, response=result.text[:500], letter=letter,
                      choice=choice, latency_s=round(result.latency_s, 3))
    except Exception as exc:
        record.update(ok=False, error=str(exc)[:400], response=None,
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
    n_cells = len({j["cell_id"] for j in all_jobs})
    print(f"grid: {n_cells} cells, {len(all_jobs)} calls total, "
          f"{len(done)} cached, {len(jobs)} to run")
    if not jobs:
        print("nothing to do")
        return

    client = LLMClient(timeout=240.0)
    remaining = client.ensure_credits(CREDIT_FLOOR)
    print(f"credits ok: ${remaining:.2f} (floor ${CREDIT_FLOOR:.2f})")

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
                      f"{rate:.1f}/s eta {int((len(jobs) - n) / max(rate, 1e-9))}s",
                      flush=True)
        if counters["done"] % 2000 == 0 and counters["done"] > 0:
            client.ensure_credits(CREDIT_FLOOR)

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

    print(f"finished: {counters['done']} ok, {counters['fail']} failed, "
          f"{int(time.time() - start)}s; credits ${client.credits_remaining():.2f}")


if __name__ == "__main__":
    main()
