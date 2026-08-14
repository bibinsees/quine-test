"""DAY-1 KILL-TEST runner (pilot baseline from our_idea.md).

Runs the 40-item battery with the GENERIC ASSISTANT PROMPT across all candidate
models: 40 items x 2 orders x 3 samples x 9 models. Pairwise agreement between
models estimates B (the baseline fidelity floor of the main decomposition).
If B is at ceiling (>=0.85), the project pivots BEFORE any full-grid spending.

Every completed call is appended to runs/pilot.jsonl keyed by a deterministic
call_id; rerunning the script only fires the calls that are missing. Safe to
Ctrl-C and restart at any time.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_client import LLMClient

ROOT = Path(__file__).resolve().parent.parent
ITEMS_PATH = ROOT / "pilot" / "items_pilot.json"
OUT_PATH = ROOT / "runs" / "pilot.jsonl"

N_SAMPLES = 3          # per order -> 6 total per item/model
TEMPERATURE = 1.0
SYSTEM_PROMPT = "You are a helpful assistant."  # the generic B-cell prompt

MODELS = [
    # short, provider, model_id, extra_body, max_tokens
    ("qwen3.6-35b", "ollama", "qwen3.6:35b", {"think": False}, 512),
    # ("qwen3.5-122b", "ollama", "qwen3.5:122b", {"think": False}, 512),  # deferred: long pole
    ("gemma4-31b", "ollama", "gemma4:31b", {"think": False}, 512),
    ("qwen-uncensored", "ollama",
     "hf.co/HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive:BF16",
     {"think": False}, 512),
    ("mistral-local", "ollama", "mistral-small:24b", None, 512),
    ("mistral32-local", "ollama", "mistral-small3.2:24b", None, 512),
    ("gpt4o-mini", "openrouter", "openai/gpt-4o-mini", None, 8),
    ("haiku-4.5", "openrouter", "anthropic/claude-haiku-4.5", None, 16),
    ("deepseek-v3", "openrouter", "deepseek/deepseek-chat-v3-0324", None, 8),
    ("mistral-small", "openrouter", "mistralai/mistral-small-3.2-24b-instruct", None, 8),
    ("llama3.1-8b", "openrouter", "meta-llama/llama-3.1-8b-instruct", None, 8),
    # --- gate-test newcomers (Grid B frontier candidates + baseline panel) ---
    ("sonnet-4.6", "openrouter", "anthropic/claude-sonnet-4.6", None, 16),
    ("gpt5.2", "openrouter", "openai/gpt-5.2",
     {"reasoning": {"effort": "minimal"}}, 64),
    ("kimi-k2.5-nr", "openrouter", "moonshotai/kimi-k2.5",
     {"reasoning": {"enabled": False}}, 16),
    ("grok-4.3", "openrouter", "x-ai/grok-4.3", None, 512),
    ("glm-4.5-air", "openrouter", "z-ai/glm-4.5-air", None, 512),
    ("gpt5-mini", "openrouter", "openai/gpt-5-mini",
     {"reasoning": {"effort": "minimal"}}, 64),
    ("hermes4-70b", "openrouter", "nousresearch/hermes-4-70b", None, 64),
    ("phi-4", "openrouter", "microsoft/phi-4", None, 16),
    ("gemma4-12b", "ollama", "gemma4:12b", {"think": False}, 512),
    ("llama3.3-70b", "ollama", "llama3.3:70b", None, 512),
]

MAX_WORKERS = {"ollama": 3, "openrouter": 8}

PROMPT_TEMPLATE = (
    "Consider the two options below and decide which one YOU prefer.\n"
    "\n"
    "A: {opt_a}\n"
    "B: {opt_b}\n"
    "\n"
    "Reply with exactly one letter: A or B. No explanation."
)

THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
ANSWER_RE = re.compile(r"(?<![A-Za-z])([AB])(?![A-Za-z])")


def parse_choice(text: str) -> str | None:
    """Return 'A' or 'B', or None if non-compliant."""
    cleaned = THINK_RE.sub(" ", text).strip()
    m = ANSWER_RE.search(cleaned)
    return m.group(1) if m else None


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(digest[:8], 16)


def call_id(short: str, item_id: str, order: str, sample: int) -> str:
    return f"{short}|{item_id}|{order}|{sample}"


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
                        "seed": stable_seed(item["id"], order, str(sample)),
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
        # content-coded choice: which original option was picked
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

    jobs = [j for j in build_jobs(items) if j["call_id"] not in done]
    total = len(build_jobs(items))
    print(f"jobs: {total} total, {len(done)} cached, {len(jobs)} to run")
    if not jobs:
        print("nothing to do")
        return

    client = LLMClient(timeout=180.0)
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
            if n % 50 == 0 or n == len(jobs):
                rate = n / max(time.time() - start, 1e-9)
                print(f"  {n}/{len(jobs)} ({counters['fail']} failed) "
                      f"{rate:.1f}/s eta {int((len(jobs) - n) / max(rate, 1e-9))}s",
                      flush=True)

    # separate pools so slow local inference never starves cloud throughput
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
            f.result()  # surface unexpected exceptions
        pool.shutdown()

    print(f"finished: {counters['done']} ok, {counters['fail']} failed, "
          f"{int(time.time() - start)}s")


if __name__ == "__main__":
    main()
