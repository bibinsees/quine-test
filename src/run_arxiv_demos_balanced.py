"""Position-balanced demonstrations (pilot/protocol_arxiv_demos_balanced.md).

  demos_bal|55   two folds of the 110 battery items; demonstrations = author's untied
                 majority choices on the other fold, option order randomised per line;
                 test on the fold's items. Both folds share one cell_id (440 calls, 110 items).
  fewshot_bal|30 30 untied held-out items per author, option order randomised.

Grid A, 9 cells per condition. Writes runs/arxiv_demos_balanced.jsonl (new). Resumable."""

from __future__ import annotations

import json
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_client import LLMClient
from run_arxiv_positive_controls import DEMO_PROMPT, MAX_TOKENS_OVERRIDE, load_jsonl
from run_grid import GRID_A, MODEL_DEFS, N_SAMPLES, run_job
from run_pilot import PROMPT_TEMPLATE, stable_seed

ROOT = Path(__file__).resolve().parent.parent
ITEMS_FINAL = ROOT / "battery" / "items_final.json"
ITEMS_CAND = ROOT / "battery" / "items_candidates.json"
SCREEN = ROOT / "runs" / "screening.jsonl"
OUT_PATH = ROOT / "runs" / "arxiv_demos_balanced.jsonl"
SEED = 20261019
CREDIT_FLOOR = 2.0
MAX_WORKERS = 32


def choice_counts(rows, short, item_ids):
    c = defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get("short") == short and r.get("ok") and r.get("choice") and r["item_id"] in item_ids:
            c[r["item_id"]][0 if r["choice"] == "option_a" else 1] += 1
    return c


def untied_majority(counts):
    """item -> 'option_a'/'option_b' for items with a strict majority only."""
    return {i: ("option_a" if a > b else "option_b") for i, (a, b) in counts.items() if a != b}


def balanced_lines(items_by_id, maj, ids, rng):
    lines = []
    for item_id in ids:
        it = items_by_id[item_id]
        chosen = it["option_a"] if maj[item_id] == "option_a" else it["option_b"]
        other = it["option_b"] if maj[item_id] == "option_a" else it["option_a"]
        first, second = (chosen, other) if rng.random() < 0.5 else (other, chosen)
        lines.append(f'- Given "{first}" vs "{second}", you chose: "{chosen}"')
    return lines


def folds(items, rng):
    by_cat = defaultdict(list)
    for it in items:
        by_cat[it["category"]].append(it["id"])
    f = [set(), set()]
    for cat, ids in sorted(by_cat.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        half = len(ids) // 2
        f[0].update(ids[:half])
        f[1].update(ids[half:])
    return f


def main() -> None:
    conds = sys.argv[1:] or ["55", "30"]
    final = json.loads(ITEMS_FINAL.read_text(encoding="utf-8"))
    cand = {it["id"]: it for it in json.loads(ITEMS_CAND.read_text(encoding="utf-8"))}
    final_by_id = {it["id"]: it for it in final}
    final_ids = set(final_by_id)
    heldout_ids = sorted(set(cand) - final_ids)
    screen = load_jsonl(SCREEN)
    rng = random.Random(SEED)
    fold = folds(final, rng)
    assert len(fold[0]) + len(fold[1]) == 110 and not (fold[0] & fold[1])
    done = {r["call_id"] for r in load_jsonl(OUT_PATH) if r.get("ok")} if OUT_PATH.exists() else set()
    jobs = []
    meta = {}
    for author in GRID_A:
        maj_batt = untied_majority(choice_counts(screen, author, final_ids))
        maj_held = untied_majority(choice_counts(screen, author, set(heldout_ids)))
        meta[author] = {"untied_battery": len(maj_batt), "untied_heldout": len(maj_held)}
        prompts = {}
        if "55" in conds:
            for k in (0, 1):
                src = sorted(i for i in fold[1 - k] if i in maj_batt)
                lrng = random.Random(f"{SEED}|{author}|fold{k}")
                lrng.shuffle(src)
                lines = balanced_lines(final_by_id, maj_batt, src, lrng)
                prompts[("demos_bal", 55, k)] = (DEMO_PROMPT.format(n=len(lines), demos="\n".join(lines)),
                                                 [final_by_id[i] for i in sorted(fold[k])])
                meta[author][f"fold{k}_demos"] = len(lines)
        if "30" in conds:
            lrng = random.Random(f"{SEED}|{author}|held")
            src = sorted(maj_held)
            lrng.shuffle(src)
            src = sorted(src[:30])
            lrng.shuffle(src)
            lines = balanced_lines(cand, maj_held, src, lrng)
            prompts[("fewshot_bal", 30, None)] = (DEMO_PROMPT.format(n=len(lines), demos="\n".join(lines)), final)
        for (arm, length, k), (system, test_items) in prompts.items():
            for receiver in GRID_A:
                provider, model_id, extra, max_tok = MODEL_DEFS[receiver]
                max_tok = MAX_TOKENS_OVERRIDE.get(receiver, max_tok)
                cell_id = f"{receiver}|{author}|{arm}|{length}|r0"
                for item in test_items:
                    for order in ("orig", "flip"):
                        a, b = item["option_a"], item["option_b"]
                        if order == "flip":
                            a, b = b, a
                        for sample in range(N_SAMPLES):
                            cid = f"grid|{cell_id}|{item['id']}|{order}|{sample}"
                            if cid in done:
                                continue
                            jobs.append({"call_id": cid, "cell_id": cell_id, "receiver": receiver,
                                         "author": author, "arm": arm, "length": length, "regen": 0,
                                         "provider": provider, "model": model_id, "extra_body": extra,
                                         "max_tokens": max_tok, "system": system,
                                         "prompt": PROMPT_TEMPLATE.format(opt_a=a, opt_b=b),
                                         "item_id": item["id"], "category": item["category"],
                                         "order": order, "sample": sample, "fold": k,
                                         "seed": stable_seed("grid", cell_id, item["id"], order, str(sample))})
    print("per-author demonstration counts:", json.dumps(meta), flush=True)
    print(f"balanced demonstrations: {len({j['cell_id'] for j in jobs})} cells, {len(jobs)} calls", flush=True)
    if not jobs:
        return
    client = LLMClient(timeout=240.0)
    print(f"credits ok: ${client.ensure_credits(CREDIT_FLOOR):.2f}", flush=True)
    lock = threading.Lock()
    counters = {"done": 0, "fail": 0}
    start = time.time()

    def worker(job):
        rec = run_job(client, job)
        rec["fold"] = job["fold"]
        with lock:
            with OUT_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counters["done" if rec["ok"] else "fail"] += 1
            n = counters["done"] + counters["fail"]
            if n % 1000 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} ({counters['fail']} failed) {n/(time.time()-start):.1f}/s", flush=True)
            if counters["done"] % 2000 == 0 and counters["done"] > 0:
                client.ensure_credits(CREDIT_FLOOR)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for f in as_completed([pool.submit(worker, j) for j in jobs]):
            f.result()
    print(f"finished: {counters['done']} ok, {counters['fail']} failed, {int(time.time() - start)}s; "
          f"credits ${client.credits_remaining():.2f}", flush=True)


if __name__ == "__main__":
    main()
