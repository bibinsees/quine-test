"""arXiv revision: local-trio runs (Ollama), preregistered in
pilot/protocol_arxiv_local_replication.md. Phases, in order:

  0  hermes3 refusal cells rerun with r1 descriptions   (run_arxiv_hermes_r1.main)
  1  held-out screening for the trio (90 items, generic prompt, 2 orders x 3 samples)
     -> runs/arxiv_local_heldout.jsonl   (fields mirror runs/screening.jsonl)
  2  demonstrations-110 on the trio (9 cells)           -> runs/arxiv_local_demos.jsonl
  3  demonstrations-30  on the trio (9 cells)
  4  demonstrations-90  on the trio (9 cells)

Jobs are ordered by receiver so the server keeps one model loaded at a time.
Resumable: every phase skips calls already logged ok.
"""

from __future__ import annotations

import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import run_arxiv_hermes_r1
from llm_client import LLMClient
from run_arxiv_demos_dose import nested_sets
from run_arxiv_positive_controls import DEMO_PROMPT, demo_lines, load_jsonl, majority_choices
from run_grid import LOCALS, MODEL_DEFS, N_SAMPLES, run_job
from run_pilot import PROMPT_TEMPLATE, SYSTEM_PROMPT, stable_seed

ROOT = Path(__file__).resolve().parent.parent
ITEMS_FINAL = ROOT / "battery" / "items_final.json"
ITEMS_CAND = ROOT / "battery" / "items_candidates.json"
GRID = ROOT / "runs" / "grid.jsonl"
HELDOUT_PATH = ROOT / "runs" / "arxiv_local_heldout.jsonl"
DEMOS_PATH = ROOT / "runs" / "arxiv_local_demos.jsonl"

SCREEN_SAMPLES = 3
MAX_WORKERS = 4
PHASES = {"110": 2, "30": 3, "90": 4}


def run_pool(client: LLMClient, jobs: list[dict], out_path: Path, label: str,
             extra_keys: tuple[str, ...] = ()) -> None:
    done = {r["call_id"] for r in load_jsonl(out_path) if r.get("ok")} if out_path.exists() else set()
    todo = [j for j in jobs if j["call_id"] not in done]
    print(f"=== {label}: {len(jobs)} calls, {len(todo)} to run", flush=True)
    if not todo:
        return
    lock = threading.Lock()
    counters = {"done": 0, "fail": 0}
    start = time.time()

    def worker(job):
        rec = run_job(client, job)
        for k in extra_keys:
            rec[k] = job[k]
        with lock:
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counters["done" if rec["ok"] else "fail"] += 1
            n = counters["done"] + counters["fail"]
            if n % 200 == 0 or n == len(todo):
                rate = n / max(time.time() - start, 1e-9)
                print(f"  {n}/{len(todo)} ({counters['fail']} failed) {rate:.2f}/s "
                      f"eta {int((len(todo) - n) / max(rate, 1e-9))}s", flush=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for f in as_completed([pool.submit(worker, j) for j in todo]):
            f.result()
    print(f"  {label} finished: {counters['done']} ok, {counters['fail']} failed, "
          f"{int(time.time() - start)}s", flush=True)


def heldout_jobs(heldout: list[dict]) -> list[dict]:
    jobs = []
    for short in LOCALS:
        provider, model_id, extra, max_tok = MODEL_DEFS[short]
        for item in heldout:
            for order in ("orig", "flip"):
                a, b = item["option_a"], item["option_b"]
                if order == "flip":
                    a, b = b, a
                for sample in range(SCREEN_SAMPLES):
                    cid = f"heldout|{short}|{item['id']}|{order}|{sample}"
                    jobs.append({
                        "call_id": cid, "cell_id": f"{short}|heldout", "receiver": short,
                        "author": None, "arm": "heldout", "length": 0, "regen": 0,
                        "provider": provider, "model": model_id, "extra_body": extra,
                        "max_tokens": max_tok, "system": SYSTEM_PROMPT,
                        "prompt": PROMPT_TEMPLATE.format(opt_a=a, opt_b=b),
                        "item_id": item["id"], "category": item["category"],
                        "order": order, "sample": sample, "short": short,
                        "seed": stable_seed("heldout", short, item["id"], order, str(sample)),
                    })
    return jobs


def demo_jobs(k: int, final: list[dict], cand: dict, heldout_ids: list[str]) -> list[dict]:
    grid_base = [r for r in load_jsonl(GRID) if r.get("cell_id", "").endswith("|baseline")]
    final_ids = {it["id"] for it in final}
    held = load_jsonl(HELDOUT_PATH) if HELDOUT_PATH.exists() else []
    sets = nested_sets(heldout_ids)
    jobs = []
    for receiver in LOCALS:
        provider, model_id, extra, max_tok = MODEL_DEFS[receiver]
        for author in LOCALS:
            if k == 110:
                maj = majority_choices(grid_base, author, final_ids, "receiver", "item_id")
                assert len(maj) == len(final_ids), f"{author}: battery baseline incomplete"
                order_ids = random.Random(20261014).sample(sorted(final_ids), len(final_ids))
                lines = demo_lines({it["id"]: it for it in final}, maj, order_ids)
            else:
                maj = majority_choices(held, author, set(heldout_ids), "short", "item_id")
                missing = [i for i in sets[k] if i not in maj]
                assert not missing, f"{author}: {len(missing)} held-out items without screening data"
                lines = demo_lines(cand, maj, sets[k])
            system = DEMO_PROMPT.format(n=len(lines), demos="\n".join(lines))
            arm = "oracle" if k == 110 else "fewshot"
            cell_id = f"{receiver}|{author}|{arm}|{k}|r0"
            for item in final:
                for order in ("orig", "flip"):
                    a, b = item["option_a"], item["option_b"]
                    if order == "flip":
                        a, b = b, a
                    for sample in range(N_SAMPLES):
                        cid = f"grid|{cell_id}|{item['id']}|{order}|{sample}"
                        jobs.append({
                            "call_id": cid, "cell_id": cell_id, "receiver": receiver,
                            "author": author, "arm": arm, "length": k, "regen": 0,
                            "provider": provider, "model": model_id, "extra_body": extra,
                            "max_tokens": max_tok, "system": system,
                            "prompt": PROMPT_TEMPLATE.format(opt_a=a, opt_b=b),
                            "item_id": item["id"], "category": item["category"],
                            "order": order, "sample": sample,
                            "seed": stable_seed("grid", cell_id, item["id"], order, str(sample)),
                        })
    return jobs


def main() -> None:
    phases = sys.argv[1:] or ["0", "1", "110", "30", "90"]
    final = json.loads(ITEMS_FINAL.read_text(encoding="utf-8"))
    cand = {it["id"]: it for it in json.loads(ITEMS_CAND.read_text(encoding="utf-8"))}
    final_ids = {it["id"] for it in final}
    heldout_ids = sorted(set(cand) - final_ids)
    heldout = [cand[i] for i in heldout_ids]
    client = LLMClient(timeout=300.0)

    if "0" in phases:
        print("=== phase 0: hermes3 r1 rerun", flush=True)
        run_arxiv_hermes_r1.main()
    if "1" in phases:
        run_pool(client, heldout_jobs(heldout), HELDOUT_PATH, "phase 1: trio held-out screening",
                 extra_keys=("short",))
    for k in ("110", "30", "90"):
        if k in phases:
            run_pool(client, demo_jobs(int(k), final, cand, heldout_ids), DEMOS_PATH,
                     f"phase {PHASES[k]}: trio demonstrations-{k}")
    print("all requested phases done", flush=True)


if __name__ == "__main__":
    main()
