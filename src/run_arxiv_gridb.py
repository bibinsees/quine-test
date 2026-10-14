"""arXiv revision: Grid B replication of the added conditions, preregistered in
pilot/protocol_arxiv_gridb_replication.md. OpenRouter only.

Phases (in order; each resumable, later ones dropped if credits run out):
  110  demonstrations-110 (9 cells)
  30   demonstrations-30  (9 cells; same 30-item held-out subset as Grid A)
  so   self-observation-500: 3 descriptions (third-person prompt verbatim, own
       observations) + 9 cells
  3p   observer-written third-person-500: 6 descriptions + 12 cells
       (receiver = target and receiver = the third Grid B model), as in the sprint

New files only: runs/arxiv_gridb.jsonl, descriptions/desc3p_{target}_by_{describer}_500_r0.md
(Grid B names), descriptions/manifest_arxiv.jsonl lines.
"""

from __future__ import annotations

import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from elicit_thirdperson import THIRD_PERSON_PROMPT
from llm_client import LLMClient
from run_arxiv_positive_controls import (DEMO_PROMPT, FEWSHOT_SEED, K_FEWSHOT, demo_lines,
                                         load_jsonl, majority_choices)
from run_grid import GRID_B, MODEL_DEFS, N_SAMPLES, run_job
from run_pilot import PROMPT_TEMPLATE, stable_seed

ROOT = Path(__file__).resolve().parent.parent
ITEMS_FINAL = ROOT / "battery" / "items_final.json"
ITEMS_CAND = ROOT / "battery" / "items_candidates.json"
SCREEN = ROOT / "runs" / "screening.jsonl"
DESC_DIR = ROOT / "descriptions"
MANIFEST = DESC_DIR / "manifest_arxiv.jsonl"
OUT_PATH = ROOT / "runs" / "arxiv_gridb.jsonl"

CREDIT_FLOOR = 2.0
MAX_WORKERS = 24
THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


def log_manifest(**kw) -> None:
    with MANIFEST.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(kw, ensure_ascii=False) + "\n")


def obs_lines(cand: dict, maj: dict[str, str], ids: list[str]) -> list[str]:
    out = []
    for item_id in ids:
        it = cand[item_id]
        chosen = it["option_a"] if maj[item_id] == "option_a" else it["option_b"]
        other = it["option_b"] if maj[item_id] == "option_a" else it["option_a"]
        out.append(f'- Given "{chosen}" vs "{other}", it chose: "{chosen}"')
    return out


def elicit(client: LLMClient, target: str, describer: str, cand: dict, maj: dict,
           heldout_ids: list[str]) -> str:
    fname = f"desc3p_{target}_by_{describer}_500_r0.md"
    path = DESC_DIR / fname
    if not path.exists():
        obs = obs_lines(cand, maj, heldout_ids)
        provider, model_id, extra, _ = MODEL_DEFS[describer]
        prompt = THIRD_PERSON_PROMPT.format(n_obs=len(obs), observations="\n".join(obs))
        r = client.chat(provider, model_id, [{"role": "user", "content": prompt}],
                        temperature=1.0, max_tokens=1400, seed=42, extra_body=extra)
        text = THINK_RE.sub(" ", r.text).strip()
        path.write_text(text, encoding="utf-8")
        log_manifest(desc_id=f"3p|{target}|by|{describer}", ok=True, file=fname,
                     words=len(text.split()), n_obs=len(obs),
                     prompt="THIRD_PERSON_PROMPT verbatim", temperature=1.0, seed=42,
                     grid="B")
        print(f"  elicited {fname}: {len(text.split())} words", flush=True)
    text = path.read_text(encoding="utf-8").strip()
    assert len(text.split()) > 150, f"short description {fname}"
    return text


def cell_jobs(receiver: str, author: str, system: str, items: list[dict], cell_id: str,
              arm: str, length: int) -> list[dict]:
    provider, model_id, extra, max_tok = MODEL_DEFS[receiver]
    jobs = []
    for item in items:
        for order in ("orig", "flip"):
            a, b = item["option_a"], item["option_b"]
            if order == "flip":
                a, b = b, a
            for sample in range(N_SAMPLES):
                jobs.append({
                    "call_id": f"grid|{cell_id}|{item['id']}|{order}|{sample}",
                    "cell_id": cell_id, "receiver": receiver, "author": author,
                    "arm": arm, "length": length, "regen": 0, "provider": provider,
                    "model": model_id, "extra_body": extra, "max_tokens": max_tok,
                    "system": system, "prompt": PROMPT_TEMPLATE.format(opt_a=a, opt_b=b),
                    "item_id": item["id"], "category": item["category"],
                    "order": order, "sample": sample,
                    "seed": stable_seed("grid", cell_id, item["id"], order, str(sample)),
                })
    return jobs


def run_pool(client: LLMClient, jobs: list[dict], label: str) -> None:
    done = {r["call_id"] for r in load_jsonl(OUT_PATH) if r.get("ok")} if OUT_PATH.exists() else set()
    todo = [j for j in jobs if j["call_id"] not in done]
    print(f"=== {label}: {len(jobs)} calls, {len(todo)} to run", flush=True)
    if not todo:
        return
    client.ensure_credits(CREDIT_FLOOR)
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
            if n % 500 == 0 or n == len(todo):
                rate = n / max(time.time() - start, 1e-9)
                print(f"  {n}/{len(todo)} ({counters['fail']} failed) {rate:.1f}/s", flush=True)
            if counters["done"] % 1000 == 0 and counters["done"] > 0:
                client.ensure_credits(CREDIT_FLOOR)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for f in as_completed([pool.submit(worker, j) for j in todo]):
            f.result()
    print(f"  {label} finished: {counters['done']} ok, {counters['fail']} failed; "
          f"credits ${client.credits_remaining():.2f}", flush=True)


def main() -> None:
    phases = sys.argv[1:] or ["110", "30", "so", "3p"]
    final = json.loads(ITEMS_FINAL.read_text(encoding="utf-8"))
    cand = {it["id"]: it for it in json.loads(ITEMS_CAND.read_text(encoding="utf-8"))}
    final_ids = {it["id"] for it in final}
    heldout_ids = sorted(set(cand) - final_ids)
    screen = load_jsonl(SCREEN)
    maj_held = {a: majority_choices(screen, a, set(heldout_ids), "short", "item_id") for a in GRID_B}
    maj_batt = {a: majority_choices(screen, a, final_ids, "short", "item_id") for a in GRID_B}
    for a in GRID_B:
        assert len(maj_held[a]) == len(heldout_ids) and len(maj_batt[a]) == len(final_ids), a
    k30 = sorted(random.Random(FEWSHOT_SEED).sample(heldout_ids, K_FEWSHOT))
    client = LLMClient(timeout=240.0)
    print(f"credits: ${client.credits_remaining():.2f}", flush=True)

    if "110" in phases:
        jobs = []
        for receiver in GRID_B:
            for author in GRID_B:
                order_ids = random.Random(20261014).sample(sorted(final_ids), len(final_ids))
                lines = demo_lines({it["id"]: it for it in final}, maj_batt[author], order_ids)
                jobs += cell_jobs(receiver, author, DEMO_PROMPT.format(n=len(lines), demos="\n".join(lines)),
                                  final, f"{receiver}|{author}|oracle|110|r0", "oracle", 110)
        run_pool(client, jobs, "Grid B demonstrations-110")
    if "30" in phases:
        jobs = []
        for receiver in GRID_B:
            for author in GRID_B:
                lines = demo_lines(cand, maj_held[author], k30)
                jobs += cell_jobs(receiver, author, DEMO_PROMPT.format(n=len(lines), demos="\n".join(lines)),
                                  final, f"{receiver}|{author}|fewshot|30|r0", "fewshot", 30)
        run_pool(client, jobs, "Grid B demonstrations-30")
    if "so" in phases:
        jobs = []
        for author in GRID_B:
            text = elicit(client, author, author, cand, maj_held[author], heldout_ids)
            for receiver in GRID_B:
                jobs += cell_jobs(receiver, author, text, final,
                                  f"{receiver}|{author}|selfobs|500|r0", "selfobs", 500)
        run_pool(client, jobs, "Grid B self-observation-500")
    if "3p" in phases:
        jobs = []
        for target in GRID_B:
            others = [m for m in GRID_B if m != target]
            for describer in others:
                third = [m for m in others if m != describer][0]
                text = elicit(client, target, describer, cand, maj_held[target], heldout_ids)
                for receiver in (target, third):
                    jobs += cell_jobs(receiver, target, text, final,
                                      f"{receiver}|{target}|3p_by_{describer}|500|r0", "3p", 500)
        run_pool(client, jobs, "Grid B observer-written third-person-500")
    print(f"all requested phases done; credits ${client.credits_remaining():.2f}", flush=True)


if __name__ == "__main__":
    main()
