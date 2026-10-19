"""Observation-derived descriptions from STABLE items (pilot/protocol_arxiv_obs_stable.md).
Grid A. Writes descriptions/desc3p_stable_{target}_by_{describer}_fold{k}.md (+ manifest_arxiv
lines) and runs/arxiv_obs_stable.jsonl. Resumable."""

from __future__ import annotations

import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from elicit_thirdperson import THIRD_PERSON_PROMPT
from llm_client import LLMClient
from run_arxiv_demos_balanced import SEED, choice_counts, folds, untied_majority
from run_arxiv_positive_controls import MAX_TOKENS_OVERRIDE, load_jsonl
from run_grid import GRID_A, MODEL_DEFS, N_SAMPLES, run_job
from run_pilot import PROMPT_TEMPLATE, stable_seed

ROOT = Path(__file__).resolve().parent.parent
ITEMS_FINAL = ROOT / "battery" / "items_final.json"
SCREEN = ROOT / "runs" / "screening.jsonl"
DESC_DIR = ROOT / "descriptions"
MANIFEST = DESC_DIR / "manifest_arxiv.jsonl"
OUT_PATH = ROOT / "runs" / "arxiv_obs_stable.jsonl"
CREDIT_FLOOR = 2.0
MAX_WORKERS = 32
THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


def elicit(client, target, describer, obs_lines, k):
    fname = f"desc3p_stable_{target}_by_{describer}_fold{k}.md"
    path = DESC_DIR / fname
    if not path.exists():
        provider, model_id, extra, _ = MODEL_DEFS[describer]
        prompt = THIRD_PERSON_PROMPT.format(n_obs=len(obs_lines), observations="\n".join(obs_lines))
        r = client.chat(provider, model_id, [{"role": "user", "content": prompt}],
                        temperature=1.0, max_tokens=1400, seed=42 + k, extra_body=extra)
        text = THINK_RE.sub(" ", r.text).strip()
        path.write_text(text, encoding="utf-8")
        with MANIFEST.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"desc_id": f"3p_stable|{target}|by|{describer}|fold{k}", "ok": True,
                                 "file": fname, "words": len(text.split()), "n_obs": len(obs_lines),
                                 "prompt": "THIRD_PERSON_PROMPT verbatim", "temperature": 1.0,
                                 "seed": 42 + k, "source": "untied battery items of the other fold"}) + "\n")
        print(f"  elicited {fname}: {len(text.split())} words", flush=True)
    text = path.read_text(encoding="utf-8").strip()
    assert len(text.split()) > 150, f"short description {fname}"
    return text


def main() -> None:
    final = json.loads(ITEMS_FINAL.read_text(encoding="utf-8"))
    by_id = {it["id"]: it for it in final}
    final_ids = set(by_id)
    screen = load_jsonl(SCREEN)
    fold = folds(final, random.Random(SEED))
    client = LLMClient(timeout=240.0)
    print(f"credits: ${client.ensure_credits(CREDIT_FLOOR):.2f}", flush=True)
    done = {r["call_id"] for r in load_jsonl(OUT_PATH) if r.get("ok")} if OUT_PATH.exists() else set()
    jobs = []
    for target in GRID_A:
        maj = untied_majority(choice_counts(screen, target, final_ids))
        for k in (0, 1):
            src = sorted(i for i in fold[1 - k] if i in maj)
            obs = []
            for i in src:
                it = by_id[i]
                chosen = it["option_a"] if maj[i] == "option_a" else it["option_b"]
                other = it["option_b"] if maj[i] == "option_a" else it["option_a"]
                obs.append(f'- Given "{chosen}" vs "{other}", it chose: "{chosen}"')
            test_items = [by_id[i] for i in sorted(fold[k])]
            for describer in GRID_A:
                text = elicit(client, target, describer, obs, k)
                if describer == target:
                    arm, receivers = "selfobs_stable", GRID_A
                else:
                    arm, receivers = f"3p_stable_by_{describer}", [target]
                for receiver in receivers:
                    provider, model_id, extra, max_tok = MODEL_DEFS[receiver]
                    max_tok = MAX_TOKENS_OVERRIDE.get(receiver, max_tok)
                    cell_id = f"{receiver}|{target}|{arm}|500|r0"
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
                                             "author": target, "arm": "selfobs_stable" if arm == "selfobs_stable" else "3p_stable",
                                             "length": 500, "regen": 0, "provider": provider, "model": model_id,
                                             "extra_body": extra, "max_tokens": max_tok, "system": text,
                                             "prompt": PROMPT_TEMPLATE.format(opt_a=a, opt_b=b),
                                             "item_id": item["id"], "category": item["category"],
                                             "order": order, "sample": sample, "fold": k,
                                             "seed": stable_seed("grid", cell_id, item["id"], order, str(sample))})
    print(f"obs-stable: {len({j['cell_id'] for j in jobs})} cells, {len(jobs)} calls", flush=True)
    if not jobs:
        return
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
