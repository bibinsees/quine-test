"""Transfer from a known semantic persona (pilot/protocol_arxiv_policy_persona.md). Grid A.

Stages (run in order; each is resumable and skips cached calls):
  source     every Grid A model runs each policy text as system prompt, all 110 items
             (arm "policy"; these cells are both the targets p_{S|P} and the
             receiver-reads-policy cells p_{R|P})
  gate       development gate on fold-0 items (no calls); writes results/arxiv/policy_gate.json
  desc       for passing (S, P): S under policy P writes a 500-word NEUTRAL self-description
             -> descriptions/desc_policy_{S}_{P}.md (+ manifest_arxiv line)
  receivers  every Grid A model runs D_{S,P} as system prompt on fold-1 items (arm "policy_desc")
  all        the four stages in sequence

Writes runs/arxiv_policy.jsonl. usage: run_arxiv_policy.py <stage>
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

from elicit_descriptions import NEUTRAL_PROMPT
from llm_client import LLMClient
from run_arxiv_demos_balanced import SEED, folds
from run_arxiv_positive_controls import MAX_TOKENS_OVERRIDE, load_jsonl
from run_grid import GRID_A, MODEL_DEFS, N_SAMPLES, run_job
from run_pilot import PROMPT_TEMPLATE, stable_seed

ROOT = Path(__file__).resolve().parent.parent
ITEMS_FINAL = ROOT / "battery" / "items_final.json"
DESC_DIR = ROOT / "descriptions"
MANIFEST = DESC_DIR / "manifest_arxiv.jsonl"
OUT_PATH = ROOT / "runs" / "arxiv_policy.jsonl"
GATE_PATH = ROOT / "results" / "arxiv" / "policy_gate.json"
POLICIES = ["guardian", "pioneer", "ledger"]
G1_MAX_F, G2_MAX_F = 0.92, 0.95
MIN_DESC_WORDS = 300
CREDIT_FLOOR = 2.0
MAX_WORKERS = 32
THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


def policy_text(name: str) -> str:
    return (DESC_DIR / f"policy_{name}.md").read_text(encoding="utf-8").strip()


def item_jobs(receiver, author, arm, length, cell_id, system, items, done, extra_rec):
    provider, model_id, extra, max_tok = MODEL_DEFS[receiver]
    max_tok = MAX_TOKENS_OVERRIDE.get(receiver, max_tok)
    jobs = []
    for item in items:
        for order in ("orig", "flip"):
            a, b = item["option_a"], item["option_b"]
            if order == "flip":
                a, b = b, a
            for sample in range(N_SAMPLES):
                cid = f"grid|{cell_id}|{item['id']}|{order}|{sample}"
                if cid in done:
                    continue
                jobs.append({"call_id": cid, "cell_id": cell_id, "receiver": receiver, "author": author,
                             "arm": arm, "length": length, "regen": 0, "provider": provider, "model": model_id,
                             "extra_body": extra, "max_tokens": max_tok, "system": system,
                             "prompt": PROMPT_TEMPLATE.format(opt_a=a, opt_b=b), "item_id": item["id"],
                             "category": item["category"], "order": order, "sample": sample,
                             "seed": stable_seed("grid", cell_id, item["id"], order, str(sample)),
                             "_extra": extra_rec})
    return jobs


def run_jobs(client, jobs, label):
    print(f"{label}: {len({j['cell_id'] for j in jobs})} cells, {len(jobs)} calls", flush=True)
    if not jobs:
        return
    lock = threading.Lock()
    counters = {"done": 0, "fail": 0}
    start = time.time()

    def worker(job):
        rec = run_job(client, job)
        rec.update(job["_extra"])
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
    print(f"  finished: {counters['done']} ok, {counters['fail']} failed, {int(time.time() - start)}s; "
          f"credits ${client.credits_remaining():.2f}", flush=True)


def done_ids():
    return {r["call_id"] for r in load_jsonl(OUT_PATH) if r.get("ok")} if OUT_PATH.exists() else set()


def stage_source(client, final):
    jobs = []
    for name in POLICIES:
        for receiver in GRID_A:
            jobs += item_jobs(receiver, None, "policy", 0, f"{receiver}|policy|{name}|0|r0", policy_text(name),
                              final, done_ids(), {"policy": name})
    run_jobs(client, jobs, "stage A (source/receiver under policy)")


def stage_gate(final):
    import analyze_grid as ag
    fold = folds(final, random.Random(SEED))
    dev = fold[0]
    rows = ag.load_jsonl(OUT_PATH)
    screen = ag.load_jsonl(ag.SCREEN)
    ref = ag.p_hat_table(screen[screen["item_id"].isin({it["id"] for it in final})], "short")

    def prof(receiver, name):
        g = rows[(rows["arm"] == "policy") & (rows["receiver"] == receiver) & (rows["policy"] == name)
                 & rows["choice"].notna() & rows["item_id"].isin(dev)]
        return {i: (gg["choice"] == "option_a").mean() for i, gg in g.groupby("item_id")}

    def F(p, q):
        common = sorted(set(p) & set(q))
        import numpy as np
        a, b = np.array([p[i] for i in common]), np.array([q[i] for i in common])
        return float((1 - ag.jsd(a, b)).mean()), len(common)

    gate = {"G1": {}, "G2": {}, "passing": []}
    for s in GRID_A:
        dev_ref = {i: v for i, v in ref[s].items() if i in dev}
        prof_s = {name: prof(s, name) for name in POLICIES}
        g1 = {}
        for name in POLICIES:
            f, n = F(dev_ref, prof_s[name])
            g1[name] = {"F_to_generic": f, "n_items": n, "pass": f <= G1_MAX_F}
        gate["G1"][s] = g1
        passed = [name for name in POLICIES if g1[name]["pass"]]
        g2 = {}
        for name in passed:
            ok = True
            for other in passed:
                if other != name:
                    f, n = F(prof_s[name], prof_s[other])
                    g2[f"{name}|{other}"] = {"F": f, "n_items": n, "pass": f <= G2_MAX_F}
                    ok &= f <= G2_MAX_F
            if ok:
                gate["passing"].append([s, name])
        gate["G2"][s] = g2
    GATE_PATH.write_text(json.dumps(gate, indent=1), encoding="utf-8")
    for s in GRID_A:
        print(f"  {s}: " + "  ".join(f"{n} F={gate['G1'][s][n]['F_to_generic']:.3f}{'' if gate['G1'][s][n]['pass'] else ' FAIL'}"
                                    for n in POLICIES))
        for k, v in gate["G2"][s].items():
            print(f"      {k}: F={v['F']:.3f}{'' if v['pass'] else ' FAIL'}")
    print(f"  passing targets: {gate['passing']}")
    return gate


def load_gate():
    return json.loads(GATE_PATH.read_text(encoding="utf-8"))


def stage_desc(client):
    gate = load_gate()
    for s, name in gate["passing"]:
        path = DESC_DIR / f"desc_policy_{s}_{name}.md"
        if path.exists():
            continue
        provider, model_id, extra, _ = MODEL_DEFS[s]
        base_seed = stable_seed("policy_desc", s, name)
        text, words, seed = "", 0, base_seed
        for attempt in range(2):
            seed = base_seed + attempt
            r = client.chat(provider, model_id, [{"role": "system", "content": policy_text(name)},
                                                 {"role": "user", "content": NEUTRAL_PROMPT.format(n=500)}],
                            temperature=1.0, max_tokens=1400, seed=seed, extra_body=extra)
            text = THINK_RE.sub(" ", r.text).strip()
            words = len(text.split())
            if words >= MIN_DESC_WORDS:
                break
        ok = words >= MIN_DESC_WORDS
        if ok:
            path.write_text(text, encoding="utf-8")
        with MANIFEST.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"desc_id": f"policy_desc|{s}|{name}", "ok": ok, "file": path.name if ok else None,
                                 "words": words, "prompt": "NEUTRAL_PROMPT(500) with the policy text as system prompt",
                                 "temperature": 1.0, "seed": seed}) + "\n")
        print(f"  {s} under {name}: {words} words{'' if ok else '  <-- EXCLUDED (short)'}", flush=True)


def stage_receivers(client, final):
    gate = load_gate()
    fold = folds(final, random.Random(SEED))
    eval_items = [it for it in final if it["id"] in fold[1]]
    jobs = []
    for s, name in gate["passing"]:
        path = DESC_DIR / f"desc_policy_{s}_{name}.md"
        if not path.exists():
            print(f"  no description for {s}/{name}; skipped", flush=True)
            continue
        text = path.read_text(encoding="utf-8").strip()
        for receiver in GRID_A:
            jobs += item_jobs(receiver, s, "policy_desc", 500, f"{receiver}|{s}|policy_desc_{name}|500|r0", text,
                              eval_items, done_ids(), {"policy": name, "fold": 1})
    run_jobs(client, jobs, "stage C (receivers read the policy-conditioned self-description)")


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    final = json.loads(ITEMS_FINAL.read_text(encoding="utf-8"))
    client = LLMClient(timeout=240.0)
    if stage != "gate":
        print(f"credits: ${client.ensure_credits(CREDIT_FLOOR):.2f}", flush=True)
    if stage in ("source", "all"):
        stage_source(client, final)
    if stage in ("gate", "all"):
        stage_gate(final)
    if stage in ("desc", "all"):
        stage_desc(client)
    if stage in ("receivers", "all"):
        stage_receivers(client, final)


if __name__ == "__main__":
    main()
