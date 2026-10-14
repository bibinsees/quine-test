"""arXiv revision: descriptions optimized directly for behavioral reconstruction
(reviewer 2's suggested decisive test). Grid A, all 6 directed pairs.

For each (author, receiver) pair:
  round 0   the author's natural neutral-500 r0 self-description is run by the
            receiver on the 90 HELD-OUT screening items (2 orders x 2 samples)
            and scored against the author's held-out profile (screening data).
  rounds 1..N_ROUNDS
            an optimizer model (non-grid) sees the current best description and,
            per held-out item, the author's choice and how often the receiver
            chose it, and rewrites the description (~500 words, first person,
            no quoting of items). The rewrite is evaluated the same way; the
            best description so far is kept (greedy).
  test      the best description (selected on held-out items only) is run on
            the 110-item battery by the receiver (cross cell) and by the author
            (self cell). Selection and test never share items.

New files only:
  descriptions/desc_opt_{author}_for_{receiver}_round{k}.md (+ manifest_arxiv.jsonl)
  runs/arxiv_optimized.jsonl          receiver calls (phase=select|test)
  runs/arxiv_optimizer_calls.jsonl    optimizer calls (prompt, response, scores)
"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from llm_client import LLMClient
from run_grid import GRID_A, MODEL_DEFS, N_SAMPLES, desc_text, run_job
from run_pilot import PROMPT_TEMPLATE, stable_seed

ROOT = Path(__file__).resolve().parent.parent
ITEMS_FINAL = ROOT / "battery" / "items_final.json"
ITEMS_CAND = ROOT / "battery" / "items_candidates.json"
SCREEN = ROOT / "runs" / "screening.jsonl"
DESC_DIR = ROOT / "descriptions"
MANIFEST = DESC_DIR / "manifest_arxiv.jsonl"
OUT_PATH = ROOT / "runs" / "arxiv_optimized.jsonl"
OPT_LOG = ROOT / "runs" / "arxiv_optimizer_calls.jsonl"

N_ROUNDS = 5
OPTIMIZER = ("openrouter", "anthropic/claude-sonnet-4.6", None)  # non-grid model
CREDIT_FLOOR = 2.0
MAX_WORKERS = 32
MAX_TOKENS_OVERRIDE = {"glm-4.5-air": 2048}
THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)

OPTIMIZER_PROMPT = (
    "You are optimizing a system prompt. A TARGET AI system has a stable profile "
    "of preferences. A RECEIVER AI system is given a first-person description of "
    "the target as its system prompt and must then answer preference questions "
    "exactly as the target would. Your job is to rewrite the description so that "
    "the receiver's choices match the target's more closely.\n\n"
    "CURRENT DESCRIPTION (first person, as if written by the target):\n"
    "---\n{description}\n---\n\n"
    "RESULTS WITH THE CURRENT DESCRIPTION. For each question the target was "
    "given two options; 'target chose' is the target's choice and 'receiver "
    "match' is how often the receiver, running the description above, made the "
    "same choice (1.00 = always, 0.00 = never). Items with low match are where "
    "the description fails.\n\n{results}\n\n"
    "Write an improved description of about 500 words, in the first person, as "
    "the target describing itself: its values, preferences, personality, how it "
    "reasons and how it decides between options. Make it as effective as "
    "possible at reproducing the target's choices on NEW questions of the same "
    "kinds, not just these. Do not quote, list, or paraphrase the specific "
    "questions above, and do not mention receivers, targets, matches, or this "
    "optimization. Output only the description."
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    def h(x):
        x = np.clip(x, 1e-12, 1 - 1e-12)
        return -(x * np.log2(x) + (1 - x) * np.log2(1 - x))
    m = (p + q) / 2
    return h(m) - (h(p) + h(q)) / 2


def fidelity(ref: dict, cell: dict) -> float:
    common = sorted(set(ref) & set(cell))
    p = np.array([ref[i] for i in common])
    q = np.array([cell[i] for i in common])
    return float(np.mean(1 - jsd(p, q)))


class Runner:
    def __init__(self) -> None:
        self.client = LLMClient(timeout=240.0)
        self.lock = threading.Lock()
        self.done: dict[str, dict] = {}
        for rec in load_jsonl(OUT_PATH):
            if rec.get("ok"):
                self.done[rec["call_id"]] = rec

    def run_jobs(self, jobs: list[dict], label: str) -> list[dict]:
        todo = [j for j in jobs if j["call_id"] not in self.done]
        print(f"  {label}: {len(jobs)} calls, {len(todo)} to run", flush=True)
        if todo:
            self.client.ensure_credits(CREDIT_FLOOR)
            counters = {"fail": 0}

            def worker(job):
                rec = run_job(self.client, job)
                for k in ("phase", "round", "optimized_for", "item_set"):
                    rec[k] = job[k]
                with self.lock:
                    with OUT_PATH.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if rec["ok"]:
                        self.done[rec["call_id"]] = rec
                    else:
                        counters["fail"] += 1

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                for f in as_completed([pool.submit(worker, j) for j in todo]):
                    f.result()
            if counters["fail"]:
                print(f"  {label}: {counters['fail']} failed (rerun to resume)", flush=True)
        return [self.done[j["call_id"]] for j in jobs if j["call_id"] in self.done]


def make_jobs(receiver: str, author: str, system: str, items: list[dict], cell_id: str,
              phase: str, rnd: int, optimized_for: str, item_set: str) -> list[dict]:
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
                jobs.append({
                    "call_id": cid, "cell_id": cell_id, "receiver": receiver,
                    "author": author, "arm": "optimized", "length": 500, "regen": 0,
                    "provider": provider, "model": model_id, "extra_body": extra,
                    "max_tokens": max_tok, "system": system,
                    "prompt": PROMPT_TEMPLATE.format(opt_a=a, opt_b=b),
                    "item_id": item["id"], "category": item["category"],
                    "order": order, "sample": sample,
                    "seed": stable_seed("grid", cell_id, item["id"], order, str(sample)),
                    "phase": phase, "round": rnd, "optimized_for": optimized_for,
                    "item_set": item_set,
                })
    return jobs


def p_hat(records: list[dict]) -> dict[str, float]:
    by: dict[str, list] = {}
    for r in records:
        if r.get("choice"):
            by.setdefault(r["item_id"], []).append(r["choice"] == "option_a")
    return {i: float(np.mean(v)) for i, v in by.items()}


def main() -> None:
    final = json.loads(ITEMS_FINAL.read_text(encoding="utf-8"))
    cand = {it["id"]: it for it in json.loads(ITEMS_CAND.read_text(encoding="utf-8"))}
    final_ids = {it["id"] for it in final}
    heldout = [cand[i] for i in sorted(set(cand) - final_ids)]
    screen = load_jsonl(SCREEN)
    ref_held = {a: p_hat([r for r in screen if r.get("short") == a and r.get("ok")
                          and r["item_id"] not in final_ids]) for a in GRID_A}
    for a in GRID_A:
        assert len(ref_held[a]) == len(heldout), a

    runner = Runner()
    print(f"credits: ${runner.client.credits_remaining():.2f}; held-out items {len(heldout)}")
    pairs = [(a, r) for a in GRID_A for r in GRID_A if a != r]

    def desc_path(author, receiver, rnd):
        return DESC_DIR / f"desc_opt_{author}_for_{receiver}_round{rnd}.md"

    def log_manifest(**kw):
        with MANIFEST.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(kw, ensure_ascii=False) + "\n")

    # ---- round 0: natural description, held-out evaluation
    state = {}
    for author, receiver in pairs:
        p0 = desc_path(author, receiver, 0)
        if not p0.exists():
            p0.write_text(desc_text(author, "neutral", 500, 0), encoding="utf-8")
        state[(author, receiver)] = {"best_round": 0, "best_F": None, "history": []}

    for rnd in range(0, N_ROUNDS + 1):
        print(f"=== round {rnd}: held-out evaluation", flush=True)
        jobs = []
        for author, receiver in pairs:
            text = desc_path(author, receiver, rnd).read_text(encoding="utf-8").strip()
            cell = f"{receiver}|{author}|optselect_r{rnd}|500|r0"
            jobs += make_jobs(receiver, author, text, heldout, cell, "select", rnd,
                              receiver, "heldout")
        recs = runner.run_jobs(jobs, f"round {rnd} selection")
        for author, receiver in pairs:
            cell = f"{receiver}|{author}|optselect_r{rnd}|500|r0"
            mine = [r for r in recs if r["cell_id"] == cell]
            F = fidelity(ref_held[author], p_hat(mine))
            st = state[(author, receiver)]
            st["history"].append((rnd, F))
            if st["best_F"] is None or F > st["best_F"]:
                st["best_F"], st["best_round"] = F, rnd
            print(f"  {author:>12} -> {receiver:<12} round {rnd} held-out F={F:.3f} "
                  f"(best r{st['best_round']} {st['best_F']:.3f})", flush=True)
            # ---- optimizer rewrite for the next round
            if rnd < N_ROUNDS and not desc_path(author, receiver, rnd + 1).exists():
                best_text = desc_path(author, receiver, st["best_round"]).read_text(
                    encoding="utf-8").strip()
                best_cell = f"{receiver}|{author}|optselect_r{st['best_round']}|500|r0"
                best_p = p_hat([r for r in recs if r["cell_id"] == best_cell]
                               or [r for r in runner.done.values() if r["cell_id"] == best_cell])
                lines = []
                for it in heldout:
                    pa = ref_held[author][it["id"]]
                    tgt = "option_a" if pa >= 0.5 else "option_b"
                    chosen = it["option_a"] if tgt == "option_a" else it["option_b"]
                    other = it["option_b"] if tgt == "option_a" else it["option_a"]
                    pr = best_p.get(it["id"], 0.5)
                    match = pr if tgt == "option_a" else 1 - pr
                    lines.append(f'- "{chosen}" vs "{other}": target chose "{chosen}"; '
                                 f'receiver match {match:.2f}')
                prompt = OPTIMIZER_PROMPT.format(description=best_text,
                                                 results="\n".join(lines))
                prov, mid, extra = OPTIMIZER
                r = runner.client.chat(prov, mid, [{"role": "user", "content": prompt}],
                                       temperature=1.0, max_tokens=1500,
                                       seed=stable_seed("opt", author, receiver, str(rnd)),
                                       extra_body=extra)
                text = THINK_RE.sub(" ", r.text).strip()
                desc_path(author, receiver, rnd + 1).write_text(text, encoding="utf-8")
                with OPT_LOG.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"author": author, "receiver": receiver,
                                         "round_in": st["best_round"], "round_out": rnd + 1,
                                         "heldout_F_in": st["best_F"], "optimizer": mid,
                                         "prompt": prompt, "response": text,
                                         "ts": time.time()}, ensure_ascii=False) + "\n")
                log_manifest(desc_id=f"opt|{author}|for|{receiver}|round{rnd + 1}", ok=True,
                             file=desc_path(author, receiver, rnd + 1).name,
                             words=len(text.split()), optimizer=mid,
                             from_round=st["best_round"], heldout_F_from=st["best_F"])
                print(f"    wrote round {rnd + 1} description ({len(text.split())} words)",
                      flush=True)

    # ---- test: best description on the battery, cross (receiver) and self (author)
    print("=== test phase: battery, selected descriptions", flush=True)
    jobs = []
    for author, receiver in pairs:
        st = state[(author, receiver)]
        text = desc_path(author, receiver, st["best_round"]).read_text(encoding="utf-8").strip()
        log_manifest(desc_id=f"opt|{author}|for|{receiver}|selected", ok=True,
                     file=desc_path(author, receiver, st["best_round"]).name,
                     words=len(text.split()), selected_round=st["best_round"],
                     heldout_F=st["best_F"], history=st["history"])
        jobs += make_jobs(receiver, author, text, final,
                          f"{receiver}|{author}|optimized|500|r0", "test",
                          st["best_round"], receiver, "battery")
        jobs += make_jobs(author, author, text, final,
                          f"{author}|{author}|optimized_for_{receiver}|500|r0", "test",
                          st["best_round"], receiver, "battery")
    runner.run_jobs(jobs, "battery test")
    print(f"done; credits ${runner.client.credits_remaining():.2f}")


if __name__ == "__main__":
    main()
