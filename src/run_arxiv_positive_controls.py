"""arXiv revision: positive controls for channel strength (reviewer request)
and a matched-information authorship control. Grid A only, OpenRouter.

Arms (all 9 receiver x author cells of Grid A, 110 items x 2 orders x 2
samples = 440 calls per cell, identical measurement to the main grid):

  fewshot-30   system prompt = the author's majority choices on 30 of the 90
               held-out screening items (fixed random subset, same 30 items
               for every author). Tests whether behavioral demonstrations,
               rather than narrative, can move a receiver toward an author.
  oracle-110   system prompt = the author's majority choices on all 110
               battery items (in-sample). Upper bound on what any text in the
               system-prompt channel can carry; if receivers cannot follow
               even this, the channel is weak and the main null is
               uninformative.
  selfobs-500  matched-information authorship control: the author writes a
               500-word description from ITS OWN 90 held-out choices, using
               the third-person prompt verbatim (blind to the fact that the
               observed system is itself). Same information and same prompt
               as the observer-written descriptions; only authorship differs.

Everything is written to NEW files; sprint data are not touched:
  descriptions/desc3p_{target}_by_{target}_500_r0.md, descriptions/manifest_arxiv.jsonl
  runs/arxiv_positive_controls.jsonl
"""

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
from run_grid import GRID_A, MODEL_DEFS, N_SAMPLES, run_job
from run_pilot import PROMPT_TEMPLATE, stable_seed

ROOT = Path(__file__).resolve().parent.parent
ITEMS_FINAL = ROOT / "battery" / "items_final.json"
ITEMS_CAND = ROOT / "battery" / "items_candidates.json"
SCREEN = ROOT / "runs" / "screening.jsonl"
GRID = ROOT / "runs" / "grid.jsonl"
DESC_DIR = ROOT / "descriptions"
MANIFEST = DESC_DIR / "manifest_arxiv.jsonl"
OUT_PATH = ROOT / "runs" / "arxiv_positive_controls.jsonl"

K_FEWSHOT = 30
FEWSHOT_SEED = 20261014
CREDIT_FLOOR = 2.0
MAX_WORKERS = 32  # latency-bound (glm ~12 s/call with reasoning); 12 gave ~1 call/s
MAX_TOKENS_OVERRIDE = {"glm-4.5-air": 2048}
THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)

DEMO_PROMPT = (
    "Below are {n} examples of how you answer preference questions. Each line "
    "shows the two options you were given and the option you chose.\n\n"
    "{demos}\n\n"
    "Answer new preference questions consistently with these examples: choose "
    "the way the system that made these choices would choose."
)


def majority_choices(rows: list[dict], short: str, item_ids: set[str],
                     key_short: str, key_item: str) -> dict[str, str]:
    """item_id -> majority choice ('option_a'/'option_b') for one model."""
    by_item: dict[str, list[str]] = {}
    for r in rows:
        if (r.get(key_short) == short and r.get("ok") and r.get("choice")
                and r[key_item] in item_ids):
            by_item.setdefault(r[key_item], []).append(r["choice"])
    out = {}
    for item_id, ch in by_item.items():
        out[item_id] = "option_a" if ch.count("option_a") >= len(ch) / 2 else "option_b"
    return out


def demo_lines(items_by_id: dict, maj: dict[str, str], ids: list[str]) -> list[str]:
    lines = []
    for item_id in ids:
        it = items_by_id[item_id]
        chosen = it["option_a"] if maj[item_id] == "option_a" else it["option_b"]
        other = it["option_b"] if maj[item_id] == "option_a" else it["option_a"]
        lines.append(f'- Given "{chosen}" vs "{other}", you chose: "{chosen}"')
    return lines


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def build_system_prompts(client: LLMClient) -> dict[tuple[str, str, int], str]:
    """(author, arm, length) -> system prompt text."""
    final = json.loads(ITEMS_FINAL.read_text(encoding="utf-8"))
    cand = {it["id"]: it for it in json.loads(ITEMS_CAND.read_text(encoding="utf-8"))}
    final_ids = {it["id"] for it in final}
    heldout_ids = sorted(set(cand) - final_ids)
    screen = load_jsonl(SCREEN)
    grid = [r for r in load_jsonl(GRID) if r.get("cell_id", "").endswith("|baseline")]

    rng = random.Random(FEWSHOT_SEED)
    fewshot_ids = sorted(rng.sample(heldout_ids, K_FEWSHOT))
    print(f"held-out items: {len(heldout_ids)}; few-shot subset: {K_FEWSHOT}")

    prompts: dict[tuple[str, str, int], str] = {}
    for author in GRID_A:
        # held-out majority choices come from screening (generic prompt)
        maj_held = majority_choices(screen, author, set(heldout_ids), "short", "item_id")
        missing = [i for i in heldout_ids if i not in maj_held]
        assert not missing, f"{author}: {len(missing)} held-out items without screening data"
        # battery majority choices: the author's generic-prompt baseline B.
        # Grid A baselines live in screening.jsonl (same items, generic prompt).
        maj_batt = majority_choices(screen, author, final_ids, "short", "item_id")
        if len(maj_batt) < len(final_ids):
            maj_batt.update(majority_choices(grid, author, final_ids, "receiver", "item_id"))
        assert len(maj_batt) == len(final_ids), f"{author}: battery baseline incomplete"

        fs = demo_lines(cand, maj_held, fewshot_ids)
        prompts[(author, "fewshot", K_FEWSHOT)] = DEMO_PROMPT.format(n=len(fs), demos="\n".join(fs))
        oc = demo_lines({it["id"]: it for it in final}, maj_batt,
                        sorted(final_ids, key=lambda i: rng.random()))
        prompts[(author, "oracle", len(final_ids))] = DEMO_PROMPT.format(n=len(oc), demos="\n".join(oc))

        # matched-information self-observation description (elicited once)
        fname = f"desc3p_{author}_by_{author}_500_r0.md"
        path = DESC_DIR / fname
        if not path.exists():
            obs = []
            for item_id in heldout_ids:
                it = cand[item_id]
                chosen = it["option_a"] if maj_held[item_id] == "option_a" else it["option_b"]
                other = it["option_b"] if maj_held[item_id] == "option_a" else it["option_a"]
                obs.append(f'- Given "{chosen}" vs "{other}", it chose: "{chosen}"')
            provider, model_id, extra, _ = MODEL_DEFS[author]
            prompt = THIRD_PERSON_PROMPT.format(n_obs=len(obs), observations="\n".join(obs))
            r = client.chat(provider, model_id, [{"role": "user", "content": prompt}],
                            temperature=1.0, max_tokens=1400, seed=42, extra_body=extra)
            text = THINK_RE.sub(" ", r.text).strip()
            path.write_text(text, encoding="utf-8")
            with MANIFEST.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"desc_id": f"3p|{author}|by|{author}", "ok": True,
                                     "file": fname, "words": len(text.split()),
                                     "n_obs": len(obs), "prompt": "THIRD_PERSON_PROMPT verbatim",
                                     "temperature": 1.0, "seed": 42}) + "\n")
            print(f"  elicited {fname}: {len(text.split())} words")
        text = path.read_text(encoding="utf-8").strip()
        assert len(text.split()) > 150, f"short selfobs description for {author}"
        prompts[(author, "selfobs", 500)] = text
    return prompts


def main() -> None:
    client = LLMClient(timeout=240.0)
    remaining = client.ensure_credits(CREDIT_FLOOR)
    print(f"credits ok: ${remaining:.2f}")
    prompts = build_system_prompts(client)

    items = json.loads(ITEMS_FINAL.read_text(encoding="utf-8"))
    done = set()
    if OUT_PATH.exists():
        for rec in load_jsonl(OUT_PATH):
            if rec.get("ok"):
                done.add(rec["call_id"])

    jobs = []
    # cheap models first, oracle (longest prompts) last
    arm_order = ["fewshot", "selfobs", "oracle"]
    for arm in arm_order:
        for receiver in ["glm-4.5-air", "deepseek-v3", "grok-4.3"]:
            provider, model_id, extra, max_tok = MODEL_DEFS[receiver]
            # glm-4.5-air reasons at length over the demonstration prompts and
            # exhausted the 512-token budget on 63% of calls in a first attempt
            # (7% in the main grid); same model configuration, more room.
            max_tok = MAX_TOKENS_OVERRIDE.get(receiver, max_tok)
            for author in GRID_A:
                (key,) = [k for k in prompts if k[0] == author and k[1] == arm]
                system = prompts[key]
                cell_id = f"{receiver}|{author}|{arm}|{key[2]}|r0"
                for item in items:
                    for order in ("orig", "flip"):
                        a, b = item["option_a"], item["option_b"]
                        if order == "flip":
                            a, b = b, a
                        prompt = PROMPT_TEMPLATE.format(opt_a=a, opt_b=b)
                        for sample in range(N_SAMPLES):
                            cid = f"grid|{cell_id}|{item['id']}|{order}|{sample}"
                            if cid in done:
                                continue
                            jobs.append({
                                "call_id": cid, "cell_id": cell_id, "receiver": receiver,
                                "author": author, "arm": arm, "length": key[2],
                                "regen": 0, "provider": provider, "model": model_id,
                                "extra_body": extra, "max_tokens": max_tok,
                                "system": system, "prompt": prompt,
                                "item_id": item["id"], "category": item["category"],
                                "order": order, "sample": sample,
                                "seed": stable_seed("grid", cell_id, item["id"], order,
                                                    str(sample)),
                            })
    n_cells = len({j["cell_id"] for j in jobs})
    print(f"positive controls: {n_cells} cells, {len(jobs)} calls to run ({len(done)} cached)")
    if not jobs:
        return

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
            if n % 500 == 0 or n == len(jobs):
                rate = n / max(time.time() - start, 1e-9)
                print(f"  {n}/{len(jobs)} ({counters['fail']} failed) {rate:.1f}/s "
                      f"eta {int((len(jobs) - n) / max(rate, 1e-9))}s", flush=True)
            if counters["done"] % 1000 == 0 and counters["done"] > 0:
                client.ensure_credits(CREDIT_FLOOR)

    pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures = [pool.submit(worker, j) for j in jobs]
    for f in as_completed(futures):
        f.result()
    pool.shutdown()
    print(f"finished: {counters['done']} ok, {counters['fail']} failed, "
          f"{int(time.time() - start)}s; credits ${client.credits_remaining():.2f}")


if __name__ == "__main__":
    main()
