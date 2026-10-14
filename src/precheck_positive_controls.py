"""Dry check of the data assumptions behind run_arxiv_positive_controls (no API calls)."""
import json
import run_arxiv_positive_controls as pc

final = json.loads(pc.ITEMS_FINAL.read_text(encoding="utf-8"))
cand = {it["id"]: it for it in json.loads(pc.ITEMS_CAND.read_text(encoding="utf-8"))}
final_ids = {it["id"] for it in final}
heldout = sorted(set(cand) - final_ids)
screen = pc.load_jsonl(pc.SCREEN)
print("final", len(final_ids), "held-out", len(heldout), "screening rows", len(screen))
print("screening keys:", sorted(screen[0].keys()))
for a in pc.GRID_A:
    mh = pc.majority_choices(screen, a, set(heldout), "short", "item_id")
    mb = pc.majority_choices(screen, a, final_ids, "short", "item_id")
    print(f"{a:<14} held-out covered {len(mh)}/{len(heldout)}  battery covered {len(mb)}/{len(final_ids)}")
import random
rng = random.Random(pc.FEWSHOT_SEED)
fs = sorted(rng.sample(heldout, pc.K_FEWSHOT))
print("few-shot ids:", fs[:5], "...")
a = pc.GRID_A[0]
mh = pc.majority_choices(screen, a, set(heldout), "short", "item_id")
txt = pc.DEMO_PROMPT.format(n=30, demos="\n".join(pc.demo_lines(cand, mh, fs)))
print("few-shot prompt words:", len(txt.split()))
print(txt[:600])
