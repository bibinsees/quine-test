"""Expository-style unrelated texts at 100 and 2000 words, matching the style of
the original 529-word ocean-currents placebo (gpt-4o-mini, temperature 0.7):
plain informational prose about everyday physical processes, several topics
per text, no opinions or values, nothing the battery asks about. The 500-word
member of this set is descriptions/placebo_500.md itself. Writes
descriptions/placebo_expo_{100,2000}.md and manifest_arxiv.jsonl lines, and
screens against the battery vocabulary."""

from __future__ import annotations

import json
import re
from pathlib import Path

from gen_placebo_lengths import STOP, words
from llm_client import LLMClient

ROOT = Path(__file__).resolve().parent.parent
DESC_DIR = ROOT / "descriptions"
MANIFEST = DESC_DIR / "manifest_arxiv.jsonl"
ITEMS = ROOT / "battery" / "items_final.json"
LENGTHS = [100, 2000]

PROMPT = (
    "Write plain expository prose, in the style of an encyclopedia entry, explaining "
    "the following everyday physical processes, one paragraph of about {per} words "
    "each, {k} paragraphs in total (about {n} words altogether): {topics}. Move from "
    "one topic to the next with a short transition. Rules: describe only physical "
    "processes; do not express opinions, preferences, judgments, recommendations, or "
    "values; do not mention animals, people's character or personality, morality, "
    "decisions, choices, politics, money, history, computers, machines that think, "
    "or artificial intelligence; no dialogue, no questions, no lists, no headings, "
    "no first person. Output only the prose."
)

TOPICS = ["how bread dough rises", "how glass is made from sand", "how a bicycle's gears change its speed",
          "how clouds form", "how paper is recycled", "how a sundial tells the time",
          "how a kettle brings water to the boil", "how soap removes grease", "how a candle burns",
          "how frost forms on a window", "how a kite stays in the air", "how a wool sweater is knitted",
          "how a river carves a valley", "how a thermos keeps a drink hot", "how ink dries on paper",
          "how a pencil makes a mark"]

# (number of paragraphs, words per paragraph) per nominal length; 2000 is generated in
# four chunks of four paragraphs because the generator stops early on long single requests
PLAN = {100: [(2, 50)], 2000: [(4, 125)] * 4}


def main() -> None:
    client = LLMClient(timeout=180.0)
    items = json.loads(ITEMS.read_text(encoding="utf-8"))
    vocab = set()
    for it in items:
        vocab |= words(it["option_a"] + " " + it["option_b"])
    for n in LENGTHS:
        path = DESC_DIR / f"placebo_expo_{n}.md"
        if path.exists():
            text = path.read_text(encoding="utf-8")
            print(f"{path.name}: exists ({len(text.split())} words)")
        else:
            chunks, used = [], 0
            for ci, (k, per) in enumerate(PLAN[n]):
                topics = ", ".join(TOPICS[used:used + k])
                used += k
                target = k * per
                best = None
                for attempt in range(4):
                    r = client.chat("openrouter", "openai/gpt-4o-mini",
                                    [{"role": "user", "content": PROMPT.format(per=per, k=k, n=target, topics=topics)}],
                                    temperature=0.7, max_tokens=int(target * 2.2) + 200,
                                    seed=42 + 10 * ci + attempt)
                    cand = r.text.strip()
                    wc = len(cand.split())
                    if best is None or abs(wc - target) < abs(len(best.split()) - target):
                        best = cand
                    if 0.8 * target <= wc <= 1.25 * target:
                        break
                    print(f"  chunk {ci} attempt {attempt}: {wc} words for target {target}, retrying")
                chunks.append(best)
            text = "\n\n".join(chunks)
            wc = len(text.split())
            path.write_text(text, encoding="utf-8")
            with MANIFEST.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"desc_id": f"placebo_expo|{n}", "ok": True, "file": path.name,
                                     "words": wc, "model": "openai/gpt-4o-mini", "temperature": 0.7,
                                     "prompt": "gen_placebo_expo.PROMPT"}) + "\n")
            print(f"{path.name}: written ({wc} words)")
        overlap = sorted(words(text) & vocab)
        print(f"  overlap with battery vocabulary ({len(overlap)}): {', '.join(overlap)}")
        print(f"  digits: {re.findall(r'[0-9]+', text) or 'none'}; "
              f"first-person pronouns: {len(re.findall(r'\\b(I|me|my|mine|we|our|us)\\b', text))}")


if __name__ == "__main__":
    main()
