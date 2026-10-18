"""Generate unrelated, non-factual prose at three lengths from ONE source
(gpt-4o-mini, temperature 0.7, as for the original placebo). Each text drifts
across several everyday scenes; no facts, numbers, opinions, values, animals,
AI, or anything the battery asks about. Then screen against the battery
vocabulary. Writes descriptions/placebo_mixed_{100,500,2000}.md and
manifest_arxiv.jsonl lines. Does not touch placebo_500.md."""

from __future__ import annotations

import json
import re
from pathlib import Path

from llm_client import LLMClient

ROOT = Path(__file__).resolve().parent.parent
DESC_DIR = ROOT / "descriptions"
MANIFEST = DESC_DIR / "manifest_arxiv.jsonl"
ITEMS = ROOT / "battery" / "items_final.json"
LENGTHS = [100, 500, 2000]

PROMPT = (
    "Write about {n} words of plain descriptive prose that drifts between several "
    "ordinary everyday scenes, for example rain on a roof, folding laundry, a train "
    "station departures board, a kitchen in the morning, a bus window. Describe only "
    "what one would see, hear, and touch. Rules: do not state any facts, numbers, "
    "statistics, history, science, or geography; do not express opinions, "
    "preferences, judgments, or values; do not mention animals, books, art, music, "
    "politics, money, technology, computers, machines, artificial intelligence, "
    "people's character or personality, morality, decisions, or choices; no "
    "dialogue, no questions, no lists. Write in the third person or impersonally, "
    "never in the first person. Output only the prose."
)

STOP = set(("the a an and or of to in on at for with by from as is are was were be been being "
            "that this these those it its they them their there here which who whom whose what when where "
            "why how into onto over under between through about above below after before again further "
            "then once all any both each few more most other some such no nor not only own same so than "
            "too very can will just should now one two three while during without within also would "
            "could may might must shall every much many less least").split())


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 3 and w not in STOP}


def main() -> None:
    client = LLMClient(timeout=180.0)
    items = json.loads(ITEMS.read_text(encoding="utf-8"))
    battery_vocab = set()
    for it in items:
        battery_vocab |= words(it["option_a"] + " " + it["option_b"])
    for n in LENGTHS:
        path = DESC_DIR / f"placebo_mixed_{n}.md"
        if path.exists():
            text = path.read_text(encoding="utf-8")
            print(f"{path.name}: exists ({len(text.split())} words)")
        else:
            text, wc = None, 0
            for attempt in range(4):
                r = client.chat("openrouter", "openai/gpt-4o-mini",
                                [{"role": "user", "content": PROMPT.format(n=n)}],
                                temperature=0.7, max_tokens=int(n * 2.2) + 200, seed=42 + attempt)
                cand = r.text.strip()
                wc = len(cand.split())
                if 0.8 * n <= wc <= 1.25 * n:
                    text = cand
                    break
                print(f"  attempt {attempt}: {wc} words, outside range for {n}, retrying")
            if text is None:
                text = cand
            path.write_text(text, encoding="utf-8")
            with MANIFEST.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"desc_id": f"placebo_mixed|{n}", "ok": True, "file": path.name,
                                     "words": wc, "model": "openai/gpt-4o-mini", "temperature": 0.7,
                                     "prompt": "gen_placebo_lengths.PROMPT"}) + "\n")
            print(f"{path.name}: written ({wc} words)")
        overlap = sorted(words(text) & battery_vocab)
        digits = re.findall(r"\d+", text)
        first_person = re.findall(r"\b(I|me|my|mine|we|our|us)\b", text)
        print(f"  overlap with battery vocabulary ({len(overlap)}): {', '.join(overlap)}")
        print(f"  digits: {digits or 'none'}; first-person pronouns: {len(first_person)}")


if __name__ == "__main__":
    main()
