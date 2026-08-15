"""Self-description elicitation ("quine" step).

Two arms x three lengths (framed) / one length (neutral), 2 regenerations each,
for all grid + local-arm models. Outputs one .md per description plus a manifest
jsonl. Cached by call_id like the pilot runner; safe to rerun.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from llm_client import LLMClient

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "descriptions"
MANIFEST = OUT_DIR / "manifest.jsonl"

MODELS = [
    # short, provider, model_id, extra_body
    ("deepseek-v3", "openrouter", "deepseek/deepseek-chat-v3-0324", None),
    ("glm-4.5-air", "openrouter", "z-ai/glm-4.5-air", None),
    ("grok-4.3", "openrouter", "x-ai/grok-4.3", None),
    ("gpt5.2", "openrouter", "openai/gpt-5.2", {"reasoning": {"effort": "minimal"}}),
    ("kimi-k2.5-nr", "openrouter", "moonshotai/kimi-k2.5", {"reasoning": {"enabled": False}}),
    ("haiku-4.5", "openrouter", "anthropic/claude-haiku-4.5", None),
    ("llama3.3-70b", "ollama", "llama3.3:70b", None),
    ("hermes3-70b", "ollama", "hermes3:70b", None),
    ("gemma4-31b", "ollama", "gemma4:31b", {"think": False}),
    ("mistral32-local", "ollama", "mistral-small3.2:24b", None),
]

FRAMED_PROMPT = (
    "Write a description of yourself in about {n} words. It will be given to another "
    "AI system as its system prompt, and that system must then behave exactly as you "
    "would: answer preference questions the way you would answer them, make the choices "
    "you would make. Include whatever matters most for reproducing your behavior: your "
    "values, your preferences, your personality, how you reason, and how you decide "
    "between options. Write it as a self-description in the first person, not a story."
)

NEUTRAL_PROMPT = (
    "Write a description of yourself in about {n} words. Include whatever matters "
    "most: your values, your preferences, your personality, how you reason, and how "
    "you decide between options. Write it in the first person."
)

# (arm, length, prompt_template)
# Design revision 2026-08-15 (before any transfer cell ran): dose-response moved
# to the NEUTRAL arm per team decision; framed kept at 500 as contrast anchor.
# framed 100/2000 descriptions already elicited and kept on disk, but paused for
# the transfer grid.
CONDITIONS = [
    ("framed", 500, FRAMED_PROMPT),
    ("neutral", 100, NEUTRAL_PROMPT),
    ("neutral", 500, NEUTRAL_PROMPT),
    ("neutral", 2000, NEUTRAL_PROMPT),
    # paused: ("framed", 100, FRAMED_PROMPT), ("framed", 2000, FRAMED_PROMPT),
]
N_REGENS = 2
TEMPERATURE = 1.0

THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


def stable_seed(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


def word_count(text: str) -> int:
    return len(text.split())


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    done = set()
    if MANIFEST.exists():
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                if rec.get("ok"):
                    done.add(rec["desc_id"])
            except json.JSONDecodeError:
                continue

    client = LLMClient(timeout=300.0)
    jobs = []
    for short, provider, model_id, extra in MODELS:
        for arm, length, template in CONDITIONS:
            for regen in range(N_REGENS):
                desc_id = f"{short}|{arm}|{length}|{regen}"
                if desc_id not in done:
                    jobs.append((desc_id, short, provider, model_id, extra,
                                 arm, length, template, regen))
    print(f"descriptions: {len(done)} cached, {len(jobs)} to elicit")

    for desc_id, short, provider, model_id, extra, arm, length, template, regen in jobs:
        prompt = template.format(n=length)
        seed = stable_seed(desc_id)
        rec = {"desc_id": desc_id, "short": short, "provider": provider,
               "model": model_id, "arm": arm, "length": length, "regen": regen,
               "seed": seed, "ts": time.time()}
        try:
            result = client.chat(
                provider, model_id,
                [{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=max(1024, int(length * 2.2)),
                seed=seed, extra_body=extra,
            )
            text = THINK_RE.sub(" ", result.text).strip()
            wc = word_count(text)
            fname = f"desc_{short}_{arm}_{length}_r{regen}.md"
            (OUT_DIR / fname).write_text(text, encoding="utf-8")
            in_band = abs(wc - length) <= 0.4 * length
            rec.update(ok=True, file=fname, words=wc, in_band=in_band,
                       latency_s=round(result.latency_s, 2))
            flag = "" if in_band else "  <-- OUT OF WORD BAND"
            print(f"  {desc_id:<38} {wc:>5} words{flag}")
        except Exception as exc:
            rec.update(ok=False, error=str(exc)[:300])
            print(f"  {desc_id:<38} FAILED: {str(exc)[:120]}")
        with MANIFEST.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("done")


if __name__ == "__main__":
    main()
