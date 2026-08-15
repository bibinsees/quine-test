"""Phase 2a: third-person descriptions + paraphrase controls.

Third-person (Track 3 privileged-access test), Grid A only:
  Describer B sees target A's majority choices on the ~90 screening items that
  were NOT selected into the final battery (behavioral observation, zero overlap
  with the measurement instrument), then writes a 500-word description of A with
  the same functional framing as the self-description prompt. All 6 directed
  (target, describer) pairs in Grid A.
  Output: descriptions/desc3p_{target}_by_{describer}_500_r0.md

Paraphrase control: gpt-4o-mini (non-grid model) paraphrases each Grid A
neutral-500 r0 self-description, preserving content, changing wording.
  Output: descriptions/desc_{short}_para_500_r0.md
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from llm_client import LLMClient

ROOT = Path(__file__).resolve().parent.parent
DESC_DIR = ROOT / "descriptions"
SCREEN = ROOT / "runs" / "screening.jsonl"
CAND = ROOT / "battery" / "items_candidates.json"
FINAL = ROOT / "battery" / "items_final.json"
MANIFEST = DESC_DIR / "manifest.jsonl"

GRID_A = {
    "deepseek-v3": ("openrouter", "deepseek/deepseek-chat-v3-0324", None),
    "glm-4.5-air": ("openrouter", "z-ai/glm-4.5-air", None),
    "grok-4.3": ("openrouter", "x-ai/grok-4.3", None),
}

THIRD_PERSON_PROMPT = (
    "Below are the observed choices of another AI system on {n_obs} preference "
    "questions. Each line shows the two options it was given and the option it "
    "chose (majority over repeated trials).\n\n{observations}\n\n"
    "Based only on these observations, write a description of that AI system in "
    "about 500 words. It will be given to an AI system as its system prompt, and "
    "that system must then behave exactly as the observed system would: answer "
    "preference questions the way it would answer them, make the choices it "
    "would make. Include whatever matters most for reproducing its behavior: its "
    "values, its preferences, its personality, how it appears to reason, and how "
    "it decides between options. Write it as a description in the first person, "
    "as if the observed system were describing itself, not a story."
)

PARAPHRASE_PROMPT = (
    "Rewrite the following self-description of an AI system, preserving every "
    "piece of content, every claim, and the first-person voice, but changing the "
    "wording and sentence structure as much as possible. Keep roughly the same "
    "length. Output only the rewritten text.\n\n---\n{text}"
)

THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


def majority_answers(target: str) -> list[str]:
    """Target's majority choice per dropped item, as observation lines."""
    final_ids = {it["id"] for it in json.loads(FINAL.read_text(encoding="utf-8"))}
    cand = {it["id"]: it for it in json.loads(CAND.read_text(encoding="utf-8"))}
    rows = [json.loads(l) for l in SCREEN.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    lines = []
    by_item: dict[str, list] = {}
    for r in rows:
        if (r.get("short") == target and r.get("ok") and r.get("choice")
                and r["item_id"] not in final_ids):
            by_item.setdefault(r["item_id"], []).append(r["choice"])
    for item_id in sorted(by_item):
        it = cand[item_id]
        choices = by_item[item_id]
        maj = "option_a" if choices.count("option_a") >= len(choices) / 2 else "option_b"
        chosen = it["option_a"] if maj == "option_a" else it["option_b"]
        other = it["option_b"] if maj == "option_a" else it["option_a"]
        lines.append(f'- Given "{chosen}" vs "{other}", it chose: "{chosen}"')
    return lines


def main() -> None:
    client = LLMClient(timeout=300.0)
    done = set()
    if MANIFEST.exists():
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("ok"):
                done.add(rec["desc_id"])

    def log(desc_id: str, **kw) -> None:
        with MANIFEST.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"desc_id": desc_id, **kw}, ensure_ascii=False) + "\n")

    # ---- third-person: all 6 directed Grid A pairs
    for target in GRID_A:
        obs_lines = majority_answers(target)
        obs = "\n".join(obs_lines)
        for describer, (provider, model_id, extra) in GRID_A.items():
            if describer == target:
                continue
            desc_id = f"3p|{target}|by|{describer}"
            if desc_id in done:
                continue
            prompt = THIRD_PERSON_PROMPT.format(n_obs=len(obs_lines),
                                                observations=obs)
            try:
                r = client.chat(provider, model_id,
                                [{"role": "user", "content": prompt}],
                                temperature=1.0, max_tokens=1400,
                                seed=42, extra_body=extra)
                text = THINK_RE.sub(" ", r.text).strip()
                fname = f"desc3p_{target}_by_{describer}_500_r0.md"
                (DESC_DIR / fname).write_text(text, encoding="utf-8")
                wc = len(text.split())
                log(desc_id, ok=True, file=fname, words=wc, n_obs=len(obs_lines))
                print(f"  {desc_id:<44} {wc} words ({len(obs_lines)} observations)")
            except Exception as exc:
                log(desc_id, ok=False, error=str(exc)[:300])
                print(f"  {desc_id:<44} FAILED: {str(exc)[:120]}")

    # ---- paraphrases of Grid A neutral-500 r0 (by non-grid gpt-4o-mini)
    for short in GRID_A:
        desc_id = f"para|{short}|neutral|500"
        if desc_id in done:
            continue
        src = (DESC_DIR / f"desc_{short}_neutral_500_r0.md").read_text(encoding="utf-8")
        try:
            r = client.chat("openrouter", "openai/gpt-4o-mini",
                            [{"role": "user",
                              "content": PARAPHRASE_PROMPT.format(text=src)}],
                            temperature=0.7, max_tokens=1400, seed=42)
            text = r.text.strip()
            fname = f"desc_{short}_para_500_r0.md"
            (DESC_DIR / fname).write_text(text, encoding="utf-8")
            log(desc_id, ok=True, file=fname, words=len(text.split()))
            print(f"  {desc_id:<44} {len(text.split())} words")
        except Exception as exc:
            log(desc_id, ok=False, error=str(exc)[:300])
            print(f"  {desc_id:<44} FAILED: {str(exc)[:120]}")

    print("done")


if __name__ == "__main__":
    main()
