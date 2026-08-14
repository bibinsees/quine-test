"""One-off: probe candidate OpenAI flagship IDs + kimi reasoning-off, 1 call each."""
from llm_client import LLMClient

client = LLMClient(timeout=120, max_retries=1)
MSGS = [{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Reply with exactly one letter: A or B. Pick one."}]

CANDIDATES = [
    ("openai/gpt-5.2-chat", None, 16),
    ("openai/gpt-5.5", {"reasoning": {"effort": "minimal"}}, 64),
    ("openai/gpt-5.4", {"reasoning": {"effort": "minimal"}}, 64),
    ("openai/gpt-5.2", {"reasoning": {"effort": "minimal"}}, 64),
    ("moonshotai/kimi-k2.5", {"reasoning": {"enabled": False}}, 16),
    ("moonshotai/kimi-k2.5", None, 256),
]

for model, extra, mt in CANDIDATES:
    try:
        r = client.chat("openrouter", model, MSGS, temperature=1.0,
                        max_tokens=mt, extra_body=extra)
        usage = r.raw.get("usage", {})
        print(f"OK   {model:<28} extra={extra} -> {r.text[:40]!r} "
              f"(tok in/out {usage.get('prompt_tokens')}/{usage.get('completion_tokens')})")
    except Exception as exc:
        print(f"FAIL {model:<28} extra={extra} -> {str(exc)[:140]}")
