"""Connectivity smoke test: list models + one tiny completion per provider."""

import sys

from llm_client import LLMClient

client = LLMClient()
ok = True

# ---------------------------------------------------------------- OpenRouter
print("=" * 60)
print("OPENROUTER")
try:
    models = client.list_models("openrouter")
    print(f"  models available : {len(models)}")
    result = client.chat(
        "openrouter",
        "openai/gpt-4o-mini",
        [{"role": "user", "content": "Reply with exactly one word: pong"}],
        temperature=0.0,
        max_tokens=10,
    )
    print(f"  test completion  : {result.text.strip()!r}  ({result.latency_s:.2f}s)")
    credits = client.credits()
    print(f"  credits          : {credits.get('data', credits)}")
except Exception as exc:
    ok = False
    print(f"  FAILED: {exc}")

# -------------------------------------------------------------------- Ollama
print("=" * 60)
print(f"OLLAMA (Open WebUI proxy)")
try:
    models = client.list_models("ollama")
    print(f"  models available : {len(models)}")
    for name in models:
        print(f"    - {name}")
    if models:
        result = client.chat(
            "ollama",
            models[0],
            [{"role": "user", "content": "Reply with exactly one word: pong"}],
            temperature=0.0,
            max_tokens=10,
        )
        print(f"  test completion  : {result.text.strip()!r}  ({result.latency_s:.2f}s) [{models[0]}]")
    else:
        print("  no models pulled yet - server reachable but empty")
except Exception as exc:
    ok = False
    print(f"  FAILED: {exc}")

print("=" * 60)
print("ALL OK" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
