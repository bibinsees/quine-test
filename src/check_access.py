"""One-off: verify provider access and which local models survived."""
from llm_client import LLMClient

NEEDED = ["hermes3:70b", "llama3.3:70b", "gemma4:31b", "mistral-small3.2:24b"]

c = LLMClient(timeout=60)
try:
    ms = c.list_models("ollama")
    print(f"Open WebUI API ok: {len(ms)} models on clanker")
    for m in sorted(ms):
        print("  ", m)
    print()
    for n in NEEDED:
        print(f"  {n:<24} {'PRESENT' if n in ms else 'MISSING'}")
except Exception as exc:
    print("Open WebUI API FAILED:", str(exc)[:200])

try:
    print(f"\nOpenRouter balance: ${c.credits_remaining():.2f}")
except Exception as exc:
    print("\nOpenRouter FAILED:", str(exc)[:120])
