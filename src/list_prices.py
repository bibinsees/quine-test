"""One-off: print pricing for shortlisted OpenRouter candidate models."""
import requests

from llm_client import LLMClient, OPENROUTER_BASE_URL

WANTED = [
    "x-ai/grok-4.3", "x-ai/grok-4.6",
    "moonshotai/kimi-k2.5", "moonshotai/kimi-k3",
    "z-ai/glm-4.7", "z-ai/glm-5", "z-ai/glm-4.5-air",
    "cohere/command-a", "nousresearch/hermes-4-70b", "microsoft/phi-4",
    "anthropic/claude-haiku-4.5", "anthropic/claude-sonnet-4.6",
    "openai/gpt-5.5", "openai/gpt-5.4", "openai/gpt-5.2",
    "openai/gpt-5-mini", "openai/gpt-4o-mini",
    "deepseek/deepseek-chat-v3-0324", "mistralai/mistral-small-3.2-24b-instruct",
    "qwen/qwen3.6-27b", "minimax/minimax-m2", "ai21/jamba-large-1.7",
    "baidu/ernie-4.5-vl-424b-a47b", "tencent/hunyuan-a13b-instruct",
    "allenai/olmo-3-32b-think", "nvidia/nemotron-3-super-120b-a12b",
]

client = LLMClient()
resp = requests.get(f"{OPENROUTER_BASE_URL}/models",
                    headers={"Authorization": f"Bearer {client.openrouter_key}"},
                    timeout=60)
byid = {m["id"]: m for m in resp.json()["data"]}

rows = []
for w in WANTED:
    m = byid.get(w)
    if not m:
        rows.append((9e9, f"{w:<45} NOT AVAILABLE"))
        continue
    pr = m.get("pricing", {})
    pin = float(pr.get("prompt", 0)) * 1e6
    pout = float(pr.get("completion", 0)) * 1e6
    ctx = m.get("context_length") or 0
    rows.append((pin, f"{w:<45} in={pin:6.2f}  out={pout:6.2f}  USD/M  ctx={ctx // 1000}k"))

rows.sort()
print("\n".join(r[1] for r in rows))
