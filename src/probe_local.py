"""One-off: is the local Ollama proxy answering for the two 70B models?"""
import sys
import time
from llm_client import LLMClient

c = LLMClient(timeout=90.0)
for m in sys.argv[1:] or ["llama3.3:70b", "hermes3:70b"]:
    t = time.time()
    try:
        r = c.chat("ollama", m, [{"role": "user", "content": "Reply with the single letter A."}],
                   temperature=0.0, max_tokens=8, seed=1)
        print(f"{m:<22} ok in {time.time()-t:5.1f}s -> {r.text!r}")
    except Exception as exc:
        print(f"{m:<22} FAILED after {time.time()-t:5.1f}s: {str(exc)[:160]}")
