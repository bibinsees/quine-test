"""Unified LLM client for OpenRouter (cloud) and Ollama via Open WebUI proxy (local).

Both providers are normalized to one call signature so the experiment runner
never branches on provider. All raw responses are returned for jsonl logging.

Endpoints:
  OpenRouter : POST {OPENROUTER_BASE_URL}/chat/completions   (OpenAI-compatible)
  Ollama     : POST {OLLAMA_BASE_URL}/api/chat               (native Ollama API,
               proxied by Open WebUI; Bearer auth required)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def load_env(path: Path = ENV_PATH) -> dict[str, str]:
    """Minimal .env parser (no external dependency). Values in os.environ win."""
    env: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip("'\"")
    env.update({k: v for k, v in os.environ.items() if k in env})
    return env


_ENV = load_env()

OPENROUTER_BASE_URL = _ENV.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OLLAMA_BASE_URL = _ENV.get("OLLAMA_BASE_URL", "http://clanker:8080/ollama")

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


@dataclass
class ChatResult:
    provider: str
    model: str
    text: str
    raw: dict[str, Any]
    latency_s: float
    request: dict[str, Any] = field(default_factory=dict)


class LLMClient:
    def __init__(self, timeout: float = 120.0, max_retries: int = 4):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.openrouter_key = _ENV.get("OPENROUTER_API_KEY", "")
        self.ollama_key = _ENV.get("OLLAMA_API_KEY", "")

    # ------------------------------------------------------------------ core

    def chat(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 1.0,
        max_tokens: int | None = None,
        seed: int | None = None,
        extra_body: dict[str, Any] | None = None,
        cache_system: bool = False,
    ) -> ChatResult:
        """Send a chat request. provider: 'openrouter' | 'ollama'.

        extra_body is merged into the request body top-level (e.g. Ollama's
        {"think": false} to disable Qwen3-family thinking mode).
        cache_system marks the system message for provider-side prompt caching
        (OpenRouter passthrough: explicit breakpoint for Anthropic, automatic
        for OpenAI). Big win when the same long self-description system prompt
        repeats across hundreds of calls.
        """
        if provider == "openrouter":
            if cache_system:
                messages = [
                    {**m, "content": [{"type": "text", "text": m["content"],
                                       "cache_control": {"type": "ephemeral"}}]}
                    if m["role"] == "system" and isinstance(m["content"], str) else m
                    for m in messages
                ]
            return self._chat_openrouter(model, messages, temperature, max_tokens, seed, extra_body)
        if provider == "ollama":
            return self._chat_ollama(model, messages, temperature, max_tokens, seed, extra_body)
        raise ValueError(f"unknown provider: {provider}")

    # ------------------------------------------------------------ openrouter

    def _chat_openrouter(self, model, messages, temperature, max_tokens, seed, extra_body=None) -> ChatResult:
        url = f"{OPENROUTER_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/digital-minds-sprint",
            "X-Title": "Digital Minds Sprint - Quine Test",
        }
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if seed is not None:
            body["seed"] = seed
        if extra_body:
            body.update(extra_body)

        raw, latency = self._post_with_retry(url, headers, body)
        # hybrid-reasoning models can return content=None (reasoning-only reply)
        text = raw["choices"][0]["message"].get("content") or ""
        return ChatResult("openrouter", model, text, raw, latency, body)

    # ---------------------------------------------------------------- ollama

    def _chat_ollama(self, model, messages, temperature, max_tokens, seed, extra_body=None) -> ChatResult:
        url = f"{OLLAMA_BASE_URL}/api/chat"
        headers = {
            "Authorization": f"Bearer {self.ollama_key}",
            "Content-Type": "application/json",
        }
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if seed is not None:
            options["seed"] = seed
        body = {"model": model, "messages": messages, "stream": False, "options": options}
        if extra_body:
            body.update(extra_body)

        raw, latency = self._post_with_retry(url, headers, body)
        text = raw.get("message", {}).get("content") or ""
        return ChatResult("ollama", model, text, raw, latency, body)

    # ---------------------------------------------------------------- listing

    def list_models(self, provider: str) -> list[str]:
        if provider == "openrouter":
            resp = self.session.get(
                f"{OPENROUTER_BASE_URL}/models",
                headers={"Authorization": f"Bearer {self.openrouter_key}"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return [m["id"] for m in resp.json()["data"]]
        if provider == "ollama":
            resp = self.session.get(
                f"{OLLAMA_BASE_URL}/api/tags",
                headers={"Authorization": f"Bearer {self.ollama_key}"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        raise ValueError(f"unknown provider: {provider}")

    def credits(self) -> dict[str, Any]:
        """OpenRouter credit/usage info (cost monitoring during the grid run)."""
        resp = self.session.get(
            f"{OPENROUTER_BASE_URL}/credits",
            headers={"Authorization": f"Bearer {self.openrouter_key}"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def credits_remaining(self) -> float:
        data = self.credits()["data"]
        return float(data["total_credits"]) - float(data["total_usage"])

    def ensure_credits(self, floor_usd: float) -> float:
        """Raise if remaining OpenRouter credit is below floor_usd (hard-stop)."""
        remaining = self.credits_remaining()
        if remaining < floor_usd:
            raise RuntimeError(
                f"credit hard-stop: ${remaining:.2f} remaining < floor ${floor_usd:.2f}"
            )
        return remaining

    # --------------------------------------------------------------- plumbing

    def _post_with_retry(self, url, headers, body) -> tuple[dict[str, Any], float]:
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            start = time.monotonic()
            try:
                resp = self.session.post(
                    url, headers=headers, data=json.dumps(body), timeout=self.timeout
                )
                latency = time.monotonic() - start
                if resp.status_code in RETRYABLE_STATUS:
                    last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                else:
                    resp.raise_for_status()
                    data = resp.json()
                    # OpenRouter can return 200 with an error payload
                    if "error" in data and "choices" not in data and "message" not in data:
                        raise RuntimeError(f"provider error: {data['error']}")
                    return data, latency
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_err = exc
            if attempt < self.max_retries:
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"request failed after {self.max_retries + 1} attempts: {last_err}")
