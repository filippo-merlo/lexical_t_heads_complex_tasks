"""GPT-5.5 through the user's Azure OpenAI deployment (chat completions).

The key and endpoint come from ``AZURE_OPENAI_API_KEY`` and ``AZURE_OPENAI_ENDPOINT``
(``~/.azure_openai_key``); neither is ever written to the repository.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

API_VERSION = "2024-12-01-preview"
DEPLOYMENT = "gpt-5.5"


def client_and_url(deployment: str = DEPLOYMENT):
    import httpx

    key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not key or not endpoint:
        sys.exit("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set (source ~/.azure_openai_key)")
    url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions?api-version={API_VERSION}"
    return httpx.Client(headers={"api-key": key}, timeout=600), url


def chat(client, url: str, prompt: str, attempts: int = 8) -> dict:
    """One user message; returns ``{"content", "model", "response_id"}``. Retries 429 and 5xx."""
    body = {"messages": [{"role": "user", "content": prompt}], "max_completion_tokens": 16384}
    for attempt in range(attempts):
        wait = 2 ** attempt
        try:
            response = client.post(url, json=body)
        except Exception as error:  # network errors: retry
            if attempt == attempts - 1:
                raise
            print(f"retry ({type(error).__name__})", file=sys.stderr)
        else:
            if response.status_code == 200:
                data = response.json()
                return {"content": data["choices"][0]["message"]["content"],
                        "model": data["model"], "response_id": data["id"]}
            try:
                message = response.json()["error"]["message"]
            except (ValueError, KeyError, TypeError):
                message = response.text[:500]
            if response.status_code != 429 and response.status_code < 500 or attempt == attempts - 1:
                raise RuntimeError(f"HTTP {response.status_code}: {message}")
            wait = max(wait, float(response.headers.get("retry-after", 0)))
            print(f"retry (HTTP {response.status_code})", file=sys.stderr)
        time.sleep(wait)


def json_list(content: str) -> list[str]:
    """The JSON list of strings in a reply, with or without a code fence."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", content, re.S)
    value = json.loads(fenced.group(1) if fenced else content)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError("reply is not a JSON list of strings")
    return value
