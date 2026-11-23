"""Is the friend's LLM server answering? Sends it one short question and prints the answer.

Run from the repo root:  python scripts/check_llm_server.py

Reads PP_LLM_SERVER_URL and PP_LLM_SERVER_KEY from the environment or .env. Never prints the
address or the key. Exits 0 when the server answers, 1 when it doesn't.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "gemma-31b"  # config/models.yaml: selfhosted-gemma-4-31b
TIMEOUT_S = 90  # what the nightly allows one request


def setting(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == name:
                return value.strip().strip("\"'")
    sys.exit(f"{name} is not set (environment or .env)")


def main() -> int:
    url = setting("PP_LLM_SERVER_URL").rstrip("/") + "/chat/completions"
    key = setting("PP_LLM_SERVER_KEY")
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with one word: ready"}],
        "max_tokens": 512,
        "temperature": 0,
    }
    request = urllib.request.Request(  # noqa: S310 (the address comes from .env)
        url,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:  # noqa: S310
            data = json.load(response)
    except urllib.error.HTTPError as error:
        print(f"No answer: HTTP {error.code} after {time.monotonic() - start:.1f} s")
        return 1
    except (urllib.error.URLError, OSError) as error:
        reason = getattr(error, "reason", error)  # the type only: a message can name the address
        print(f"No answer: {type(reason).__name__} after {time.monotonic() - start:.1f} s")
        return 1
    except ValueError:
        print("No answer: the reply wasn't JSON (a proxy's or tunnel's page?)")
        return 1
    seconds = time.monotonic() - start
    try:
        text = str(data["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        print(f"No answer: the reply after {seconds:.1f} s isn't an OpenAI-style answer")
        return 1
    if not text.strip():
        print(f"No answer: the server replied with nothing after {seconds:.1f} s")
        return 1
    print(f"Answered in {seconds:.1f} s: {' '.join(text.split())[:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
