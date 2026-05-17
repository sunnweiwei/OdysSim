"""
Pangram text analysis client.

Example (equivalent to the provided HTTP request):

    POST https://text.api.pangram.com HTTP/1.1
    Content-Type: application/json
    x-api-key: your_api_key_here

    {"text": "The text to analyze"}

This module intentionally avoids hard dependencies (e.g. requests) and uses
Python stdlib `urllib` so it works in minimal environments.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://text.api.pangram.com/v3"


@dataclass(frozen=True)
class PangramAPIError(RuntimeError):
    message: str
    status_code: int = -1
    response_text: str = ""

    def __str__(self) -> str:  # pragma: no cover
        extra = f" (status={self.status_code})" if self.status_code is not None else ""
        if self.response_text:
            return f"{self.message}{extra}: {self.response_text}"
        return f"{self.message}{extra}"


def analyze_text(
    text: str,
    *,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """
    Analyze a piece of text via Pangram's `text.api.pangram.com`.

    - **text**: text to analyze
    - **api_key**: Pangram API key; defaults to env var `PANGRAM_API_KEY`
    - **base_url**: service base URL (default: https://text.api.pangram.com)
    - **timeout_s**: network timeout in seconds

    Returns parsed JSON response as a dict.
    """
    text = str(text or "")
    api_key = (api_key or os.environ.get("PANGRAM_API_KEY") or "").strip()
    if not api_key:
        raise PangramAPIError(
            f"Missing API key. Pass api_key=... or set env var PANGRAM_API_KEY",
            status_code=-1,
        )

    url = base_url.rstrip("/")
    payload = json.dumps({"text": text}).encode("utf-8")
    req = Request(
        url=url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-api-key": api_key,
            "User-Agent": "user-sim-data/ai_detector",
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=float(timeout_s)) as resp:
            raw = resp.read() or b""
            body = raw.decode("utf-8", errors="replace").strip()
            if not body:
                return {}
            try:
                obj = json.loads(body)
                if isinstance(obj, dict):
                    return obj
                return {"result": obj}
            except json.JSONDecodeError:
                return {"raw_response": body}
    except HTTPError as e:
        try:
            err_body = (e.read() or b"").decode("utf-8", errors="replace").strip()
        except Exception:
            err_body = ""
        raise PangramAPIError(
            "Pangram API request failed",
            status_code=int(getattr(e, "code", -1) or -1),
            response_text=err_body,
        ) from e
    except URLError as e:
        raise PangramAPIError(
            "Pangram API request failed (network error)",
            status_code=-1,
            response_text=str(getattr(e, "reason", e)),
        ) from e


def _main(argv: list[str]) -> int:  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description="Analyze text via Pangram API")
    p.add_argument("--text", type=str, default="What are some arguments people use against adoption by gay individuals, and what are the counterarguments?", help="Text to analyze. If empty, read from stdin.")
    p.add_argument("--api-key", type=str, default="", help=f"API key (or set PANGRAM_API_KEY).")
    p.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="Service base URL.")
    p.add_argument("--timeout-s", type=float, default=30.0, help="Request timeout in seconds.")
    args = p.parse_args(argv)

    text = args.text
    if not text:
        text = sys.stdin.read()

    try:
        result = analyze_text(
            text,
            api_key=(args.api_key or None),
            base_url=args.base_url,
            timeout_s=args.timeout_s,
        )
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return 0
    except PangramAPIError as e:
        sys.stderr.write(str(e).rstrip() + "\n")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main(sys.argv[1:]))