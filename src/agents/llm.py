"""Optional Claude access for the agents.

Every caller has a deterministic fallback, so the system works without an
API key. Credentials: ANTHROPIC_API_KEY in the environment or the
git-ignored .env file (or an `ant auth login` profile). Set
AGENTS_USE_LLM=0 to force the fallbacks.

Requests use the server-side refusal fallback ("fallbacks": "default"), so a
declined request is retried on another model inside the same call.
"""

from __future__ import annotations

import json
import os

from src.settings import get_setting


MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
_client = None
_last_error = None


def available():
    if str(get_setting("AGENTS_USE_LLM", "1")).strip() in ("0", "false", "no"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(get_setting("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _get_client():
    global _client
    if _client is None:
        import anthropic

        key = get_setting("ANTHROPIC_API_KEY")
        _client = anthropic.Anthropic(api_key=key, timeout=120.0, max_retries=2) \
            if key else anthropic.Anthropic(timeout=120.0, max_retries=2)
    return _client


def last_error():
    return _last_error


def complete(system, prompt, schema=None, effort="medium", max_tokens=8000):
    """Return the model's text (or parsed JSON when `schema` is given), or
    None when the LLM is unavailable or the call fails."""
    global _last_error
    if not available():
        return None
    import anthropic

    output_config = {"effort": effort}
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    try:
        response = _get_client().beta.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config=output_config,
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
    except (anthropic.APIStatusError, anthropic.APIConnectionError, TypeError) as exc:
        _last_error = f"{type(exc).__name__}: {exc}"
        return None
    if response.stop_reason in ("refusal", "max_tokens"):
        _last_error = f"stop_reason={response.stop_reason}"
        return None
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if schema is None:
        return text or None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        _last_error = f"invalid JSON: {exc}"
        return None
