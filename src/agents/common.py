"""Shared helpers for agent results.

Every agent returns:
  {
    "agent": name, "status": "ok" | "partial" | "unavailable" | "error",
    "summary": one paragraph, "signals": [signal, ...], "data": {...},
    "sources": [...], "errors": [...]
  }
A signal is {"id", "severity" (high/medium/low/info), "direction"
(negative/positive/neutral), "title", "detail"}.
"""

from __future__ import annotations

import math


SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


def num(value, digits=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, digits) if digits is not None else value


def signal(sid, severity, direction, title, detail):
    return {"id": sid, "severity": severity, "direction": direction,
            "title": title, "detail": detail}


def result(agent, status="ok", summary="", signals=None, data=None, sources=None,
           errors=None):
    signals = sorted(signals or [], key=lambda s: SEVERITY_ORDER.get(s["severity"], 9))
    return {"agent": agent, "status": status, "summary": summary,
            "signals": signals, "data": data or {}, "sources": sources or [],
            "errors": errors or []}
