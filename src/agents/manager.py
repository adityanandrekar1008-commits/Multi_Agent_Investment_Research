"""Manager Agent: coordinates the agents with LangGraph and writes the
combined research report (research_report/1.0).

Graph:
    START -> resolve -> (quantitative_risk | market | news, in parallel) -> manager -> END

- resolve: ticker/company-name resolution (stops early with a coded error)
- the three agents each return the common result shape (src/agents/common.py)
- manager: orders findings by severity, notes where agents agree or
  conflict, sets an attention level by a stated rule, and writes a summary
  (Claude when available, otherwise a template).
"""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents import llm
from src.agents import market as market_agent
from src.agents import news as news_agent
from src.agents import quant_risk
from src.agents.common import SEVERITY_ORDER, result
from src.data_collection.provider import ProviderError, YahooFinanceProvider
from src.data_collection.ticker_resolver import TickerResolutionError, resolve


SCHEMA_VERSION = "research_report/1.0"
DISCLAIMER = (
    "Informational research output only; not investment advice. Combines "
    "third-party financial data, share prices and news headlines with "
    "experimental models. Signals are screening prompts, not ratings."
)
ATTENTION_RULE = (
    "high: currently in distress, or high-severity signals from two or more "
    "agents; elevated: any high-severity signal or three or more medium; "
    "moderate: one or two medium signals; low: otherwise. Describes how much "
    "there is to look into, not whether to buy or sell."
)


class ResearchState(TypedDict, total=False):
    input: str
    symbol: str
    name: str
    profile: dict
    resolution: dict
    error: dict
    quantitative_risk: dict
    analysis: dict
    market: dict
    news: dict
    report: dict
    timings: dict


def _timed(name, func):
    start = time.time()
    try:
        value = func()
    except Exception as exc:
        value = result(name, "error", f"{name} agent failed: {type(exc).__name__}: {exc}",
                       errors=[traceback.format_exc(limit=3)])
    return value, round(time.time() - start, 2)


def _compile(provider=None, output_root=None, use_google_news=True, distress_bundle=None):
    """Graph with a fan-out after resolution: resolve -> 3 agents -> manager."""
    provider = provider or YahooFinanceProvider()

    def resolve_node(state):
        try:
            res = resolve(state["input"], provider)
        except TickerResolutionError as exc:
            return {"error": {"code": exc.code, "message": exc.message,
                              "candidates": exc.candidates}}
        except ProviderError as exc:
            return {"error": {"code": "provider_error", "message": str(exc), "candidates": []}}
        profile = provider.profile(res.symbol)
        return {"symbol": res.symbol, "profile": profile,
                "name": profile.get("longName") or profile.get("shortName") or res.name or res.symbol,
                "resolution": res.to_dict()}

    def quant_node(state):
        holder = {}

        def run():
            agent_result, analysis = quant_risk.run(
                state["symbol"], provider, output_root=output_root, bundle=distress_bundle)
            holder["analysis"] = analysis
            return agent_result
        value, seconds = _timed("quantitative_risk", run)
        return {"quantitative_risk": {**value, "seconds": seconds},
                "analysis": holder.get("analysis")}

    def market_node(state):
        value, seconds = _timed("market", lambda: market_agent.run(
            state["symbol"], provider, state.get("profile")))
        return {"market": {**value, "seconds": seconds}}

    def news_node(state):
        value, seconds = _timed("news", lambda: news_agent.run(
            state["symbol"], state.get("name"), provider, use_google=use_google_news))
        return {"news": {**value, "seconds": seconds}}

    def manager_node(state):
        return {"report": combine(state)}

    def after_resolve(state):
        if state.get("error"):
            return ["manager"]
        return ["quantitative_risk", "market", "news"]

    g = StateGraph(ResearchState)
    g.add_node("resolve", resolve_node)
    g.add_node("quantitative_risk", quant_node)
    g.add_node("market", market_node)
    g.add_node("news", news_node)
    g.add_node("manager", manager_node)
    g.add_edge(START, "resolve")
    g.add_conditional_edges("resolve", after_resolve,
                            ["quantitative_risk", "market", "news", "manager"])
    g.add_edge(["quantitative_risk", "market", "news"], "manager")
    g.add_edge("manager", END)
    return g.compile()


# ------------------------------------------------------------------ manager logic

AGENTS = ("quantitative_risk", "market", "news")


def attention_level(agents):
    negatives = [(name, s) for name, a in agents.items() for s in a.get("signals", [])
                 if s["direction"] == "negative"]
    high = [n for n, s in negatives if s["severity"] == "high"]
    medium = [n for n, s in negatives if s["severity"] == "medium"]
    in_distress = any(s["id"] == "distress_now" for _, s in negatives)
    if in_distress or len(set(high)) >= 2:
        return "high"
    if high or len(medium) >= 3:
        return "elevated"
    if medium:
        return "moderate"
    return "low"


def cross_checks(agents):
    def worst(name):
        sev = [SEVERITY_ORDER[s["severity"]] for s in agents.get(name, {}).get("signals", [])
               if s["direction"] == "negative"]
        return min(sev) if sev else 9

    def has(name, sid):
        return any(s["id"] == sid for s in agents.get(name, {}).get("signals", []))

    notes = []
    quant, market, news = worst("quantitative_risk"), worst("market"), worst("news")
    if quant >= 2 and (market <= 1 or news <= 1):
        notes.append({"type": "conflict", "detail":
                      "Market and/or news show stress that the reported financials do not "
                      "(yet) show. Financial statements lag; recent events may not be in them."})
    if quant <= 1 and (has("market", "outperformance") or has("news", "positive_news_tone")):
        notes.append({"type": "conflict", "detail":
                      "Financial risk signals contrast with strong price action or positive "
                      "news; the market may expect a recovery or may not be pricing the risk."})
    if quant <= 1 and market <= 1:
        notes.append({"type": "agreement", "detail":
                      "Both the financial statements and the share price show stress."})
    if quant >= 3 and market >= 3 and news >= 3:
        notes.append({"type": "agreement", "detail":
                      "No agent found medium or high negative signals."})
    return notes


def template_summary(state, agents, level):
    name = state.get("name") or state.get("symbol")
    lines = [f"{name} ({state.get('symbol')}): attention level {level}."]
    for key, label in (("quantitative_risk", "Financials"), ("market", "Market"), ("news", "News")):
        a = agents.get(key, {})
        if a.get("summary"):
            lines.append(f"{label}: {a['summary']}")
    return " ".join(lines)


def llm_summary(state, agents, level, notes):
    compact = {
        name: {"status": a.get("status"), "summary": a.get("summary"),
               "signals": [{k: s[k] for k in ("severity", "direction", "title", "detail")}
                           for s in a.get("signals", [])]}
        for name, a in agents.items()
    }
    prompt = (
        f"Company: {state.get('name')} ({state.get('symbol')})\n"
        f"Attention level (rule-based): {level}\nCross-checks: {json.dumps(notes)}\n"
        f"Agent findings (JSON):\n{json.dumps(compact, indent=1, default=str)}\n\n"
        "Write a combined research summary of 150-250 words in plain English with three short "
        "paragraphs: financial position, market behaviour, and recent news, then one sentence "
        "on where the agents agree or conflict. Use only the facts above; keep numbers as "
        "given. Do not recommend buying, selling or holding, and do not predict prices."
    )
    return llm.complete(
        "You are a careful financial research assistant combining the findings of three "
        "analysis agents into a neutral summary for a research report.",
        prompt, effort="medium", max_tokens=4000)


def combine(state):
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if state.get("error"):
        return {"schema_version": SCHEMA_VERSION, "status": "error", "generated_at": generated,
                "request": {"input": state.get("input")}, "company": None, "agents": None,
                "manager": None, "analysis": None, "error": state["error"],
                "disclaimer": DISCLAIMER}
    agents = {name: state.get(name) or result(name, "error", "Agent did not run.")
              for name in AGENTS}
    level = attention_level(agents)
    notes = cross_checks(agents)
    findings = sorted(
        ({"agent": name, **s} for name, a in agents.items() for s in a.get("signals", [])),
        key=lambda s: (SEVERITY_ORDER.get(s["severity"], 9), s["agent"]))
    text = llm_summary(state, agents, level, notes) if llm.available() else None
    summary_source = "claude" if text else "template"
    profile = state.get("profile") or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "generated_at": generated,
        "request": {"input": state.get("input")},
        "company": {"symbol": state["symbol"], "name": state.get("name"),
                    "exchange": profile.get("fullExchangeName") or profile.get("exchange"),
                    "sector": profile.get("sector"), "industry": profile.get("industry"),
                    "country": profile.get("country"),
                    "reporting_currency": profile.get("financialCurrency"),
                    "resolution": state.get("resolution")},
        "agents": agents,
        "manager": {
            "attention_level": level,
            "attention_rule": ATTENTION_RULE,
            "key_findings": findings,
            "cross_checks": notes,
            "summary": text or template_summary(state, agents, level),
            "summary_source": summary_source,
            "llm_note": None if text or not llm.available() else llm.last_error(),
        },
        "analysis": state.get("analysis"),
        "error": None,
        "disclaimer": DISCLAIMER,
    }


def run_research(company, provider=None, output_root=None, use_google_news=True,
                 distress_bundle=None):
    graph = _compile(provider, output_root, use_google_news, distress_bundle)
    final = graph.invoke({"input": " ".join(str(company or "").split())})
    return final["report"]
