"""News Agent: recent headlines, sentiment and notable events.

Sources (free): Yahoo Finance news for the symbol and Google News RSS for
the company name. Headlines from the last NEWS_DAYS days are de-duplicated.

Classification:
  - with Claude available: one call classifies every headline (sentiment
    -1..1, event type, relevance) and writes a short summary;
  - otherwise a finance keyword lexicon tags events and sentiment.
Only headlines and short summaries are used; full articles are not fetched.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import requests

from src.agents import llm
from src.agents.common import result, signal


NEWS_DAYS = 45
MAX_ITEMS = 25
GOOGLE_NEWS = "https://news.google.com/rss/search?q={q}&hl=en-{gl}&gl={gl}&ceid={gl}:en"

EVENTS = {
    # event: (severity if negative, regex)
    "bankruptcy_or_default": ("high", r"bankrupt|insolven|chapter 11|default(s|ed)? on|debt restructur|going concern"),
    "fraud_or_accounting": ("high", r"fraud|accounting (irregular|scandal)|restat|misstat|embezzl"),
    "legal_or_regulatory": ("medium", r"lawsuit|sued|sues|probe|investigat|regulator|fine[sd]?\b|penalt|antitrust|sec charges|ban(s|ned)?\b|recall"),
    "downgrade": ("medium", r"downgrad|cut(s)? (to|rating)|price target cut|lowers? (price )?target"),
    "guidance_cut_or_miss": ("medium", r"miss(es|ed)? (estimates|expectations)|cuts? (guidance|outlook|forecast)|profit warning|weak (guidance|outlook)"),
    "layoffs": ("medium", r"layoff|lay off|job cuts|cuts? \d[\d,]* jobs|workforce reduction"),
    "management_change": ("low", r"ceo (resign|steps down|exit|quit)|cfo (resign|steps down)|resigns"),
    "dilution_or_financing": ("low", r"share sale|stock offering|dilut|raises? \$?\d.*(debt|notes)|convertible"),
    "upgrade": (None, r"upgrad|raises? (price )?target|price target (hike|raised)"),
    "beat_or_raise": (None, r"beat(s)? (estimates|expectations)|record (revenue|profit)|raises? (guidance|outlook|forecast)|strong (results|quarter)"),
    "deal": (None, r"acquir|acquisition|merger|partnership|stake in|contract win"),
}
POSITIVE = r"surge|soar|jump|rall(y|ies)|gain|record|beat|upgrad|strong|growth|profit rises|bullish|outperform|all-time high|new high|hits? high"
CORPORATE_SUFFIX = r"[,.]?\s+(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|llc|holdings?|group|sa|ag|nv)\.?$"


def search_name(name):
    """'Reliance Industries Limited' -> 'Reliance Industries'."""
    clean = (name or "").strip()
    for _ in range(2):
        clean = re.sub(CORPORATE_SUFFIX, "", clean, flags=re.IGNORECASE).strip()
    return clean
NEGATIVE = r"plunge|slump|tumble|fall|drop|slide|crash|loss|weak|miss|downgrad|cut|warn|lawsuit|probe|bearish|layoff|decline|concern|\d+-(week|month|year) low|hits? (a )?(new |\d+-\w+ )?low"


def _parse_yahoo(items):
    out = []
    for item in items:
        content = item.get("content", item)
        title = content.get("title")
        if not title:
            continue
        when = content.get("pubDate") or content.get("displayTime")
        if when:
            try:
                when = datetime.fromisoformat(str(when).replace("Z", "+00:00"))
            except ValueError:
                when = None
        elif item.get("providerPublishTime"):
            when = datetime.fromtimestamp(item["providerPublishTime"], tz=timezone.utc)
        provider = content.get("provider") or {}
        url = (content.get("canonicalUrl") or {}).get("url") or item.get("link")
        out.append({"title": title.strip(),
                    "summary": (content.get("summary") or "")[:400],
                    "published": when, "publisher": provider.get("displayName") or item.get("publisher"),
                    "url": url, "source": "Yahoo Finance"})
    return out


def _google_news(query, country="US", timeout=15):
    url = GOOGLE_NEWS.format(q=quote_plus(query), gl=country)
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except (requests.RequestException, ET.ParseError):
        return []
    out = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        try:
            when = parsedate_to_datetime(item.findtext("pubDate"))
        except (TypeError, ValueError):
            when = None
        source = item.find("source")
        out.append({"title": title, "summary": "", "published": when,
                    "publisher": source.text if source is not None else None,
                    "url": item.findtext("link"), "source": "Google News"})
    return out


def collect_headlines(symbol, name, provider, google=True):
    items = _parse_yahoo(provider.news(symbol))
    if google and name:
        country = "IN" if symbol.upper().endswith((".NS", ".BO")) else "US"
        items += _google_news(f"{search_name(name)} share", country)
    cutoff = datetime.now(timezone.utc) - timedelta(days=NEWS_DAYS)
    seen, unique = set(), []
    for it in sorted(items, key=lambda x: x["published"] or cutoff, reverse=True):
        key = re.sub(r"[^a-z0-9]", "", it["title"].lower().split(" - ")[0])[:80]
        if key in seen or (it["published"] and it["published"] < cutoff):
            continue
        seen.add(key)
        unique.append(it)
    return unique[:MAX_ITEMS]


def lexicon_classify(item):
    text = f"{item['title']} {item.get('summary', '')}".lower()
    event = next((e for e, (_, pattern) in EVENTS.items() if re.search(pattern, text)), "other")
    pos = len(re.findall(POSITIVE, text))
    neg = len(re.findall(NEGATIVE, text))
    sentiment = 0.0 if pos == neg else (pos - neg) / (pos + neg)
    if EVENTS.get(event, (None,))[0] in ("high", "medium"):
        sentiment = min(sentiment, -0.5)
    return {"sentiment": round(sentiment, 2), "event": event, "relevant": True}


LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "index": {"type": "integer"},
                "sentiment": {"type": "number"},
                "event": {"type": "string", "enum": list(EVENTS) + ["other"]},
                "relevant": {"type": "boolean"},
            },
            "required": ["index", "sentiment", "event", "relevant"],
            "additionalProperties": False,
        }},
        "summary": {"type": "string"},
    },
    "required": ["items", "summary"],
    "additionalProperties": False,
}


def llm_classify(items, name):
    lines = "\n".join(f"{i}. {it['title']}" + (f" | {it['summary'][:200]}" if it["summary"] else "")
                      for i, it in enumerate(items))
    prompt = (
        f"Company: {name}\nHeadlines (index. title | summary):\n{lines}\n\n"
        "For each headline give: sentiment for the company's financial outlook from -1 "
        "(clearly negative) to 1 (clearly positive), 0 if neutral; the event type; and "
        "whether it is really about this company. Then write a 2-3 sentence factual summary "
        "of what the news says about the company. Use only the headlines given; do not add "
        "facts, predictions or investment advice."
    )
    return llm.complete(
        "You classify financial news headlines for a research tool. Be factual and neutral.",
        prompt, schema=LLM_SCHEMA, effort="low", max_tokens=4000)


def run(symbol, name, provider, use_google=True):
    try:
        items = collect_headlines(symbol, name, provider, google=use_google)
    except Exception as exc:  # news must never break the report
        return result("news", "error", f"News could not be collected: {exc}", errors=[str(exc)])
    if not items:
        return result("news", "unavailable", f"No recent news found in the last {NEWS_DAYS} days.")

    method, llm_summary = "lexicon", None
    classified = llm_classify(items, name or symbol) if llm.available() else None
    if classified and len(classified.get("items", [])) >= len(items) // 2:
        method, llm_summary = "claude", classified.get("summary")
        by_index = {c["index"]: c for c in classified["items"]}
        for i, it in enumerate(items):
            c = by_index.get(i) or lexicon_classify(it)
            it.update({"sentiment": max(-1.0, min(1.0, float(c["sentiment"]))),
                       "event": c["event"], "relevant": c["relevant"]})
    else:
        for it in items:
            it.update(lexicon_classify(it))

    relevant = [it for it in items if it["relevant"]]
    scores = [it["sentiment"] for it in relevant]
    avg = sum(scores) / len(scores) if scores else 0.0
    counts = {}
    for it in relevant:
        counts[it["event"]] = counts.get(it["event"], 0) + 1

    signals = []
    for event, (severity, _) in EVENTS.items():
        if severity and counts.get(event):
            examples = [it["title"] for it in relevant if it["event"] == event][:2]
            signals.append(signal(f"news_{event}", severity, "negative",
                                  f"News: {event.replace('_', ' ')} ({counts[event]})",
                                  "e.g. " + " | ".join(examples)))
    if len(relevant) >= 5 and avg <= -0.3:
        signals.append(signal("negative_news_tone", "medium", "negative", "Mostly negative news",
                              f"Average sentiment {avg:+.2f} across {len(relevant)} headlines."))
    elif len(relevant) >= 5 and avg >= 0.3:
        signals.append(signal("positive_news_tone", "info", "positive", "Mostly positive news",
                              f"Average sentiment {avg:+.2f} across {len(relevant)} headlines."))
    if len(relevant) < 3:
        signals.append(signal("low_news_coverage", "info", "neutral", "Little recent news",
                              f"Only {len(relevant)} relevant headlines in {NEWS_DAYS} days."))

    summary = llm_summary or (
        f"{len(relevant)} relevant headlines in the last {NEWS_DAYS} days; average sentiment "
        f"{avg:+.2f} ({'keyword-based' if method == 'lexicon' else 'Claude'}). "
        + (("Notable events: " + ", ".join(f"{k.replace('_', ' ')} ({v})" for k, v in counts.items()
                                          if k != "other") + ".") if any(k != "other" for k in counts) else "")
    )
    data = {
        "method": method, "articles": len(items), "relevant_articles": len(relevant),
        "average_sentiment": round(avg, 3), "event_counts": counts,
        "headlines": [{"title": it["title"], "publisher": it["publisher"],
                       "published": it["published"].isoformat() if it["published"] else None,
                       "url": it["url"], "sentiment": it["sentiment"], "event": it["event"],
                       "relevant": it["relevant"]} for it in items],
    }
    sources = sorted({it["source"] for it in items})
    return result("news", "ok", summary.strip(), signals, data, sources)
