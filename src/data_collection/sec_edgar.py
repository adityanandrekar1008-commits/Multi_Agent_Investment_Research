"""SEC EDGAR client: ticker -> CIK, and XBRL company facts.

US-listed filers only (RELIANCE.NS and other non-US listings are not on
EDGAR). The SEC's fair-access policy requires a User-Agent that names the
requester and a contact email, and at most 10 requests per second. Set
SEC_EDGAR_USER_AGENT in the environment or in the git-ignored .env file:

    SEC_EDGAR_USER_AGENT=Project Name contact@example.com

Downloads are cached under data/raw/sec_edgar/cache/ (git-ignored).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from src.data_collection.provider import ProviderError
from src.settings import get_setting


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "sec_edgar" / "cache"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
MIN_SECONDS_BETWEEN_REQUESTS = 0.2   # 5 requests/second, half the SEC limit


class EdgarConfigError(RuntimeError):
    code = "configuration_error"


class EdgarClient:
    name = "SEC EDGAR (XBRL company facts)"

    def __init__(self, user_agent=None, cache_dir=CACHE_DIR, max_cache_age_days=7):
        self.user_agent = user_agent or get_setting("SEC_EDGAR_USER_AGENT")
        if not self.user_agent or "@" not in self.user_agent:
            raise EdgarConfigError(
                "Set SEC_EDGAR_USER_AGENT (a name and contact email) in the "
                "environment or in .env before downloading from SEC EDGAR."
            )
        self.cache_dir = Path(cache_dir)
        self.max_cache_age = max_cache_age_days * 86400
        self._last_request = 0.0
        self._tickers = None

    def _get_json(self, url, cache_name):
        cache = self.cache_dir / cache_name
        if cache.exists() and time.time() - cache.stat().st_mtime < self.max_cache_age:
            return json.loads(cache.read_text(encoding="utf-8"))

        wait = MIN_SECONDS_BETWEEN_REQUESTS - (time.time() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            response = requests.get(
                url, timeout=30,
                headers={"User-Agent": self.user_agent,
                         "Accept-Encoding": "gzip, deflate"},
            )
        except requests.RequestException as exc:
            raise ProviderError(f"SEC EDGAR request failed: {exc}") from exc
        finally:
            self._last_request = time.time()
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise ProviderError(
                f"SEC EDGAR returned HTTP {response.status_code} for {url}"
            )
        data = response.json()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data), encoding="utf-8")
        return data

    def cik(self, ticker):
        """CIK for a US ticker, or None if the SEC does not list it."""
        if self._tickers is None:
            data = self._get_json(TICKERS_URL, "company_tickers.json") or {}
            self._tickers = {
                row["ticker"].upper(): int(row["cik_str"]) for row in data.values()
            }
        return self._tickers.get(ticker.upper().replace(".", "-"))

    def company_facts(self, ticker):
        """(cik, company facts JSON) or (None, None) for non-SEC tickers."""
        cik = self.cik(ticker)
        if cik is None:
            return None, None
        return cik, self._get_json(FACTS_URL.format(cik=cik), f"CIK{cik:010d}.json")
