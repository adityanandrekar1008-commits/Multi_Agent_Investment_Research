"""Local test page for the single-company analysis (standard library only).

    python main.py serve            -> http://127.0.0.1:8765

GET /                              test page
GET /api/analyze?company=<text>    runs the same analyze_company() as
                                   `python main.py analyze` and returns its
                                   summary JSON

HTTP status: 200 ok, 400 empty input, 404 unknown company / no data /
not a company, 502 provider failure.

This is a developer test page, not the product front end (the FinTech side
owns that). It listens on 127.0.0.1 only.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.pipeline.analyze import analyze_company


PAGE = Path(__file__).with_name("index.html")
STATUS_BY_ERROR = {
    "empty_input": 400,
    "unknown_company": 404,
    "no_financial_data": 404,
    "not_a_company": 404,
    "provider_error": 502,
}

# One analysis at a time: each one makes several provider requests.
_analysis_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = "InvestmentResearchTest/1.0"
    output_root = None

    def _send(self, status, body, content_type, include_body=True):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if include_body:
            self.wfile.write(data)

    def do_HEAD(self):
        # Health checks; never runs an analysis.
        if urlparse(self.path).path in ("/", "/index.html"):
            self._send(200, PAGE.read_text(encoding="utf-8"),
                       "text/html; charset=utf-8", include_body=False)
        else:
            self._send(404, "", "text/plain", include_body=False)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._send(200, PAGE.read_text(encoding="utf-8"),
                       "text/html; charset=utf-8")
        elif url.path == "/api/analyze":
            company = parse_qs(url.query).get("company", [""])[0]
            with _analysis_lock:
                summary = analyze_company(
                    company, output_root=self.output_root, echo=lambda *_: None
                )
            status = 200
            if summary["status"] == "error":
                status = STATUS_BY_ERROR.get(summary["error"]["code"], 500)
            self._send(status, json.dumps(summary, default=str),
                       "application/json; charset=utf-8")
        elif url.path == "/api/research":
            from src.agents.manager import run_research

            company = parse_qs(url.query).get("company", [""])[0]
            with _analysis_lock:
                report = run_research(company, output_root=self.output_root)
            status = 200
            if report["status"] == "error":
                status = STATUS_BY_ERROR.get(report["error"]["code"], 500)
            self._send(status, json.dumps(report, default=str),
                       "application/json; charset=utf-8")
        elif url.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self._send(404, json.dumps({"error": "not found"}),
                       "application/json; charset=utf-8")

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}", flush=True)


def serve(host="127.0.0.1", port=8765, output_root=None):
    Handler.output_root = output_root
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Local test page: http://{host}:{port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
