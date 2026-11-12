"""HTTP API for the FinTech application (FastAPI).

    python main.py api                     -> http://127.0.0.1:8000  (docs at /docs)

Endpoints
  GET /health                          service status, model and LLM availability
  GET /api/analyze?company=AAPL        company_analysis/1.0 (financials, risk flags)
  GET /api/research?company=AAPL       research_report/1.0 (all agents + manager)
  GET /api/model                       distress-model evaluation summary
  GET /                                local test page

Errors use the payload's coded `error` with HTTP 400 (empty input), 404
(unknown company / not a company / no data), 502 (data provider failure)
or 500 (internal error).
"""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from src.agents import llm
from src.agents.manager import run_research
from src.ml.distress import model as distress_model
from src.pipeline.analyze import analyze_company


STATUS_BY_ERROR = {
    "empty_input": 400, "unknown_company": 404, "no_financial_data": 404,
    "not_a_company": 404, "provider_error": 502, "internal_error": 500,
}
PAGE = Path(__file__).resolve().parents[1] / "web" / "index.html"


def _status(payload):
    if payload.get("status") == "error":
        return STATUS_BY_ERROR.get((payload.get("error") or {}).get("code"), 500)
    return 200


def create_app(output_root=None, provider_factory=None, use_google_news=True):
    app = FastAPI(title="Multi-Agent Investment Research API", version="1.1.0",
                  description="Informational research output only; not investment advice.")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"],
                       allow_headers=["*"])
    state = {"bundle": None, "loaded": False}
    lock = threading.Lock()

    def bundle():
        with lock:
            if not state["loaded"]:
                state["bundle"] = distress_model.load_model()
                state["loaded"] = True
        return state["bundle"]

    def provider():
        return provider_factory() if provider_factory else None

    @app.get("/health")
    def health():
        b = bundle()
        return {"status": "ok",
                "distress_model": None if b is None else {
                    "model": b["meta"]["model_name"],
                    "trained_at": b["meta"]["trained_at"],
                    "training_rows": b["meta"]["training_rows"]},
                "llm_available": llm.available()}

    @app.get("/api/analyze")
    def analyze(company: str = Query("", description="Ticker or company name")):
        payload = analyze_company(company, output_root=output_root, provider=provider(),
                                  echo=lambda *a: None)
        return JSONResponse(payload, status_code=_status(payload))

    @app.get("/api/research")
    def research(company: str = Query("", description="Ticker or company name")):
        report = run_research(company, provider=provider(), output_root=output_root,
                              use_google_news=use_google_news, distress_bundle=bundle())
        return JSONResponse(report, status_code=_status(report))

    @app.get("/api/model")
    def model_info():
        b = bundle()
        if b is None:
            return JSONResponse({"status": "unavailable",
                                 "message": "Run `python main.py distress` first."}, 404)
        meta = b["meta"]
        return {"status": "ok", "model": meta["model_name"], "trained_at": meta["trained_at"],
                "training_rows": meta["training_rows"],
                "training_companies": meta["training_companies"],
                "evaluation": meta["evaluation"], "conclusions": meta.get("conclusions"),
                "global_importance": meta.get("global_importance")}

    @app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def page():
        return PAGE.read_text(encoding="utf-8")

    return app
