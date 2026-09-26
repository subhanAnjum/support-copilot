"""HTTP API and web UI."""
import os
import time
import logging
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.graph import run_ticket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("support-copilot")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # seed the vector store and load the classifier before serving
    from app.rag.ingest import seed_if_empty
    from app.classifier.infer import classify_ticket

    seed_if_empty()
    classify_ticket("warm-up")
    yield


app = FastAPI(title="Support Copilot", version="0.1.0", lifespan=lifespan)

_recent_requests: dict[str, deque] = defaultdict(deque)


def _check_rate_limit(request: Request):
    # simple in-memory per-IP limit (per process, resets on restart)
    forwarded = request.headers.get("x-forwarded-for")
    ip = forwarded.split(",")[0].strip() if forwarded else request.client.host
    now = time.monotonic()
    window = _recent_requests[ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(429, "Too many requests, please try again in a minute.")
    window.append(now)


class TicketRequest(BaseModel):
    text: str = Field(min_length=3, max_length=2000)


class TicketResponse(BaseModel):
    response: str
    category: str
    classifier_confidence: float | None
    classifier_method: str
    sources_cited: list[str]
    escalated: bool
    latency_ms: int


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ticket", response_model=TicketResponse)
def handle_ticket(req: TicketRequest, request: Request):
    _check_rate_limit(request)
    start = time.perf_counter()
    result = run_ticket(req.text)
    latency_ms = int((time.perf_counter() - start) * 1000)

    logger.info(
        "ticket processed category=%s confidence=%s latency_ms=%d",
        result.get("category"), result.get("classifier_confidence"), latency_ms,
    )

    return TicketResponse(
        response=result["response"],
        category=result["category"],
        classifier_confidence=result.get("classifier_confidence"),
        classifier_method=result.get("classifier_method", "unknown"),
        sources_cited=result.get("sources_cited", []),
        escalated=result.get("escalated", False),
        latency_ms=latency_ms,
    )
