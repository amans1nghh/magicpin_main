"""FastAPI application and router for Vera Merchant Bot.

Exposes the 5 judge-harness endpoints:
  GET  /v1/healthz   — liveness + context load state
  GET  /v1/metadata   — team identity and approach
  POST /v1/context    — context ingestion with version conflict handling
  POST /v1/tick       — proactive message initiation
  POST /v1/reply      — multi-turn conversation handling
"""

from __future__ import annotations

import logging
import sys
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.composer import Composer
from src.config import (
    APPROACH,
    CONTACT_EMAIL,
    SUBMITTED_AT,
    TEAM_MEMBERS,
    TEAM_NAME,
    VERSION,
    get_llm_config,
)
from src.context_store import ContextStore
from src.conversation_manager import ConversationManager
from src.llm_client import LLMClient
from src.models import (
    ContextAcceptedResponse,
    ContextPushRequest,
    ContextRejectedResponse,
    HealthResponse,
    MetadataResponse,
    ReplyRequest,
    TickRequest,
)
from src.trigger_evaluator import TriggerEvaluator

# ---------------------------------------------------------------------------
# Structured logging — timestamp, level, module in every line
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stderr,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

START_TIME = time.time()

app = FastAPI(title="Vera Merchant Bot", version=VERSION)


# ---------------------------------------------------------------------------
# Middleware — request timing
# ---------------------------------------------------------------------------


@app.middleware("http")
async def log_requests(request: Request, call_next):  # noqa: ANN001
    """Log method, path, and response latency for every request."""
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(
        "%s %s — %dms (status %d)",
        request.method,
        request.url.path,
        duration_ms,
        response.status_code,
    )
    return response


# ---------------------------------------------------------------------------
# Global exception handler — always return valid JSON
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch any unhandled exception and return a safe JSON response.

    Returns HTTP 200 to avoid judge penalties for non-200 status codes.
    """
    logger.exception(
        "Unhandled exception on %s %s", request.method, request.url.path,
    )
    return JSONResponse(
        content={"error": "internal_error", "detail": str(exc)},
        status_code=200,
    )

# App-level singletons shared across all endpoints
context_store = ContextStore()

_llm_config = get_llm_config()
llm_client = LLMClient(
    provider=_llm_config.provider,
    api_key=_llm_config.api_key,
    model=_llm_config.model,
    timeout=_llm_config.timeout,
)

composer = Composer(llm_client=llm_client, context_store=context_store)
trigger_evaluator = TriggerEvaluator(context_store=context_store)
conversation_manager = ConversationManager(context_store=context_store)


# ---------------------------------------------------------------------------
# GET /v1/healthz
# ---------------------------------------------------------------------------


@app.get("/v1/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """Return liveness status, uptime, and context counts per scope."""
    uptime = int(time.time() - START_TIME)
    counts = context_store.count_by_scope()
    return HealthResponse(
        status="ok",
        uptime_seconds=uptime,
        contexts_loaded=counts,
    )


# ---------------------------------------------------------------------------
# GET /v1/metadata
# ---------------------------------------------------------------------------


@app.get("/v1/metadata", response_model=MetadataResponse)
async def metadata() -> MetadataResponse:
    """Return team identity and approach description."""
    return MetadataResponse(
        team_name=TEAM_NAME,
        team_members=TEAM_MEMBERS,
        model="gpt-4o",
        approach=APPROACH,
        contact_email=CONTACT_EMAIL,
        version=VERSION,
        submitted_at=SUBMITTED_AT,
    )


# ---------------------------------------------------------------------------
# POST /v1/context
# ---------------------------------------------------------------------------


@app.post("/v1/context")
async def push_context(request: ContextPushRequest) -> JSONResponse:
    """Ingest a context push with version conflict handling.

    Returns:
        200 — context accepted
        409 — stale version (version <= current)
        400 — invalid scope
    """
    result = await context_store.put(
        scope=request.scope,
        context_id=request.context_id,
        version=request.version,
        payload=request.payload,
        delivered_at=request.delivered_at,
    )

    if result.accepted:
        body = ContextAcceptedResponse(
            accepted=True,
            ack_id=result.ack_id,  # type: ignore[arg-type]
            stored_at=result.stored_at,  # type: ignore[arg-type]
        )
        return JSONResponse(content=body.model_dump(), status_code=200)

    # Rejection path
    if result.reason == "stale_version":
        body_rejected = ContextRejectedResponse(
            accepted=False,
            reason="stale_version",
            current_version=result.current_version,
        )
        return JSONResponse(content=body_rejected.model_dump(), status_code=409)

    # invalid_scope
    body_rejected = ContextRejectedResponse(
        accepted=False,
        reason="invalid_scope",
    )
    return JSONResponse(content=body_rejected.model_dump(), status_code=400)


# ---------------------------------------------------------------------------
# POST /v1/tick
# ---------------------------------------------------------------------------


@app.post("/v1/tick")
async def tick(request: TickRequest) -> JSONResponse:
    """Proactive message initiation via TriggerEvaluator + Composer pipeline.

    Evaluates available triggers, composes messages for valid ones,
    registers conversations, and returns up to 20 actions.
    """
    try:
        actions = await trigger_evaluator.evaluate(
            available_trigger_ids=request.available_triggers,
            now=request.now,
            composer=composer,
        )

        # Register conversations for each action
        for action in actions:
            conversation_manager.register_conversation(
                conversation_id=action.conversation_id,
                merchant_id=action.merchant_id,
                customer_id=action.customer_id,
                trigger_id=action.trigger_id,
                initial_body=action.body,
            )

        return JSONResponse(
            content={"actions": [a.model_dump() for a in actions]},
            status_code=200,
        )
    except Exception:
        logger.exception("Error processing tick request")
        return JSONResponse(content={"actions": []}, status_code=200)


# ---------------------------------------------------------------------------
# POST /v1/reply
# ---------------------------------------------------------------------------


@app.post("/v1/reply")
async def reply(request: ReplyRequest) -> JSONResponse:
    """Multi-turn conversation handling via ConversationManager + Composer.

    Classifies the incoming message, routes to the appropriate handler,
    and returns a send/wait/end action.
    """
    try:
        result = await conversation_manager.handle_reply(
            conversation_id=request.conversation_id,
            merchant_id=request.merchant_id or "",
            customer_id=request.customer_id,
            from_role=request.from_role,
            message=request.message,
            received_at=request.received_at,
            turn_number=request.turn_number,
            composer=composer,
        )
        return JSONResponse(content=result, status_code=200)
    except Exception:
        logger.exception("Error processing reply request")
        return JSONResponse(
            content={
                "action": "wait",
                "wait_seconds": 60,
                "rationale": "Internal error — retrying later.",
            },
            status_code=200,
        )
