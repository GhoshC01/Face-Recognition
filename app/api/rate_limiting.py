from __future__ import annotations

import time
from collections import deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.schemas.common import ErrorResponse
from app.utils.request_context import get_request_id


class InMemoryRateLimiter(BaseHTTPMiddleware):
    """Best-effort, per-process sliding-window rate limiter, keyed by API key
    when present (so one registered caller shares one budget regardless of
    which host/NAT it calls from) or by client IP otherwise.

    This is in-memory and therefore per-process: it does NOT enforce a
    correct global limit across multiple replicas/workers, since each one
    keeps its own counters. For a horizontally-scaled deployment, prefer
    rate limiting at the API gateway/ingress (which sees all traffic) or a
    shared store (e.g. Redis). This middleware is a reasonable default for
    single-instance deployments and defense-in-depth alongside gateway-level
    limiting, not a replacement for it.
    """

    def __init__(self, app, requests_per_window: int, window_seconds: float) -> None:
        super().__init__(app)
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self._hits: dict[str, deque] = {}

    @staticmethod
    def _client_key(request: Request) -> str:
        api_key = request.headers.get("x-api-key")
        if api_key:
            return f"key:{api_key}"
        client = request.client
        return f"ip:{client.host}" if client else "ip:unknown"

    async def dispatch(self, request: Request, call_next) -> Response:
        key = self._client_key(request)
        now = time.monotonic()
        window_start = now - self.window_seconds

        hits = self._hits.setdefault(key, deque())
        while hits and hits[0] < window_start:
            hits.popleft()

        if len(hits) >= self.requests_per_window:
            body = ErrorResponse(
                error_code="rate_limit_exceeded",
                message="Too many requests; please slow down.",
                request_id=get_request_id(),
            )
            return JSONResponse(status_code=429, content=body.model_dump(mode="json"))

        hits.append(now)
        return await call_next(request)
