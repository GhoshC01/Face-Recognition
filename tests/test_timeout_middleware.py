from __future__ import annotations

import asyncio

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.api.timeout_middleware import RequestTimeoutMiddleware


def _make_app(timeout_seconds: float, route_delay_seconds: float) -> Starlette:
    async def slow_endpoint(request):
        await asyncio.sleep(route_delay_seconds)
        return JSONResponse({"status": "ok"})

    app = Starlette(routes=[Route("/slow", slow_endpoint)])
    app.add_middleware(RequestTimeoutMiddleware, timeout_seconds=timeout_seconds)
    return app


def test_request_within_timeout_succeeds():
    app = _make_app(timeout_seconds=1.0, route_delay_seconds=0.01)
    with TestClient(app) as client:
        response = client.get("/slow")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_request_exceeding_timeout_returns_504():
    app = _make_app(timeout_seconds=0.05, route_delay_seconds=1.0)
    with TestClient(app) as client:
        response = client.get("/slow")

    assert response.status_code == 504
    assert response.json()["error_code"] == "request_timeout"
