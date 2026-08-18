#!/bin/sh
# Production ASGI startup for the Face Verification API.
#
# Worker count and trusted-proxy IPs are intentionally NOT hardcoded here:
# uvicorn reads them itself from $WEB_CONCURRENCY (defaults to 1) and
# $FORWARDED_ALLOW_IPS (defaults to 127.0.0.1) if set in the container
# environment -- see docs/deployment.md "Production ASGI startup
# configuration" for what to actually set them to and why.
set -e

exec uvicorn app.main:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --proxy-headers \
    --no-server-header \
    --timeout-keep-alive "${TIMEOUT_KEEP_ALIVE:-30}" \
    --timeout-graceful-shutdown "${TIMEOUT_GRACEFUL_SHUTDOWN:-30}"
