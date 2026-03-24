"""Rate limiting middleware using Redis."""

import time

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.core.config import get_settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket rate limiting using Redis or in-memory fallback."""

    def __init__(self, app, redis_url: str | None = None):
        super().__init__(app)
        self.redis_url = redis_url
        self._redis = None
        self._local_store: dict[str, list[float]] = {}
        self.settings = get_settings()

    async def _get_redis(self):
        """Lazy-initialize Redis connection."""
        if self._redis is None and self.redis_url:
            try:
                import redis.asyncio as aioredis

                self._redis = aioredis.from_url(self.redis_url, decode_responses=True)
                await self._redis.ping()
            except Exception:
                self._redis = None
        return self._redis

    def _get_client_key(self, request: Request) -> str:
        """Extract rate limit key from request."""
        # Use API key if present, otherwise use IP
        api_key = request.headers.get("X-API-Key", "")
        if api_key:
            return f"rate_limit:apikey:{api_key[:16]}"

        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        return f"rate_limit:ip:{client_ip}"

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip rate limiting for health checks and docs
        if request.url.path in ("/api/v1/health", "/docs", "/openapi.json", "/health"):
            return await call_next(request)

        client_key = self._get_client_key(request)
        max_requests = self.settings.rate_limit_requests
        window = self.settings.rate_limit_period

        redis = await self._get_redis()

        if redis:
            allowed, remaining, reset_at = await self._check_redis(
                redis, client_key, max_requests, window
            )
        else:
            allowed, remaining, reset_at = self._check_local(
                client_key, max_requests, window
            )

        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(reset_at)),
                    "Retry-After": str(int(reset_at - time.time())),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(reset_at))
        return response

    async def _check_redis(
        self, redis, key: str, max_requests: int, window: int
    ) -> tuple[bool, int, float]:
        """Check rate limit using Redis sliding window."""
        now = time.time()
        window_start = now - window

        pipe = redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, window)
        results = await pipe.execute()

        request_count = results[2]
        remaining = max(0, max_requests - request_count)
        reset_at = now + window

        return request_count <= max_requests, remaining, reset_at

    def _check_local(
        self, key: str, max_requests: int, window: int
    ) -> tuple[bool, int, float]:
        """Fallback in-memory rate limiting."""
        now = time.time()
        window_start = now - window

        if key not in self._local_store:
            self._local_store[key] = []

        # Remove expired entries
        self._local_store[key] = [
            t for t in self._local_store[key] if t > window_start
        ]

        request_count = len(self._local_store[key])

        if request_count >= max_requests:
            reset_at = self._local_store[key][0] + window
            return False, 0, reset_at

        self._local_store[key].append(now)
        remaining = max_requests - request_count - 1
        reset_at = now + window

        return True, remaining, reset_at
