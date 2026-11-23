"""Polite SEC EDGAR HTTP client.

* User-Agent comes from the ``SEC_USER_AGENT`` env var (or an explicit argument); construction fails
  with a clear error if it is missing (unless an httpx client/transport is injected, e.g. in tests).
* Token-bucket rate limit, hard-capped at 8 requests/second (SEC allows 10).
* Exponential backoff (honouring ``Retry-After``) on 429 / 5xx / timeouts / transport errors.
* 403 cooldown: SEC blocks an IP for ~10 minutes when it decides traffic is too heavy. If an earlier request
  in this session succeeded (so the User-Agent is accepted), a 403 waits ``block_cooldown`` seconds and
  retries, at most ``max_block_cooldowns`` times per client. A 403 on the very first request (bad or
  missing User-Agent) still fails immediately.
* Immutable on-disk cache of filing documents: ``<cache_dir>/<accession>.xml`` (+ ``.meta.json``).
  Filings never change, so a filing is fetched once and then re-parsed offline.

Nothing here runs at import time: no env reads, no network.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

MAX_RATE_PER_SEC = 8.0
BLOCK_COOLDOWN_S = 600.0  # SEC: "traffic exceeding the limit ... will be limited for 10 minutes"
# Phrases from SEC's 403 page, used only to make the error message say which kind of block it is.
_BLOCK_REASONS = (
    ("Request Rate Threshold Exceeded", "rate threshold exceeded"),
    ("Undeclared Automated Tool", "User-Agent not accepted as a declared automated tool"),
)
DEFAULT_CACHE_DIR = Path("data/raw/edgar")
USER_AGENT_ENV = "SEC_USER_AGENT"
_ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class EdgarError(Exception):
    """Base class for EDGAR client errors."""


class EdgarConfigError(EdgarError):
    """Client misconfigured (e.g. SEC_USER_AGENT not set)."""


class EdgarHTTPError(EdgarError):
    def __init__(self, message: str, status: int | None = None, url: str | None = None):
        super().__init__(message)
        self.status = status
        self.url = url


class EdgarNotFound(EdgarHTTPError):
    """404: e.g. no daily index for a weekend/holiday, or an unknown filing folder."""


class EdgarBlockedError(EdgarHTTPError):
    """403: SEC is refusing us (missing/invalid User-Agent or temporary IP block). Never retried."""


class TokenBucket:
    """Thread-safe token bucket. ``acquire()`` blocks (via the injected sleep) until a token is available."""

    def __init__(
        self,
        rate_per_sec: float,
        capacity: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be > 0")
        self.rate = float(rate_per_sec)
        self.capacity = float(capacity)
        self._clock = clock
        self._sleep = sleep
        self._tokens = self.capacity
        self._last = clock()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Reserve one token; returns the seconds slept."""
        with self._lock:
            now = self._clock()
            self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
            self._last = now
            self._tokens -= 1.0
            wait = -self._tokens / self.rate if self._tokens < 0 else 0.0
        if wait > 0:
            self._sleep(wait)
        return wait


def _check_accession(accession: str) -> str:
    if not _ACCESSION_RE.match(accession or ""):
        raise ValueError(f"bad accession number: {accession!r} (expected 0001234567-26-000123)")
    return accession


class EdgarClient:
    def __init__(
        self,
        user_agent: str | None = None,
        *,
        cache_dir: str | os.PathLike | None = DEFAULT_CACHE_DIR,
        rate_per_sec: float = MAX_RATE_PER_SEC,
        max_retries: int = 5,
        backoff_base: float = 0.5,
        backoff_max: float = 30.0,
        block_cooldown: float = BLOCK_COOLDOWN_S,
        max_block_cooldowns: int = 2,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if rate_per_sec > MAX_RATE_PER_SEC:
            raise ValueError(f"rate_per_sec must be <= {MAX_RATE_PER_SEC} (SEC limit is 10 req/s)")
        ua = user_agent if user_agent is not None else os.environ.get(USER_AGENT_ENV)
        if not ua and client is None and transport is None:
            raise EdgarConfigError(
                f"{USER_AGENT_ENV} is not set. SEC requires a descriptive User-Agent with contact "
                f"info, e.g. export {USER_AGENT_ENV}='your-app-name you@example.com'"
            )
        self.user_agent = ua
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.block_cooldown = block_cooldown
        self.max_block_cooldowns = max_block_cooldowns
        self.block_cooldowns_used = 0
        self._had_success = False  # a 2xx proves the User-Agent is accepted; only then is a 403 worth waiting out
        self.timeout = timeout
        self._sleep = sleep
        self._bucket = TokenBucket(rate_per_sec, 1.0, clock, sleep)
        self._owns_client = client is None
        self._http = client or httpx.Client(transport=transport, follow_redirects=True, timeout=timeout)
        self.request_count = 0  # HTTP requests actually sent (including retries)

    # --- lifecycle ----------------------------------------------------------------------
    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- HTTP ---------------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        h = {"Accept-Encoding": "gzip, deflate"}
        if self.user_agent:
            h["User-Agent"] = self.user_agent
        return h

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), self.backoff_max)
            except ValueError:
                pass
        return min(self.backoff_base * (2**attempt), self.backoff_max)

    def get(self, url: str) -> httpx.Response:
        """Rate-limited GET with retries and 403 cooldowns. Returns a 2xx response or raises an EdgarError."""
        while True:
            try:
                resp = self._get_with_retries(url)
            except EdgarBlockedError as exc:
                if not self._had_success or self.block_cooldowns_used >= self.max_block_cooldowns:
                    raise
                self.block_cooldowns_used += 1
                logger.warning(
                    "SEC returned 403 (%s); cooling down %.0fs before retrying (cooldown %d/%d)",
                    exc,
                    self.block_cooldown,
                    self.block_cooldowns_used,
                    self.max_block_cooldowns,
                )
                self._sleep(self.block_cooldown)
                continue
            self._had_success = True
            return resp

    def _get_with_retries(self, url: str) -> httpx.Response:
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._bucket.acquire()
            self.request_count += 1
            retry_after = None
            try:
                resp = self._http.get(url, headers=self._headers(), timeout=self.timeout)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = EdgarHTTPError(f"{type(exc).__name__} fetching {url}", None, url)
                last.__cause__ = exc
            else:
                code = resp.status_code
                if code < 300:
                    return resp
                if code == 403:
                    body = resp.text or ""
                    reason = next((r for phrase, r in _BLOCK_REASONS if phrase in body), None)
                    hint = reason or f"check {USER_AGENT_ENV} / rate"
                    raise EdgarBlockedError(f"403 from {url}: SEC blocked the request ({hint})", code, url)
                if code == 404:
                    raise EdgarNotFound(f"404 not found: {url}", code, url)
                if code not in RETRY_STATUSES:
                    raise EdgarHTTPError(f"HTTP {code} for {url}", code, url)
                last = EdgarHTTPError(f"HTTP {code} for {url}", code, url)
                retry_after = resp.headers.get("Retry-After")
            if attempt < self.max_retries:
                self._sleep(self._backoff(attempt, retry_after))
        assert last is not None
        raise last

    def get_text(self, url: str) -> str:
        return self.get(url).text

    def get_json(self, url: str):
        return self.get(url).json()

    # --- immutable cache ----------------------------------------------------------------
    def cache_path(self, accession: str, suffix: str = ".xml") -> Path:
        if self.cache_dir is None:
            raise EdgarConfigError("cache_dir is disabled")
        return self.cache_dir / f"{_check_accession(accession)}{suffix}"

    def read_cached(self, accession: str, suffix: str = ".xml") -> bytes | None:
        if self.cache_dir is None:
            return None
        p = self.cache_path(accession, suffix)
        return p.read_bytes() if p.exists() else None

    def write_cached(self, accession: str, data: bytes, suffix: str = ".xml") -> Path:
        """Write-once: an existing cache entry is never overwritten."""
        p = self.cache_path(accession, suffix)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            return p
        tmp = p.with_name(f".{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_bytes(data)
        if p.exists():
            tmp.unlink()
        else:
            os.replace(tmp, p)  # atomic on POSIX
        return p

    def fetch_document_cached(self, accession: str, url: str, suffix: str = ".xml") -> bytes:
        """Return the cached document for `accession`, fetching `url` only on a cache miss."""
        hit = self.read_cached(accession, suffix)
        if hit is not None:
            return hit
        data = self.get(url).content
        if self.cache_dir is not None:
            self.write_cached(accession, data, suffix)
        return data
