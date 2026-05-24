"""Token-aware rate limiter — bonus feature (+2 pts).

:class:`TokenBudget` enforces a tokens-per-minute (TPM) ceiling using a
60-second sliding window of recorded token events. Before each AI call,
the service layer reserves the estimated tokens; if the call would
push usage past the configured limit, the budget sleeps until enough
window has elapsed for the reservation to fit.

A token budget is more accurate than a simple request-count semaphore
because providers bill by token, not by request — a single long prompt
can exhaust the ceiling faster than ten short ones.

Approximate provider limits as of May 2026:

* Anthropic Claude Sonnet: 200 000 TPM (free tier 40 000)
* OpenAI gpt-4o-mini: 200 000 TPM
* Gemini 2.0 Flash: 1 000 000 TPM (free tier 15 000)

Tune ``AI_CALL_TPM_LIMIT`` in ``.env`` to match the account tier in use.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

log = logging.getLogger(__name__)


class TokenBudget:
    """Async token-bucket rate limiter with a 60-second sliding window.

    Recorded events are dropped from the window the moment they age past
    sixty seconds, so the rate cap is genuinely a rolling minute rather
    than a fixed-interval reset.
    """

    def __init__(self, tokens_per_minute: int) -> None:
        """Initialise the budget.

        Args:
            tokens_per_minute: Hard ceiling on tokens consumed per
                60-second window. Must be positive.

        Raises:
            ValueError: ``tokens_per_minute`` is not positive.
        """
        if tokens_per_minute <= 0:
            raise ValueError("tokens_per_minute must be positive")
        self.tpm = tokens_per_minute
        self._events: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int) -> None:
        """Block until the budget has room for ``estimated_tokens``.

        Args:
            estimated_tokens: Tokens the upcoming call is expected to
                consume. The caller owns the estimate (typically
                ``len(text.split()) * 1.3`` for English).

        Behaviour:
            * If the reservation fits in the current window, it is
              recorded immediately and the call returns.
            * Otherwise, the coroutine sleeps until the oldest recorded
              event has aged out of the window, then retries from scratch.
        """
        while True:
            wait_seconds = 0.0
            async with self._lock:
                now = time.monotonic()
                while self._events and self._events[0][0] < now - 60.0:
                    self._events.popleft()

                used = sum(t for _, t in self._events)
                if used + estimated_tokens <= self.tpm:
                    self._events.append((now, estimated_tokens))
                    log.debug(
                        "rate_limiter.acquired",
                        extra={
                            "tokens": estimated_tokens,
                            "used": used,
                            "limit": self.tpm,
                        },
                    )
                    return

                oldest_ts, _ = self._events[0]
                wait_seconds = max(60.0 - (now - oldest_ts), 0.1)
                log.warning(
                    "rate_limiter.budget_exhausted",
                    extra={
                        "used": used,
                        "limit": self.tpm,
                        "requested": estimated_tokens,
                        "wait_s": f"{wait_seconds:.2f}",
                    },
                )

            await asyncio.sleep(wait_seconds)

    @property
    def tokens_used_in_window(self) -> int:
        """Sum of tokens consumed within the current 60-second window."""
        now = time.monotonic()
        return sum(t for ts, t in self._events if ts >= now - 60.0)

    def reset(self) -> None:
        """Clear every recorded event. Used by tests for a clean slate."""
        self._events.clear()


_budget: TokenBudget | None = None
"""Process-wide singleton; built lazily by :func:`get_token_budget`."""


def get_token_budget() -> TokenBudget:
    """Return the shared :class:`TokenBudget`, constructing it on first call.

    The budget reads ``AI_CALL_TPM_LIMIT`` from settings the first time
    it is requested, so importing this module does not eagerly require
    a fully-configured environment.
    """
    global _budget
    if _budget is None:
        from src.config import settings

        tpm = getattr(settings, "AI_CALL_TPM_LIMIT", 40_000)
        _budget = TokenBudget(tpm)
        log.info("rate_limiter.initialized", extra={"tpm": tpm})
    return _budget
