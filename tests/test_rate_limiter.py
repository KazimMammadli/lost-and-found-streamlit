"""Tests for the token-aware rate limiter (src/concurrency/rate_limiter.py).

Uses a frozen-time approach by patching ``time.monotonic`` to simulate
token budget exhaustion and replenishment without sleeping in CI.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from src.concurrency.rate_limiter import TokenBudget


def test_token_budget_rejects_zero_tpm():
    with pytest.raises(ValueError, match="positive"):
        TokenBudget(0)


def test_token_budget_rejects_negative_tpm():
    with pytest.raises(ValueError, match="positive"):
        TokenBudget(-100)


def test_token_budget_created_with_valid_tpm():
    budget = TokenBudget(10_000)
    assert budget.tpm == 10_000


@pytest.mark.asyncio
async def test_acquire_within_limit():
    budget = TokenBudget(1000)
    await budget.acquire(500)
    assert budget.tokens_used_in_window == 500


@pytest.mark.asyncio
async def test_acquire_exact_limit():
    budget = TokenBudget(1000)
    await budget.acquire(1000)
    assert budget.tokens_used_in_window == 1000


@pytest.mark.asyncio
async def test_acquire_accumulates_within_window():
    budget = TokenBudget(1000)
    await budget.acquire(300)
    await budget.acquire(400)
    await budget.acquire(300)
    assert budget.tokens_used_in_window == 1000


@pytest.mark.asyncio
async def test_tokens_expire_after_window():
    """Tokens from >60s ago should not count toward the current window."""
    budget = TokenBudget(100)

    with patch("src.concurrency.rate_limiter.time.monotonic", return_value=0.0):
        budget._events.append((0.0, 80))

    with patch("src.concurrency.rate_limiter.time.monotonic", return_value=90.0):
        used = budget.tokens_used_in_window
        assert used == 0


@pytest.mark.asyncio
async def test_acquire_after_window_expiry():
    """After old tokens expire, new tokens fit under the limit."""
    budget = TokenBudget(100)
    budget._events.append((0.0, 90))

    with patch("src.concurrency.rate_limiter.time.monotonic", return_value=70.0):
        await budget.acquire(90)
        assert budget.tokens_used_in_window <= 90


@pytest.mark.asyncio
async def test_reset_clears_events():
    budget = TokenBudget(500)
    await budget.acquire(200)
    assert budget.tokens_used_in_window > 0

    budget.reset()
    assert budget.tokens_used_in_window == 0


@pytest.mark.asyncio
async def test_tokens_used_reflects_recent_acquires():
    budget = TokenBudget(10_000)
    await budget.acquire(100)
    await budget.acquire(200)
    assert budget.tokens_used_in_window == 300


def test_tokens_used_is_zero_before_any_acquire():
    budget = TokenBudget(1000)
    assert budget.tokens_used_in_window == 0


@pytest.mark.asyncio
async def test_concurrent_acquires_respect_limit():
    """Multiple concurrent callers should collectively not exceed the limit."""
    budget = TokenBudget(1000)
    results = await asyncio.gather(
        budget.acquire(200),
        budget.acquire(200),
        budget.acquire(200),
    )
    assert all(r is None for r in results)
    assert budget.tokens_used_in_window <= 1000


def test_get_token_budget_returns_budget_instance():
    """get_token_budget() returns a TokenBudget without raising."""
    import src.concurrency.rate_limiter as rl_module

    rl_module._budget = None
    from src.concurrency.rate_limiter import get_token_budget
    budget = get_token_budget()
    assert isinstance(budget, TokenBudget)
    assert budget.tpm > 0


def test_get_token_budget_is_singleton():
    import src.concurrency.rate_limiter as rl_module
    rl_module._budget = None
    from src.concurrency.rate_limiter import get_token_budget
    b1 = get_token_budget()
    b2 = get_token_budget()
    assert b1 is b2
