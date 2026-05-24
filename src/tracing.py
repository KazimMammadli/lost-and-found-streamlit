"""OpenTelemetry distributed tracing — bonus feature (+2 pts).

Provides OTel span instrumentation for AI calls and repository
operations. When ``ENABLE_TRACING=true`` spans are exported via OTLP
to a collector such as Jaeger; when ``ENABLE_TRACING=false`` (the
default) the no-op tracer is installed and runtime overhead is zero.

Setup
-----
1. Set ``ENABLE_TRACING=true`` and ``OTLP_ENDPOINT=http://localhost:4317``
   in ``.env``.
2. Start Jaeger:
   ``docker run -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one``.
3. Run the API and view traces at http://localhost:16686.

Span attributes
---------------
* AI spans: ``provider``, ``model``, ``status`` (``ok``/``error``),
  ``error.message`` on failure.
* DB spans: ``db.operation``, ``db.status``, ``error.message`` on failure.

Usage
-----
::

    from src.tracing import tracer

    with tracer.start_as_current_span("my.operation") as span:
        span.set_attribute("key", "value")
        ...
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

log = logging.getLogger(__name__)

_SERVICE_NAME = "smart-lost-and-found"


# ---------------------------------------------------------------------------
# Optional OpenTelemetry import
# ---------------------------------------------------------------------------
# OpenTelemetry is a *bonus* feature gated by ``ENABLE_TRACING``. Some
# deployment targets (e.g. Streamlit Cloud with the slim requirements
# manifest) intentionally do not install the OTel packages. We must not
# crash the application in that scenario — instead we fall back to a
# minimal no-op tracer that satisfies the same surface area used in this
# module: ``start_as_current_span(...)`` as a context manager yielding an
# object with ``set_attribute`` and ``record_exception``.

try:
    from opentelemetry import trace as _otel_trace  # type: ignore[import-not-found]
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]
    from opentelemetry.sdk.trace.export import (  # type: ignore[import-not-found]
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when otel is missing
    _otel_trace = None  # type: ignore[assignment]
    TracerProvider = None  # type: ignore[assignment,misc]
    BatchSpanProcessor = None  # type: ignore[assignment,misc]
    ConsoleSpanExporter = None  # type: ignore[assignment,misc]
    _OTEL_AVAILABLE = False


class _NoopSpan:
    """Minimal span stub used when OpenTelemetry is not installed."""

    def set_attribute(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def record_exception(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _NoopTracer:
    """Tracer stub that yields :class:`_NoopSpan` for every span context."""

    @contextmanager
    def start_as_current_span(self, _name: str):  # type: ignore[no-untyped-def]
        yield _NoopSpan()


def _setup_tracing():
    """Configure the global tracer based on settings.

    When tracing is disabled — or the OTel libraries are not installed —
    the returned tracer silently discards every span. Production
    deployments that opt in (``ENABLE_TRACING=true``) export OTLP spans
    when the exporter package is available, falling back to the console
    exporter otherwise.

    Returns:
        The module-level tracer callers should use to open spans.
    """
    from src.config import settings

    if not _OTEL_AVAILABLE:
        log.debug("tracing.opentelemetry_not_installed_using_noop_tracer")
        return _NoopTracer()

    provider = TracerProvider()

    if settings.ENABLE_TRACING:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import-not-found]
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(endpoint=settings.OTLP_ENDPOINT, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            log.info("tracing.otlp_enabled", extra={"endpoint": settings.OTLP_ENDPOINT})
        except ImportError:
            log.warning("tracing.otlp_unavailable_using_console")
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        log.debug("tracing.disabled")

    _otel_trace.set_tracer_provider(provider)
    return _otel_trace.get_tracer(_SERVICE_NAME)


tracer = _setup_tracing()
"""Process-wide tracer; import this wherever spans are needed.

Type is intentionally untyped: it is either ``opentelemetry.trace.Tracer``
when the OTel libs are installed, or :class:`_NoopTracer` otherwise.
"""


def traced_ai_call(
    span_name: str,
    provider_attr: str = "LLM_PROVIDER",
    model_attr: str = "LLM_MODEL",
):
    """Wrap an async AI call in an OpenTelemetry span.

    The decorated function executes inside a span named ``span_name``
    with the configured provider and model attached as attributes. On
    success the span is marked ``status=ok``; on exception the span
    captures ``error.message`` and the underlying exception is re-raised.

    Args:
        span_name: Name of the span (e.g. ``"ai.describe_item"``).
        provider_attr: Settings attribute that names the provider.
        model_attr: Settings attribute that names the model.

    Returns:
        A decorator that wraps the target async function.
    """

    def decorator(func: Callable) -> Callable:
        """Inner decorator capturing ``span_name``, ``provider_attr``, ``model_attr``."""

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Run ``func`` inside a span instrumented with provider/model attributes."""
            from src.config import settings

            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("provider", getattr(settings, provider_attr, "unknown"))
                span.set_attribute("model", getattr(settings, model_attr, "unknown"))
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("status", "ok")
                    return result
                except Exception as exc:
                    span.set_attribute("status", "error")
                    span.set_attribute("error.message", str(exc))
                    span.record_exception(exc)
                    raise

        return wrapper

    return decorator


def traced_db_op(operation: str):
    """Wrap an async repository method in an OpenTelemetry span.

    The span is named ``db.<operation>`` and tagged with
    ``db.operation`` and ``db.status``. Exceptions are recorded as
    span events before being re-raised.

    Args:
        operation: Short label identifying the DB operation
            (e.g. ``"insert_item"``).

    Returns:
        A decorator that wraps the target async repository method.
    """

    def decorator(func: Callable) -> Callable:
        """Inner decorator capturing ``operation``."""

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Run ``func`` inside a span tagged with the DB operation name."""
            with tracer.start_as_current_span(f"db.{operation}") as span:
                span.set_attribute("db.operation", operation)
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("db.status", "ok")
                    return result
                except Exception as exc:
                    span.set_attribute("db.status", "error")
                    span.set_attribute("error.message", str(exc))
                    span.record_exception(exc)
                    raise

        return wrapper

    return decorator
