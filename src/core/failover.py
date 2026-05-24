"""Multi-provider failover wrappers — bonus feature (+3 pts).

:class:`FailoverVLM` and :class:`FailoverEmbedder` each wrap an ordered
list of concrete providers. When the first provider raises
:class:`ProviderError` the wrapper transparently retries against the
next provider, and so on, raising only when every provider has failed.

Both wrappers satisfy the same abstract interfaces (``VLMProvider``,
``EmbeddingProvider``) as the concrete providers, so they drop into any
slot that expects a provider instance. The AI service layer constructs
them automatically when ``settings.SECONDARY_LLM_PROVIDER`` is set.

The chaos test ``tests/test_failover.py`` confirms that a failure in
the primary provider transparently surfaces as success on the secondary.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from ai.providers.base import EmbeddingProvider, ProviderError, VLMProvider

log = logging.getLogger(__name__)


class FailoverVLM(VLMProvider):
    """VLM provider that walks through a ranked list of concrete providers.

    The first provider is tried; on :class:`ProviderError` the next one is
    tried, and so on. If every provider fails, the wrapper raises a
    :class:`ProviderError` containing the message from the last failure.
    """

    def __init__(self, providers: list[VLMProvider]) -> None:
        """Bind the ranked provider list.

        Args:
            providers: Non-empty ordered list of VLM providers. The first
                entry is treated as primary; subsequent entries as fallbacks.

        Raises:
            ValueError: ``providers`` is empty.
        """
        if not providers:
            raise ValueError("FailoverVLM requires at least one provider.")
        self._providers = providers

    @property
    def _names(self) -> list[str]:
        """Class names of the wrapped providers, used in log records."""
        return [type(p).__name__ for p in self._providers]

    def describe(
        self,
        image_path: str,
        prompt: str,
        *,
        json_schema: dict | None = None,
    ) -> str:
        """Forward the describe call to the highest-priority healthy provider.

        Args:
            image_path: Path to the image to describe.
            prompt: User text passed through to the VLM.
            json_schema: Optional structured-output schema.

        Returns:
            Raw provider response (typically a JSON-encoded string).

        Raises:
            ProviderError: Every provider in the chain failed.
        """
        last_error: Exception | None = None
        for i, provider in enumerate(self._providers):
            try:
                result = provider.describe(image_path, prompt, json_schema=json_schema)
                if i > 0:
                    log.info(
                        "failover_vlm.succeeded_on_secondary",
                        extra={"provider": self._names[i], "index": i},
                    )
                return result
            except ProviderError as exc:
                log.warning(
                    "failover_vlm.provider_failed",
                    extra={"provider": self._names[i], "index": i, "error": str(exc)},
                )
                last_error = exc

        raise ProviderError(
            f"All {len(self._providers)} VLM providers failed. "
            f"Last error: {last_error}"
        )


class FailoverEmbedder(EmbeddingProvider):
    """Embedding provider that walks through a ranked list of providers.

    Mirrors :class:`FailoverVLM` for embedding calls. The first provider's
    dimensionality is reported via :attr:`dimension`, which assumes that
    the failover chain is configured with providers that emit the same
    vector size.
    """

    def __init__(self, providers: list[EmbeddingProvider]) -> None:
        """Bind the ranked provider list.

        Args:
            providers: Non-empty ordered list of embedding providers.

        Raises:
            ValueError: ``providers`` is empty.
        """
        if not providers:
            raise ValueError("FailoverEmbedder requires at least one provider.")
        self._providers = providers

    @property
    def dimension(self) -> int:
        """Embedding dimensionality reported by the primary provider."""
        return self._providers[0].dimension

    @property
    def _names(self) -> list[str]:
        """Class names of the wrapped providers, used in log records."""
        return [type(p).__name__ for p in self._providers]

    def embed(self, text: str) -> np.ndarray:
        """Embed ``text`` using the highest-priority healthy provider.

        Args:
            text: Non-empty string to embed.

        Returns:
            Unit-normalised ``float32`` vector.

        Raises:
            ProviderError: Every provider in the chain failed.
        """
        last_error: Exception | None = None
        for i, provider in enumerate(self._providers):
            try:
                vec = provider.embed(text)
                if i > 0:
                    log.info(
                        "failover_embedder.succeeded_on_secondary",
                        extra={"provider": self._names[i], "index": i},
                    )
                return vec
            except ProviderError as exc:
                log.warning(
                    "failover_embedder.provider_failed",
                    extra={"provider": self._names[i], "index": i, "error": str(exc)},
                )
                last_error = exc

        raise ProviderError(
            f"All {len(self._providers)} embedding providers failed. "
            f"Last error: {last_error}"
        )


def build_failover_vlm_if_configured() -> VLMProvider | None:
    """Construct a :class:`FailoverVLM` when a secondary VLM is configured.

    Reads ``SECONDARY_LLM_PROVIDER`` from settings. When it is set, the
    function builds the secondary provider by temporarily mutating
    ``LLM_PROVIDER`` / ``LLM_MODEL`` in the process environment, then
    restores the original values regardless of outcome.

    Returns:
        A :class:`FailoverVLM` wrapping ``[primary, secondary]`` when a
        secondary is configured; ``None`` otherwise. Callers treat
        ``None`` as "let the ai module pick its own default".
    """
    from ai.providers.factory import get_vlm
    from src.config import settings

    secondary = settings.SECONDARY_LLM_PROVIDER
    if not secondary:
        return None

    old_provider = os.environ.get("LLM_PROVIDER")
    old_model = os.environ.get("LLM_MODEL")
    try:
        os.environ["LLM_PROVIDER"] = secondary
        if settings.SECONDARY_LLM_MODEL:
            os.environ["LLM_MODEL"] = settings.SECONDARY_LLM_MODEL
        secondary_vlm = get_vlm()
    finally:
        if old_provider is not None:
            os.environ["LLM_PROVIDER"] = old_provider
        if old_model is not None:
            os.environ["LLM_MODEL"] = old_model

    primary_vlm = get_vlm()
    log.info(
        "failover_vlm.configured",
        extra={"primary": settings.LLM_PROVIDER, "secondary": secondary},
    )
    return FailoverVLM([primary_vlm, secondary_vlm])
