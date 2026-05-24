"""Application-services package.

Modules in this package orchestrate calls to the provided ``ai`` library
and surrounding side-effects (retries, timeouts, caching, rate limiting,
cost telemetry). Service functions are the only place AI calls are made
from; presentation layers (API, CLI, UI) depend on this package, not on
the ``ai`` module directly.
"""
