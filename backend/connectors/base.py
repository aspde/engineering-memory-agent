"""Connector abstract base class.

Every external data source adapter implements these four methods.
The ``process()`` method has a sensible default that writes directly
to the memory pipeline — individual connectors can override it when
they need custom storage logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Connector(ABC):
    """Abstract connector for an external data source.

    Subclasses must provide:
      - ``source_type`` — a unique string used as ``memories.source_type``
      - ``display_name`` — human-readable label (e.g. "Jira", "CI/CD")
      - ``validate(payload)`` — check whether the webhook payload is well-formed
      - ``normalize(payload)`` — transform the payload into EMA's standard
        content text format

    Subclasses may override:
      - ``supports_batch`` — set to True when the connector implements
        a real ``normalize_batch()`` (default loops over ``normalize()``)
      - ``process(content, metadata)`` — only needed when the connector
        needs custom storage logic beyond a simple ``write_memory()`` call
    """

    # ── Subclass MUST set these ──────────────────────────────────────

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Unique source_type value written to ``memories.source_type``."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable label shown in the frontend."""
        ...

    # ── Subclass MUST implement these ─────────────────────────────────

    @abstractmethod
    def validate(self, payload: dict[str, Any]) -> bool:
        """Return True if *payload* is a well-formed webhook body.

        This is a pure synchronous check — no IO, no side effects.
        """
        ...

    @abstractmethod
    def normalize(self, payload: dict[str, Any]) -> str:
        """Transform a validated *payload* into EMA-standard content text.

        The returned string is what gets stored as the memory content
        (and later fed into ``extract_memory()`` for structured extraction).

        This is a pure data transformation — no IO, no side effects.
        """
        ...

    # ── Subclass MAY override these ───────────────────────────────────

    def build_metadata(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Extract traceability metadata from *payload*.

        Called by the webhook route before ``process()``.  The returned
        dict is passed as the *metadata* argument to ``process()``.
        Default returns an empty dict — override to capture source URLs,
        issue keys, commit SHAs, etc.
        """
        return {}

    @property
    def supports_batch(self) -> bool:
        """Whether this connector has a real ``normalize_batch()`` impl.

        Default is False — the base ``normalize_batch()`` just loops over
        ``normalize()``.  Set to True once a connector implements true
        batch normalisation (Phase 3).
        """
        return False

    @property
    def batch_mode(self) -> str:
        """Batch normalisation readiness.

        Returns one of:
        - ``"supported"`` — true batch ``normalize_batch()`` implemented
        - ``"pending"`` — batch would help but not yet implemented
        - ``"not_applicable"`` — batch doesn't make sense for this connector

        The default checks ``supports_batch`` to decide between
        ``"supported"`` and ``"pending"``.  Override and return
        ``"not_applicable"`` when batch is irrelevant.
        """
        return "supported" if self.supports_batch else "pending"

    def normalize_batch(self, payloads: list[dict[str, Any]]) -> list[str]:
        """Normalize a batch of payloads — default loops over ``normalize()``.

        Connectors that benefit from true batch processing (e.g. shared
        context across payloads, single embedding call) should override
        this method and set ``supports_batch = True``.
        """
        return [self.normalize(p) for p in payloads]

    async def process(self, content: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Write *content* into EMA's memory pipeline.

        The default implementation calls ``write_memory()``.  Override
        when a connector needs custom storage (e.g. writing chunks in
        addition to memories, or using a specific similarity threshold).
        """
        from backend.service.memory import write_memory

        return await write_memory(
            content,
            source_type=self.source_type,
            metadata=metadata,
        )
