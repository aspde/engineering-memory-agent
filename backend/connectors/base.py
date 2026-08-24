"""Connector abstract base class.

Every external data source adapter implements these four methods.
The ``prepare()`` method produces storage-ready data — individual
connectors override it when they need enrichment or derived source
types.  Writing to the memory pipeline belongs to the caller (the
webhook route), keeping this package free of storage-layer imports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Connector(ABC):
    """Abstract connector for an external data source.

    Subclasses must provide:
      - ``source_type`` — a unique string used as ``memories.source_type``
      - ``display_name`` — human-readable label (e.g. "PingCode", "CI/CD")
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

    # ── Event-analysis capability (Phase 3 event-driven response) ────

    @property
    def triggers_event_analysis(self) -> bool:
        """Whether ingested events should trigger agent analysis.

        When True, the webhook path spawns an analysis run after the
        delivery reaches a terminal state (see
        ``backend/runner/event_analysis.py``).  Default False — only
        connectors whose events warrant an immediate historical lookup
        opt in (currently CI failures).
        """
        return False

    @property
    def event_analysis_cooldown_key(self) -> str | None:
        """Metadata field whose value keys the per-entity cooldown gate.

        Only meaningful when ``triggers_event_analysis`` is True.  For CI
        this is ``"job_name"`` — repeated failures of the same job within
        the cooldown window are analysed once.  None disables the gate
        (every event is analysed).
        """
        return None

    @property
    def event_analysis_prompt_key(self) -> str | None:
        """Registry key of this connector's analysis System Prompt.

        Only meaningful when ``triggers_event_analysis`` is True.  Declared
        per-connector (rather than derived from ``source``) because a
        connector may ingest several memory types — the prompt describes
        the *analysis task* ("judge this failed build"), not the source.
        """
        return None

    @property
    def event_analysis_display(self) -> dict[str, Any]:
        """How the generic analysis UI renders this connector's events.

        Two optional keys:

        - ``"title_entity"`` — metadata field shown in the Feishu card
          title (e.g. ``"job_name"`` for CI, ``"item_id"`` for PingCode).
          Absent → the title is just the source name.
        - ``"context_fields"`` — list of metadata fields appended to the
          user message as context lines (CI: ``["branch", "source_url"]``).
          Absent → no context lines.

        Declared here so the runner and card renderer stay source-agnostic;
        a new opted-in connector gets a sensible card without touching
        event-analysis code.
        """
        return {}

    def normalize_batch(self, payloads: list[dict[str, Any]]) -> list[str]:
        """Normalize a batch of payloads — default loops over ``normalize()``.

        Connectors that benefit from true batch processing (e.g. shared
        context across payloads, single embedding call) should override
        this method and set ``supports_batch = True``.
        """
        return [self.normalize(p) for p in payloads]

    async def prepare(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> tuple[str, str, dict[str, Any] | None]:
        """Produce storage-ready ``(content, source_type, metadata)``.

        The default implementation is a plain pass-through tagged with
        this connector's ``source_type``.  Override when a connector
        needs enrichment or a derived source type (e.g. CI regression
        detection).  The caller — the webhook route — owns the actual
        ``write_memory()`` call.
        """
        return content, self.source_type, metadata
