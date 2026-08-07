"""Unit tests for entity normalisation service."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.model.llm import LLMStructuredError


class _MockRow:
    """Simulates a SQLAlchemy Row so ``row._mapping`` doesn't crash."""

    def __init__(self, data: dict):
        self._mapping = data  # Row._mapping is dict-like

    def __getitem__(self, key):
        return self._mapping[key]


def _make_session_factory(mock_session: AsyncMock, side_effect: list | None = None):
    """Build a mock session factory that returns *mock_session* inside
    ``async with`` blocks.

    The real ``get_session_factory()`` returns an ``async_sessionmaker``
    callable whose ``()`` returns an ``AsyncSession`` (an async context
    manager).  We mirror that with a ``MagicMock`` (not ``AsyncMock`` —
    ``()`` must return a plain value, not a coroutine) that returns a
    mock session set up with ``__aenter__`` / ``__aexit__``.
    """
    if side_effect is not None:
        mock_session.execute.side_effect = side_effect

    mock_sess = AsyncMock()
    mock_sess.__aenter__ = AsyncMock(return_value=mock_session)
    mock_sess.__aexit__ = AsyncMock(return_value=None)

    factory = MagicMock()
    factory.return_value = mock_sess
    return factory


class TestNormalizeEntities:
    """Tests for normalize_entities() — the core normalisation pipeline."""

    @pytest.mark.asyncio
    async def test_normalize_empty_entities_returns_empty(self):
        """Empty entity list returns an empty list immediately."""
        from backend.service.entity import normalize_entities

        result = await normalize_entities("some-id", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_normalize_new_entity_creates_record(self):
        """A brand-new entity with no similar existing entities creates a new row."""
        from backend.service.entity import normalize_entities

        with (
            patch(
                "backend.service.entity.get_embedding_provider"
            ) as mock_emb_provider,
            patch(
                "backend.service.entity.get_session_factory"
            ) as mock_sess_factory,
        ):
            mock_emb = AsyncMock()
            mock_emb.embed.return_value = [[0.1] * 1024]
            mock_emb_provider.return_value = mock_emb

            mock_session = AsyncMock()
            # SELECT similar → no candidates (MagicMock — fetchall is sync)
            mock_select = MagicMock()
            mock_select.fetchall.return_value = []
            # INSERT entity RETURNING id
            mock_insert = MagicMock()
            mock_insert.fetchone.return_value = [
                "11111111-1111-1111-1111-111111111111"
            ]
            mock_mem_ent = MagicMock()

            mock_sess_factory.return_value = _make_session_factory(
                mock_session,
                side_effect=[mock_select, mock_insert, mock_mem_ent],
            )

            entity_ids = await normalize_entities(
                "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                [{"name": "NewTech", "type": "technology"}],
            )

            assert len(entity_ids) == 1
            assert entity_ids[0] == "11111111-1111-1111-1111-111111111111"

    @pytest.mark.asyncio
    async def test_normalize_entity_llm_confirms_match(self):
        """When LLM confirms a match, link to the existing entity."""
        from backend.service.entity import normalize_entities

        with (
            patch(
                "backend.service.entity.get_embedding_provider"
            ) as mock_emb_provider,
            patch(
                "backend.service.entity.get_session_factory"
            ) as mock_sess_factory,
            patch(
                "backend.service.entity._llm_confirm_match"
            ) as mock_llm_confirm,
        ):
            mock_emb = AsyncMock()
            mock_emb.embed.return_value = [[0.1] * 1024]
            mock_emb_provider.return_value = mock_emb

            mock_llm_confirm.return_value = True

            existing_id = "22222222-2222-2222-2222-222222222222"

            mock_session = AsyncMock()
            mock_select = MagicMock()
            mock_select.fetchall.return_value = [
                _MockRow({
                    "id": existing_id,
                    "canonical_name": "PostgreSQL",
                    "name": "Postgres",
                    "type": "technology",
                    "similarity": 0.92,
                })
            ]
            mock_mem_ent = MagicMock()

            mock_sess_factory.return_value = _make_session_factory(
                mock_session,
                side_effect=[mock_select, mock_mem_ent],
            )

            entity_ids = await normalize_entities(
                "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                [{"name": "pg16", "type": "technology"}],
            )

            assert len(entity_ids) == 1
            assert entity_ids[0] == existing_id

    @pytest.mark.asyncio
    async def test_normalize_entity_llm_rejects_match(self):
        """When LLM rejects all candidates, create a new entity."""
        from backend.service.entity import normalize_entities

        with (
            patch(
                "backend.service.entity.get_embedding_provider"
            ) as mock_emb_provider,
            patch(
                "backend.service.entity.get_session_factory"
            ) as mock_sess_factory,
            patch(
                "backend.service.entity._llm_confirm_match"
            ) as mock_llm_confirm,
        ):
            mock_emb = AsyncMock()
            mock_emb.embed.return_value = [[0.1] * 1024]
            mock_emb_provider.return_value = mock_emb

            mock_llm_confirm.return_value = False

            new_id = "33333333-3333-3333-3333-333333333333"

            mock_session = AsyncMock()
            mock_select = MagicMock()
            mock_select.fetchall.return_value = [
                _MockRow({
                    "id": "22222222-2222-2222-2222-222222222222",
                    "canonical_name": "MySQL",
                    "name": "MySQL",
                    "type": "technology",
                    "similarity": 0.88,
                })
            ]
            mock_insert = MagicMock()
            mock_insert.fetchone.return_value = [new_id]
            mock_mem_ent = MagicMock()

            mock_sess_factory.return_value = _make_session_factory(
                mock_session,
                side_effect=[mock_select, mock_insert, mock_mem_ent],
            )

            entity_ids = await normalize_entities(
                "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                [{"name": "PostgreSQL", "type": "technology"}],
            )

            assert len(entity_ids) == 1
            assert entity_ids[0] == new_id

    @pytest.mark.asyncio
    async def test_normalize_entity_llm_rejects_match_creates_new(self):
        """When the LLM judges no match, a new entity is created."""
        from backend.service.entity import normalize_entities

        with (
            patch(
                "backend.service.entity.get_embedding_provider"
            ) as mock_emb_provider,
            patch(
                "backend.service.entity.get_session_factory"
            ) as mock_sess_factory,
            patch(
                "backend.service.entity._llm_confirm_match"
            ) as mock_llm_confirm,
        ):
            mock_emb = AsyncMock()
            mock_emb.embed.return_value = [[0.1] * 1024]
            mock_emb_provider.return_value = mock_emb

            # The LLM judges the candidate is NOT the same entity
            mock_llm_confirm.return_value = False

            new_id = "44444444-4444-4444-4444-444444444444"

            mock_session = AsyncMock()
            mock_select = MagicMock()
            mock_select.fetchall.return_value = [
                _MockRow({
                    "id": "22222222-2222-2222-2222-222222222222",
                    "canonical_name": "PostgreSQL",
                    "name": "Postgres",
                    "type": "technology",
                    "similarity": 0.95,
                })
            ]
            mock_insert = MagicMock()
            mock_insert.fetchone.return_value = [new_id]
            mock_mem_ent = MagicMock()

            mock_sess_factory.return_value = _make_session_factory(
                mock_session,
                side_effect=[mock_select, mock_insert, mock_mem_ent],
            )

            entity_ids = await normalize_entities(
                "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                [{"name": "pg", "type": "technology"}],
            )

            # Falls back to creating new entity (LLM returned False = no match)
            assert len(entity_ids) == 1
            assert entity_ids[0] == new_id


class TestLLMConfirmMatch:
    """_llm_confirm_match — the correctness-critical entity-match judgement."""

    @pytest.mark.asyncio
    async def test_returns_true_on_match(self, monkeypatch) -> None:
        from backend.service.entity import _llm_confirm_match

        mock_llm = AsyncMock()
        mock_llm.chat_json.return_value = '{"match": true}'
        monkeypatch.setattr("backend.service.structured.get_llm_provider", lambda: mock_llm)

        assert await _llm_confirm_match("pg16", "PostgreSQL", "technology") is True

    @pytest.mark.asyncio
    async def test_returns_false_on_no_match(self, monkeypatch) -> None:
        from backend.service.entity import _llm_confirm_match

        mock_llm = AsyncMock()
        mock_llm.chat_json.return_value = '{"match": false}'
        monkeypatch.setattr("backend.service.structured.get_llm_provider", lambda: mock_llm)

        assert await _llm_confirm_match("pg16", "MySQL", "technology") is False

    @pytest.mark.asyncio
    async def test_propagates_on_exhaustion(self, monkeypatch) -> None:
        """Never silently defaults to \"no match\" (which would create a
        duplicate entity) when the structured judgement fails."""
        from backend.service.entity import _llm_confirm_match

        async def _raise(*args, **kwargs):
            raise LLMStructuredError("no schema-valid JSON after retries")

        monkeypatch.setattr("backend.service.structured.chat_structured", _raise)

        with pytest.raises(LLMStructuredError):
            await _llm_confirm_match("pg16", "PostgreSQL", "technology")
