"""Database schema via SQLAlchemy ORM declarations.

Tables:
  - chunks:  document fragments with pgvector embeddings
  - memories: structured long-term memories with entities, relations, decay
"""

from __future__ import annotations

import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import Float, Integer, Text, text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMPTZ, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    __abstract__ = True


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    document_id: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding = mapped_column(Vector(1024), nullable=True)
    metadata: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'"))
    created_at = mapped_column(TIMESTAMPTZ(timezone=True), server_default=text("now()"))
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    source_type: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    entities: Mapped[dict] = mapped_column(JSONB, server_default=text("'[]'"))
    relations: Mapped[dict] = mapped_column(JSONB, server_default=text("'[]'"))
    embedding = mapped_column(Vector(1024), nullable=True)
    decay_factor: Mapped[float] = mapped_column(Float, server_default=text("1.0"))
    recalled_at = mapped_column(TIMESTAMPTZ(timezone=True), server_default=text("now()"))
    recall_count: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    metadata: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'"))
    created_at = mapped_column(TIMESTAMPTZ(timezone=True), server_default=text("now()"))
