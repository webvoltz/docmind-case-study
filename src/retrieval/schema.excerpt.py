# Excerpt: src/db/database.py
# Trimmed for the case study. "..." marks omitted code.
#
# Four tables: documents -> pages, sections, chunks. Deleting a document
# cascades to everything it owns. Pages are stored whole so the agent can read
# them and every citation can point at a real page.

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, ForeignKey, Index, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from config import settings


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Indexing lifecycle: processing -> ready | failed. `stage` says where processing is.
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ready")
    stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    # False for DOCX/TXT, whose "pages" are ~3000-char virtual pages.
    has_real_pages: Mapped[bool] = ...


class Page(Base):
    __tablename__ = "pages"
    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    __table_args__ = (UniqueConstraint("document_id", "page_number", name="uq_pages_doc_page"),)


class Section(Base):
    __tablename__ = "sections"
    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    parent_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("sections.id", ondelete="CASCADE"), nullable=True
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)  # document order
    level: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)        # "Chapter 2 > 2.1 Scope"
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False, default="")  # own text, children excluded
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)


class Chunk(Base):
    __tablename__ = "chunks"
    document_id: Mapped[str] = ...
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dim), nullable=True)
    start_char: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    end_char: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("sections.id", ondelete="SET NULL"), nullable=True
    )
    section_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    # "Document: x.pdf\nSection: A > B" — embedded and reranked together with the text.
    header: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        # HNSW index for approximate nearest-neighbour search (cosine distance).
        # Unlike IVFFlat it needs no training data, so recall stays good on small/new tables.
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Detect an embedding column whose dimension doesn't match the configured model.
        # We never wipe embeddings at startup: re-embedding is an explicit step
        # (`python scripts/reindex.py`) so a config typo can't destroy the index.
        # Until then, uploads return 409 instead of writing vectors of the wrong size.
        ...

        # Idempotent migrations: ALTER TABLE ... ADD COLUMN IF NOT EXISTS, then create_all.
        ...

        # Full-text GIN indexes on chunks (hybrid search) and pages.
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_chunks_text_gin ON chunks USING GIN (to_tsvector('english', text))"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_pages_text_gin ON pages USING GIN (to_tsvector('english', text))"
        ))
