# Excerpt: src/services/indexer.py
# Trimmed for the case study. "..." marks omitted code.
#
# Background indexing. Upload returns 202 as soon as the file is stored; this
# module does the slow part and records progress on the document row, which
# GET /api/documents/{id}/status exposes.
#
# Stages: parsing -> structuring -> embedding -> summarizing -> ready (or failed).

from __future__ import annotations

import asyncio
import logging
from typing import Any

from config import settings
from db.database import AsyncSessionLocal
from services import parser as parser_module
from services import structure, summarizer, vector_store
from services.chunker import TokenChunker
from services.embedder import embedder, embedding_text

logger = logging.getLogger(__name__)

_running: set[asyncio.Task[Any]] = set()
_semaphore: asyncio.Semaphore | None = None


def _sem() -> asyncio.Semaphore:
    # One document at a time by default: the embedder and reranker share one GPU.
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(max(1, settings.max_concurrent_indexing))
    return _semaphore


def schedule(document_id: str, raw_bytes: bytes, filename: str, content_type: str) -> None:
    """Start indexing in the background (keeps a reference so the task isn't GC'd)."""
    task = asyncio.create_task(index_document(document_id, raw_bytes, filename, content_type))
    _running.add(task)
    task.add_done_callback(_running.discard)


async def index_document(document_id: str, raw_bytes: bytes, filename: str, content_type: str) -> None:
    async with _sem():
        try:
            await _index(document_id, raw_bytes, filename, content_type)
        except Exception as exc:
            # A failed job is visible to the client, never silently stuck.
            logger.exception("Indexing failed for %s (%s)", filename, document_id)
            await _set(document_id, status="failed", stage=None, error=str(exc)[:2000])


async def _index(document_id: str, raw_bytes: bytes, filename: str, content_type: str) -> None:
    await _set(document_id, status="processing", stage="parsing", error=None)
    parsed = await parser_module.parse_bytes(raw_bytes, filename, content_type)
    if not parsed.text.strip():
        raise ValueError("The file contains no extractable text.")

    # --- structure -------------------------------------------------------
    await _set(document_id, stage="structuring", page_count=parsed.page_count,
               file_type=parsed.file_type, has_real_pages=parsed.has_real_pages)
    full_text, offsets = structure.assemble(parsed.pages)
    located = structure.locate_headings(full_text, offsets, parsed.headings)
    sections = structure.build_sections(full_text, offsets, located)
    chunker = TokenChunker(settings.chunk_size, settings.chunk_overlap, token_counter=embedder.count_tokens)
    chunks = structure.chunk_sections(full_text, offsets, sections, chunker, filename)
    if not chunks:
        raise ValueError("The document produced zero chunks.")

    async with AsyncSessionLocal() as session:
        await vector_store.save_pages(session, document_id, parsed.pages)
        section_ids = await vector_store.save_sections(session, document_id, [...])
        await session.commit()

    # --- embed + store chunks -------------------------------------------
    await _set(document_id, stage="embedding")
    for c in chunks:
        c["section_id"] = section_ids.get(c["section_ordinal"])
    embeddings = await embedder.embed([embedding_text(c["header"], c["text"]) for c in chunks])
    async with AsyncSessionLocal() as session:
        await vector_store.save_chunks(session, chunks, embeddings, document_id)
        await session.commit()

    # --- summaries (search already works while these are written) -------
    await _set(document_id, stage="summarizing")
    by_ordinal, doc_summary = await summarizer.summarize(filename, full_text, sections)
    async with AsyncSessionLocal() as session:
        await vector_store.set_section_summaries(
            session, {section_ids[o]: s for o, s in by_ordinal.items() if o in section_ids}
        )
        await vector_store.update_document(
            session, document_id, summary=doc_summary or None, status="ready", stage=None
        )
        await session.commit()


# On startup, main.py calls vector_store.fail_interrupted(): any document still
# "processing" after a restart is marked failed with "upload again", instead of
# showing a progress bar that will never move.
