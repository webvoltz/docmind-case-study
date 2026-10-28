# Excerpt: src/services/vector_store.py and src/services/retrieval.py
# Trimmed for the case study. "..." marks omitted code.
#
# Retrieval: embed the query -> hybrid search (pgvector HNSW + Postgres full-text,
# fused by Reciprocal Rank Fusion) -> cross-encoder rerank -> top-k.
# Shared by fast mode and the deep agent's `search` tool.

from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy import String, bindparam, func, literal_column, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from db.database import Chunk, Document
from services.embedder import embedder, embedding_text

# --- vector_store.py --------------------------------------------------------


async def similarity_search(
    session: AsyncSession, query_embedding: list[float], k: int = 5, document_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Top-k chunks by cosine similarity (HNSW)."""
    distance = Chunk.embedding.cosine_distance(query_embedding)
    # HNSW only returns ef_search candidates; keep it above the requested k.
    ef_search = max(int(settings.hnsw_ef_search), k)
    await session.execute(text(f"SET LOCAL hnsw.ef_search = {ef_search}"))

    stmt = (
        select(*_HIT_COLUMNS, (1 - distance).label("similarity"))
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.embedding.isnot(None))
        .order_by(distance)
        .limit(k)
    )
    if document_ids:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))
    rows = (await session.execute(stmt)).fetchall()
    return [_hit(r, similarity=float(r.similarity)) for r in rows]


async def lexical_search(
    session: AsyncSession, query: str, k: int = 5, document_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Postgres full-text search (websearch_to_tsquery + ts_rank_cd)."""
    lang = literal_column("'english'")
    tsvec = func.to_tsvector(lang, Chunk.text)
    # The user's words are a bound parameter, never formatted into SQL.
    tsquery = func.websearch_to_tsquery(lang, bindparam("fts_query", value=query, type_=String))
    rank_col = func.ts_rank_cd(tsvec, tsquery).label("fts_rank")
    stmt = ...  # same shape as above, ordered by rank_col

    try:
        # Savepoint: a bad tsquery must not abort the caller's transaction.
        async with session.begin_nested():
            rows = (await session.execute(stmt)).fetchall()
    except Exception:
        return []
    return [_hit(r, fts_rank=float(r.fts_rank)) for r in rows]


async def hybrid_search(
    session: AsyncSession,
    query: str,
    query_embedding: list[float],
    k: int = 5,
    document_ids: list[str] | None = None,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Vector + full-text search fused with Reciprocal Rank Fusion: score = Σ 1/(rrf_k + rank + 1)."""
    fetch_k = min(k * 2, 200)
    vector_hits = await similarity_search(session, query_embedding, k=fetch_k, document_ids=document_ids)
    lexical_hits = await lexical_search(session, query, k=fetch_k, document_ids=document_ids)

    # RRF uses ranks, not raw scores, so cosine similarity and ts_rank never
    # have to be put on the same scale.
    scores: dict[str, float] = {}
    data: dict[str, dict[str, Any]] = {}
    for hits in (vector_hits, lexical_hits):
        for rank, hit in enumerate(hits):
            key = hit["chunk_id"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            data.setdefault(key, hit)

    top = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
    return [{**data[key], "score": round(scores[key], 6)} for key in top]


# --- retrieval.py -----------------------------------------------------------


def rerank(query: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort chunks by cross-encoder score (blocking; call via a thread). Falls back to input order."""
    cross_encoder = _get_cross_encoder()  # BAAI/bge-reranker-v2-m3, fp16 on MPS / CUDA
    if cross_encoder is None or not chunks:
        return chunks
    try:
        pairs = [(query, embedding_text(c.get("header"), c["text"])) for c in chunks]
        scores = [float(s) for s in cross_encoder.predict(pairs, batch_size=settings.reranker_batch_size)]
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [{**c, "rerank_score": round(s, 4)} for s, c in ranked]
    except Exception:
        return chunks


async def retrieve(
    session: AsyncSession, query: str, document_ids: list[str] | None, k: int,
) -> list[dict[str, Any]]:
    """Top-k chunks for *query* (each dict has text, header, pages, section, filename, score)."""
    # Qwen3-Embedding: an instruction prefix on queries only, never on documents.
    query_embedding = (await embedder.embed([query], input_type="query"))[0]
    pool = max(settings.retrieval_k, k)  # 50 candidates by default
    candidates = await hybrid_search(
        session=session, query=query, query_embedding=query_embedding,
        k=pool, document_ids=document_ids, rrf_k=settings.rrf_k,
    )
    # The cross-encoder is GPU/CPU-bound; run it in a worker thread so other requests keep flowing.
    ranked = await asyncio.to_thread(rerank, query, candidates)
    return ranked[:k]
