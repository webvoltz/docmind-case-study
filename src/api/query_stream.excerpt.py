# Excerpt: src/api/query.py and src/models/schemas.py
# Trimmed for the case study. "..." marks omitted code.

import json
from typing import Literal

import openai
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from services.rag_pipeline import stream_answer

QueryMode = Literal["auto", "fast", "deep", "exhaustive"]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The natural-language question")
    document_ids: list[str] | None = Field(
        default=None, description="Restrict search to these document IDs. None = search all."
    )
    # Bounded: a client can't ask fast mode to stuff 500 passages into the prompt.
    k: int = Field(default=5, ge=1, le=10)
    mode: QueryMode = "auto"


router = APIRouter(tags=["query"])


@router.post("/query/stream", response_class=StreamingResponse)
async def query_documents_stream(
    request: QueryRequest,
    session: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """
    Same pipeline as POST /query but streamed as Server-Sent Events:

    - event: mode      -> chosen mode and why
    - event: step      -> one per agent action (deep / exhaustive)
    - event: sources   -> numbered evidence the answer cites as [n]
    - event: token     -> answer text
    - event: citations -> markers used (with pages) + sentences the checker removed
    - event: done      -> latency, mode and budget spent
    - event: error     -> emitted if the LLM fails mid-stream
    """

    async def event_source():
        try:
            async for event, data in stream_answer(
                query=request.query,
                document_ids=request.document_ids,
                k=request.k,
                session=session,
                mode=request.mode,
            ):
                yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
        except openai.APIError:
            # Provider details stay in the server log; the client gets a stable message.
            yield f"event: error\ndata: {json.dumps({'detail': 'LLM API error'})}\n\n"
        except Exception:
            yield f"event: error\ndata: {json.dumps({'detail': 'Unexpected error'})}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        # No proxy buffering: each step reaches the browser as it happens.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
