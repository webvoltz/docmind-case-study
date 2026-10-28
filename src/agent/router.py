# From: src/services/agent/router.py (complete, imports trimmed)
#
# Pick a mode for mode=auto.
#
#   fast        one retrieval pass — direct factual lookups
#   deep        tool loop — multi-hop, comparisons, "where/why/how", named sections
#   exhaustive  map-reduce over every section — "all / every / list each / whole document"
#
# Clear cases are decided by rules (no LLM call); ambiguous ones ask the cheap model.

from __future__ import annotations

import logging
import re
from typing import Literal

from config import settings
from services import llm

logger = logging.getLogger(__name__)

Mode = Literal["fast", "deep", "exhaustive"]

_EXHAUSTIVE_RE = re.compile(
    r"\b(list (all|every|each)|all (the )?(instances|mentions|occurrences|references|requirements|risks|"
    r"obligations|dates|deadlines|names|people|figures|tables|recommendations|findings)|every (time|mention|"
    r"instance|section|chapter)|each (section|chapter)|throughout (the )?(document|report|book|paper)|"
    r"(entire|whole|full) (document|report|book|paper|contract)|"
    r"summari[sz]e (the|this|each|every) (document|report|book|paper|contract|chapter)|"
    r"across (all|every|the) (chapters|sections|documents)|complete list|exhaustive)\b",
    re.IGNORECASE,
)
_DEEP_RE = re.compile(
    r"\b(compare|comparison|contrast|differ|difference|versus|vs\.?|why|how (does|do|did|is|are)|explain|"
    r"relationship|relate|impact|effect of|cause|trace|timeline|evolve|change[sd]? (over|between)|"
    r"(which|what) (chapter|section|page|appendix)|in (chapter|section|appendix) \w+|"
    r"according to (chapter|section)|consistent|contradict|both|step[- ]by[- ]step|"
    r"(section|chapter|appendix|part|clause|article) [0-9ivxlc]+(\.[0-9]+)*)\b",
    re.IGNORECASE,
)

_ROUTER_PROMPT = """\
Choose how to research this question about uploaded documents.
- "fast": a single lookup of one fact, definition or number.
- "deep": needs several lookups, reading a named section, comparing parts, or reasoning across passages.
- "exhaustive": needs every relevant item from the whole document (lists of all X, summaries of everything).

Question: {query}

Reply with JSON only: {{"mode": "fast" | "deep" | "exhaustive", "reason": "<short>"}}"""


def heuristic_mode(query: str) -> tuple[Mode | None, str]:
    q = query.strip()
    if _EXHAUSTIVE_RE.search(q):
        return "exhaustive", "asks for everything across the document"
    if _DEEP_RE.search(q):
        return "deep", "needs multi-step reading or comparison"
    if len(q.split()) <= 12:
        return "fast", "short direct question"
    return None, ""


async def route(query: str) -> tuple[Mode, str]:
    mode, reason = heuristic_mode(query)
    if mode:
        return mode, reason
    try:
        res = await llm.chat(
            [{"role": "user", "content": _ROUTER_PROMPT.format(query=query)}],
            model=settings.fast_model,
            max_tokens=80,
        )
        data = llm.parse_json(res.content)
        if isinstance(data, dict) and data.get("mode") in ("fast", "deep", "exhaustive"):
            return data["mode"], str(data.get("reason") or "router model")
    except Exception as exc:
        logger.warning("Router model failed (%s) — defaulting to deep", exc)
    # When unsure, spend a bounded amount more rather than answer from too little.
    return "deep", "default for longer questions"
