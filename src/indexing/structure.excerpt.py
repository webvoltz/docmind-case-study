# Excerpt: src/services/structure.py
# Trimmed for the case study. "..." marks omitted code.
#
# Turns a parsed document into a section tree and section-aware chunks.
# Everything here is pure (no database, no models), so it is tested directly:
#
#   pages + headings
#       -> assemble():        one string with known page start offsets
#       -> locate_headings(): heading -> character offset in that string
#       -> build_sections():  tree with parent links, page ranges and own body span
#       -> chunk_sections():  chunks that never cross a section boundary, each with
#                             a "Document / Section path" header and a page range

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any

from services.chunker import TokenChunker

PAGE_SEP = "\n\n"
FRONT_MATTER_MIN_CHARS = 200
FALLBACK_PAGES_PER_SECTION = 5


@dataclass
class SectionNode:
    ordinal: int
    level: int
    title: str
    parent: int | None
    start: int          # heading position (start of the section incl. children)
    end: int            # end of the section incl. children
    body_start: int     # own text: from the heading to the next heading of any level
    body_end: int
    page_start: int = 1
    page_end: int = 1
    path: list[str] = field(default_factory=list)

    @property
    def path_str(self) -> str:
        return " > ".join(self.path)


def assemble(pages: list[str]) -> tuple[str, list[int]]:
    """Join pages and return (full_text, start offset of each page)."""
    offsets: list[int] = []
    parts: list[str] = []
    pos = 0
    for page in pages:
        offsets.append(pos)
        parts.append(page)
        pos += len(page) + len(PAGE_SEP)
    return PAGE_SEP.join(parts), offsets


def page_at(offsets: list[int], pos: int) -> int:
    """1-based page number containing character *pos*. A binary search, not a scan."""
    return max(1, bisect.bisect_right(offsets, pos))


def build_sections(
    full_text: str, offsets: list[int], located: list[tuple[int, int, str]]
) -> list[SectionNode]:
    """Build the section tree. Without headings, fall back to page-group sections."""
    if not located:
        return _page_group_sections(full_text, offsets)

    located = list(located)
    preamble = full_text[: located[0][0]]
    if len(preamble.strip()) >= FRONT_MATTER_MIN_CHARS:
        located.insert(0, (0, min(l for _, l, _ in located), "Front matter"))
    elif located[0][0] > 0:
        # Short preamble (title page etc.) folds into the first section.
        first = located[0]
        located[0] = (0, first[1], first[2])

    nodes: list[SectionNode] = []
    stack: list[int] = []
    for i, (pos, level, title) in enumerate(located):
        # A heading closes every open section at its level or deeper.
        while stack and nodes[stack[-1]].level >= level:
            stack.pop()
        parent = stack[-1] if stack else None
        body_end = located[i + 1][0] if i + 1 < len(located) else len(full_text)
        node = SectionNode(
            ordinal=i, level=level, title=title.strip()[:500], parent=parent,
            start=pos, end=len(full_text), body_start=pos, body_end=body_end,
        )
        node.path = (nodes[parent].path if parent is not None else []) + [node.title]
        nodes.append(node)
        stack.append(i)

    for i, node in enumerate(nodes):
        for later in nodes[i + 1:]:
            if later.level <= node.level:
                node.end = later.start
                break
        node.page_start = page_at(offsets, node.start)
        node.page_end = page_at(offsets, max(node.start, node.end - 1))
    return nodes


def _page_group_sections(full_text: str, offsets: list[int]) -> list[SectionNode]:
    """No outline and no detectable headings: group every 5 pages into a section."""
    ...


def chunk_header(filename: str, section_path: str) -> str:
    return f"Document: {filename}\nSection: {section_path}" if section_path else f"Document: {filename}"


def chunk_sections(
    full_text: str,
    offsets: list[int],
    sections: list[SectionNode],
    chunker: TokenChunker,
    filename: str,
) -> list[dict[str, Any]]:
    """Chunk each section's own body; chunks carry header, section ordinal and page range."""
    chunks: list[dict[str, Any]] = []
    for node in sections:
        body = full_text[node.body_start:node.body_end]
        # Skip bodies that are just the heading line.
        if len(body.strip()) - len(node.title) < 10:
            continue
        # Spans are offsets into full_text, so every chunk maps back to exact pages.
        for start, end in chunker.split_spans(body, offset=node.body_start):
            chunks.append({
                "text": full_text[start:end],
                "chunk_index": len(chunks),
                "start_char": start,
                "end_char": end,
                "page_start": page_at(offsets, start),
                "page_end": page_at(offsets, max(start, end - 1)),
                "section_ordinal": node.ordinal,
                "section_path": node.path_str,
                # Embedded and reranked together with the text, so a chunk that
                # says "the fee is waived" still knows it lives under "Termination".
                "header": chunk_header(filename, node.path_str),
            })
    return chunks
