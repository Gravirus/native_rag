from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}
_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*$")


@dataclass(frozen=True)
class DocumentChunk:
    id: int
    source: str
    text: str
    start_char: int
    end_char: int
    heading: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def iter_document_files(root: Path) -> Iterable[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"documents directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(root)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def _boundary(text: str, start: int, preferred_end: int) -> int:
    """Choose a readable boundary without letting a long paragraph stall chunking."""
    if preferred_end >= len(text):
        return len(text)
    search_start = start + max(1, int((preferred_end - start) * 0.55))
    candidates = [
        text.rfind("\n\n", search_start, preferred_end),
        text.rfind(". ", search_start, preferred_end),
        text.rfind("! ", search_start, preferred_end),
        text.rfind("? ", search_start, preferred_end),
        text.rfind(" ", search_start, preferred_end),
    ]
    boundary = max(candidates)
    if boundary <= start:
        return preferred_end
    if text[boundary:boundary + 2] == "\n\n":
        return boundary + 2
    if text[boundary:boundary + 2] in {". ", "! ", "? "}:
        return boundary + 1
    return boundary


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> list[tuple[str, int, int, str | None]]:
    """Return paragraph/sentence-aware chunks with section metadata.

    Chunks prefer blank lines and sentence boundaries, but a section heading is
    also treated as a soft boundary.  The active heading is carried as metadata
    so retrieval can weight it without duplicating the heading in model input.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be in [0, max_chars)")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return []

    headings = list(_HEADING_RE.finditer(text))
    result: list[tuple[str, int, int, str | None]] = []
    start = 0
    while start < len(text):
        while start < len(text) and text[start].isspace():
            start += 1
        if start >= len(text):
            break
        preferred_end = min(start + max_chars, len(text))
        next_heading = next((match.start() for match in headings if match.start() > start), None)
        if next_heading is not None and next_heading < preferred_end and next_heading > start:
            preferred_end = next_heading
        end = _boundary(text, start, preferred_end)
        while end > start and text[end - 1].isspace():
            end -= 1
        if end <= start:
            end = min(start + max_chars, len(text))
        active_headings = [match for match in headings if match.start() <= start]
        if active_headings:
            heading = active_headings[-1].group(1).strip()
        else:
            in_chunk = next((match for match in headings if start <= match.start() < end), None)
            heading = in_chunk.group(1).strip() if in_chunk else None
        result.append((text[start:end].strip(), start, end, heading))
        if end >= len(text):
            break
        next_start = max(start + 1, end - overlap)
        while next_start < end and not text[next_start].isspace():
            next_start += 1
        start = next_start
    return result


def load_chunks(root: Path, max_chars: int = 1200, overlap: int = 200) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    root = Path(root)
    for path in iter_document_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        source = path.relative_to(root).as_posix()
        for chunk, start, end, heading in chunk_text(text, max_chars, overlap):
            chunks.append(DocumentChunk(len(chunks), source, chunk, start, end, heading))
    if not chunks:
        raise ValueError(f"no .md/.markdown/.txt documents found in {root}")
    return chunks
