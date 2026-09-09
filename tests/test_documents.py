from pathlib import Path

from native_rag.documents import chunk_text, load_chunks


def test_chunking_has_overlap_and_offsets() -> None:
    text = "Абзац один. " * 80
    chunks = chunk_text(text, max_chars=180, overlap=40)
    assert len(chunks) > 1
    assert all(chunk for chunk, _, _, _ in chunks)
    assert all(0 <= start < end <= len(text) for _, start, end, _ in chunks)
    assert any(set(chunks[i][0].split()) & set(chunks[i + 1][0].split()) for i in range(len(chunks) - 1))


def test_loader_is_deterministic_and_preserves_source(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")
    (tmp_path / "a.md").write_text("# A\n\nText", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("not a document", encoding="utf-8")
    chunks = load_chunks(tmp_path, max_chars=100, overlap=10)
    assert [c.id for c in chunks] == list(range(len(chunks)))
    assert [c.source for c in chunks] == ["a.md", "b.txt"]
    assert chunks[0].heading == "A"


def test_chunking_carries_active_section_heading() -> None:
    text = "# Installation\n\n" + ("Install the package carefully. " * 20)
    text += "\n\n## Configuration\n\n" + ("Configure the timeout value. " * 20)

    chunks = chunk_text(text, max_chars=180, overlap=20)

    assert any(heading == "Installation" for _, _, _, heading in chunks)
    assert any(heading == "Configuration" for _, _, _, heading in chunks)
