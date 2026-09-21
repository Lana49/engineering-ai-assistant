"""One presentation boundary for answers backed by retrieved document excerpts.

Evidence selection/validation belongs to the QA engine.  This module never
generates a claim, a document title or a clause number from a model response.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


INSUFFICIENT_EVIDENCE = (
    "В загруженных документах не найдено достаточного подтверждения для точного ответа."
)

_ABBREVIATIONS = {
    "г.", "гг.", "п.", "пп.", "рис.", "табл.", "стр.", "ст.", "см.",
    "им.", "т.е.", "т.д.", "т.п.", "т.к.", "др.", "пр.", "ед.", "ч.",
}


def evidence_key(text: str) -> str:
    """Compare only case and whitespace; keep numbers, punctuation and negation."""
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def _sentences(text: str) -> list[str]:
    """Conservative boundaries that do not split clause IDs or abbreviations."""
    sentences: list[str] = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+", text):
        prefix = text[start:match.start()]
        token = prefix.rsplit(None, 1)[-1].casefold() if prefix.strip() else ""
        if token in _ABBREVIATIONS or re.fullmatch(r"[а-яёa-z]\.", token):
            continue
        # A line-leading '7.2.2.' or list marker is not a whole sentence.
        if re.fullmatch(r"\s*\d+(?:\.\d+)*\.", prefix):
            continue
        sentences.append(prefix.strip())
        start = match.end()
    if text[start:].strip():
        sentences.append(text[start:].strip())
    return sentences


def compact_evidence(text: str) -> str:
    """Remove exact repeated paragraphs/sentences without shortening conditions.

    No fuzzy similarity or sentence-count cutoff is used: an exception near
    the end of a long normative paragraph must remain attached to its rule.
    Sentence deduplication stays within a paragraph, so an identical sentence
    under a different numbered clause/condition is not silently discarded.
    """
    paragraphs: list[str] = []
    seen_paragraphs: set[str] = set()
    for raw in re.split(r"\n\s*\n", str(text or "")):
        paragraph = re.sub(r"\s+", " ", raw).strip()
        if not paragraph:
            continue
        unique_sentences: list[str] = []
        seen_sentences: set[str] = set()
        for sentence in _sentences(paragraph):
            key = evidence_key(sentence)
            if key not in seen_sentences:
                seen_sentences.add(key)
                unique_sentences.append(sentence)
        paragraph = " ".join(unique_sentences)
        key = evidence_key(paragraph)
        if key not in seen_paragraphs:
            seen_paragraphs.add(key)
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def source_excerpt(text: str, max_chars: int = 180) -> str:
    """A contiguous, verbatim source span; an ellipsis explicitly marks clipping."""
    text = str(text or "").strip()
    if not text or max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars == 1:
        return text[:1] + "…"
    end = max_chars - 1
    boundary = max(text.rfind(" ", 0, end), text.rfind("\n", 0, end))
    if boundary > end // 2:
        end = boundary
    return text[:end].rstrip() + "…"


def actual_clause(text: str) -> str:
    """Read a real, line-leading clause/section number, never a chunk identifier."""
    match = re.search(r"(?m)^\s*(\d+(?:\.\d+)*)(?:[.)])?(?=\s+\S)", str(text or ""))
    return match.group(1) if match else ""


def _source_location(excerpt: Mapping[str, Any]) -> str:
    locations: list[str] = []
    clause = actual_clause(str(excerpt.get("text") or ""))
    if clause:
        locations.append(f"п. {clause}" if "." in clause else f"раздел {clause}")
    metadata = excerpt.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    page = metadata.get("page")
    pages = metadata.get("page_numbers")
    if page not in (None, "", 0):
        locations.append(f"страница {page}")
    elif isinstance(pages, (list, tuple)) and pages:
        locations.append("страницы " + ", ".join(str(value) for value in pages))
    return "; ".join(locations)


def format_document_answer(
    excerpts: list[dict], *, insufficient: bool = False, limitations: str = ""
) -> str:
    """Render validated evidence once, in relevance order supplied by the caller.

    Up to three distinct excerpts are accepted.  The caller must select complete
    relevant evidence units: presentation must not remove late qualifications
    just to reach a desired sentence count.
    """
    selected: list[tuple[Mapping[str, Any], str, int]] = []
    seen: set[str] = set()
    for excerpt in excerpts or []:
        if not isinstance(excerpt, Mapping):
            continue
        text = compact_evidence(str(excerpt.get("text") or ""))
        key = evidence_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        try:
            reference_id = int(excerpt.get("reference_id") or len(selected) + 1)
        except (TypeError, ValueError, OverflowError):
            reference_id = len(selected) + 1
        if reference_id < 1:
            reference_id = len(selected) + 1
        selected.append((excerpt, text, reference_id))
        if len(selected) == 3:
            break

    insufficient = insufficient or not selected
    body: list[str] = [INSUFFICIENT_EVIDENCE] if insufficient else []
    sources: list[str] = []
    for excerpt, text, reference_id in selected:
        name = re.sub(r"\s+", " ", str(excerpt.get("doc_name") or "")).strip()
        citation = f" [Источник {reference_id}]" if name else ""
        body.append(text + citation)
        if name:
            location = _source_location(excerpt)
            location = "; " + location if location else ""
            original = str(excerpt.get("text") or "").strip()
            # Cite a short original span instead of repeating the complete body.
            quote_limit = min(180, max(1, len(original) * 2 // 3))
            quote = source_excerpt(original, max_chars=quote_limit)
            sources.append(f"- [Источник {reference_id}] {name}{location}: «{quote}»")

    sections = ["### Краткий ответ\n" + "\n\n".join(body)]
    if sources:
        sections.append("### Источник\n" + "\n".join(sources))
    limitation = compact_evidence(limitations)
    # The lack-of-evidence message already appears in the answer, once.
    if evidence_key(limitation) == evidence_key(INSUFFICIENT_EVIDENCE):
        limitation = ""
    if not limitation and any(excerpt.get("truncated") for excerpt, _, _ in selected):
        limitation = "Найденный фрагмент неполон; условия за его пределами не проверены."
    if limitation:
        sections.append("### Ограничения\n" + limitation)
    return "\n\n".join(sections)
