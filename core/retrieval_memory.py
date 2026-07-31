# core/retrieval_memory.py
"""
Retrieval Memory — память успешных поисковых запросов для бустинга релевантности.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(slots=True)
class RetrievalRecord:
    """Запись об успешном поисковом запросе."""
    pattern: str
    query_type: str
    keywords: list[str] = field(default_factory=list)
    preferred_sources: list[str] = field(default_factory=list)
    boost_terms: list[str] = field(default_factory=list)
    success_count: int = 1
    last_used: str = field(default_factory=lambda: datetime.now().isoformat())


class RetrievalMemory:
    """
    Память успешных поисковых запросов для бустинга релевантности.
    """

    def __init__(self, memory_path: Path | str):
        self.memory_path = Path(memory_path)
        self.records: list[RetrievalRecord] = []
        self.load()

    def load(self) -> None:
        """Загружает записи из файла."""
        if not self.memory_path.exists():
            self.records = []
            return

        try:
            data = json.loads(self.memory_path.read_text(encoding="utf-8"))
            self.records = [RetrievalRecord(**item) for item in data if isinstance(item, dict)]
        except (json.JSONDecodeError, KeyError, TypeError, OSError):
            self.records = []

    def save(self) -> None:
        """Сохраняет записи в файл."""
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(record) for record in self.records]
        self.memory_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def normalize_query(query: str) -> str:
        """Нормализует запрос для использования как ключа."""
        cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in query)
        tokens = [token for token in cleaned.split() if len(token) > 2]
        return " ".join(tokens[:8])

    def save_success(
        self,
        query: str,
        query_type: str,
        keywords: list[str],
        sources: list[str],
    ) -> None:
        """Сохраняет успешный запрос с источниками и ключевыми словами."""
        if not query or not sources:
            return

        pattern = self.normalize_query(query)
        if not pattern:
            return

        boost_terms = [term for term in keywords if term][:6]
        preferred_sources = [src for src in sources if src][:8]

        for existing_record in self.records:
            if existing_record.pattern == pattern and existing_record.query_type == query_type:
                existing_record.success_count += 1
                existing_record.last_used = datetime.now().isoformat()

                for src in preferred_sources:
                    if src not in existing_record.preferred_sources:
                        existing_record.preferred_sources.append(src)

                for term in boost_terms:
                    if term not in existing_record.boost_terms:
                        existing_record.boost_terms.append(term)

                for term in keywords:
                    if term and term not in existing_record.keywords:
                        existing_record.keywords.append(term)

                existing_record.preferred_sources = existing_record.preferred_sources[:10]
                existing_record.boost_terms = existing_record.boost_terms[:10]
                existing_record.keywords = existing_record.keywords[:15]
                self.save()
                return

        self.records.append(
            RetrievalRecord(
                pattern=pattern,
                query_type=query_type,
                keywords=keywords[:10],
                preferred_sources=preferred_sources,
                boost_terms=boost_terms,
            )
        )

        if len(self.records) > 300:
            self.records = self.records[-300:]

        self.save()

    def get_boosts(self, query: str, query_type: str) -> dict[str, list[str]]:
        """Возвращает бустинги для похожего запроса."""
        pattern = self.normalize_query(query)
        if not pattern:
            return {"sources": [], "terms": []}

        query_tokens = set(pattern.split())
        matched_records: list[RetrievalRecord] = []

        for record in self.records:
            if record.query_type != query_type:
                continue

            record_tokens = set(record.pattern.split())
            overlap = len(query_tokens & record_tokens)

            if overlap >= 2 or pattern == record.pattern:
                matched_records.append(record)

        matched_records.sort(
            key=lambda r: (r.success_count, r.last_used),
            reverse=True,
        )

        matched_sources: list[str] = []
        matched_terms: list[str] = []

        for record in matched_records:
            for src in record.preferred_sources:
                if src not in matched_sources:
                    matched_sources.append(src)
            for term in record.boost_terms:
                if term not in matched_terms:
                    matched_terms.append(term)

        return {
            "sources": matched_sources[:8],
            "terms": matched_terms[:8],
        }

    def clear(self) -> None:
        """Очищает всю память."""
        self.records = []
        self.save()

    def get_stats(self) -> dict[str, int | float]:
        """Возвращает статистику по памяти."""
        total = len(self.records)
        if total == 0:
            return {
                "total_records": 0,
                "unique_patterns": 0,
                "avg_success_count": 0.0,
            }

        return {
            "total_records": total,
            "unique_patterns": len({r.pattern for r in self.records}),
            "avg_success_count": sum(r.success_count for r in self.records) / total,
        }

    def get_records_by_type(self, query_type: str) -> list[RetrievalRecord]:
        """Возвращает записи по типу запроса."""
        return [r for r in self.records if r.query_type == query_type]

    def get_most_successful(self, limit: int = 10) -> list[RetrievalRecord]:
        """Возвращает наиболее успешные записи."""
        return sorted(self.records, key=lambda r: r.success_count, reverse=True)[:limit]