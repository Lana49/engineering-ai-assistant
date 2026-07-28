# -*- coding: utf-8 -*-
"""
Retrieval Memory — память успешных поисковых запросов для бустинга релевантности.
Сохраняет паттерны запросов, которые привели к хорошим результатам,
и использует их для улучшения будущих поисков.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict


@dataclass
class RetrievalRecord:
    """Запись об успешном поисковом запросе."""
    pattern: str
    query_type: str
    keywords: List[str] = field(default_factory=list)
    preferred_sources: List[str] = field(default_factory=list)
    boost_terms: List[str] = field(default_factory=list)
    success_count: int = 1
    last_used: str = field(default_factory=lambda: datetime.now().isoformat())


class RetrievalMemory:
    """
    Память успешных поисковых запросов для бустинга релевантности.

    Функциональность:
    - запоминание источников, которые оказались полезными
    - запоминание терминов, которые помогли найти релевантные документы
    - применение бустов к будущим похожим запросам
    """

    def __init__(self, memory_path: Path):
        """
        Инициализация памяти.

        Args:
            memory_path: Путь к файлу для хранения данных
        """
        self.memory_path = Path(memory_path)
        self.records: List[RetrievalRecord] = []
        self.load()

    def load(self) -> None:
        """Загружает записи из файла."""
        if not self.memory_path.exists():
            self.records = []
            return

        try:
            data = json.loads(self.memory_path.read_text(encoding="utf-8"))
            self.records = [RetrievalRecord(**item) for item in data if isinstance(item, dict)]
        except (json.JSONDecodeError, KeyError, TypeError):
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
        """
        Нормализует запрос для использования в качестве ключа.

        Args:
            query: Исходный запрос

        Returns:
            Нормализованный шаблон запроса
        """
        cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in query)
        tokens = [t for t in cleaned.split() if len(t) > 2]
        return " ".join(tokens[:8])

    def save_success(
        self,
        query: str,
        query_type: str,
        keywords: List[str],
        sources: List[str],
    ) -> None:
        """
        Сохраняет успешный запрос с источниками и ключевыми словами.

        Args:
            query: Исходный запрос пользователя
            query_type: Тип запроса (semantic, lexical, hybrid)
            keywords: Список ключевых слов из запроса
            sources: Список источников, которые оказались релевантными
        """
        if not query or not sources:
            return

        pattern = self.normalize_query(query)
        if not pattern:
            return

        boost_terms = keywords[:6]
        preferred_sources = [s for s in sources if s][:8]

        for record in self.records:
            if record.pattern == pattern and record.query_type == query_type:
                record.success_count += 1
                record.last_used = datetime.now().isoformat()

                for src in preferred_sources:
                    if src not in record.preferred_sources:
                        record.preferred_sources.append(src)

                for term in boost_terms:
                    if term not in record.boost_terms:
                        record.boost_terms.append(term)

                record.preferred_sources = record.preferred_sources[:10]
                record.boost_terms = record.boost_terms[:10]
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

        # Ограничиваем размер памяти последними 300 записями
        if len(self.records) > 300:
            self.records = self.records[-300:]

        self.save()

    def get_boosts(self, query: str, query_type: str) -> Dict[str, List[str]]:
        """
        Возвращает бустинги для запроса.

        Args:
            query: Исходный запрос пользователя
            query_type: Тип запроса

        Returns:
            Словарь с ключами 'sources' и 'terms'
        """
        pattern = self.normalize_query(query)
        if not pattern:
            return {"sources": [], "terms": []}

        query_tokens = set(pattern.split())

        matched_records: List[RetrievalRecord] = []
        for record in self.records:
            if record.query_type != query_type:
                continue

            record_tokens = set(record.pattern.split())
            overlap = len(query_tokens & record_tokens)

            if overlap >= 2 or pattern == record.pattern:
                matched_records.append(record)

        # Сортируем по успешности и свежести
        matched_records.sort(
            key=lambda r: (r.success_count, r.last_used),
            reverse=True,
        )

        matched_sources: List[str] = []
        matched_terms: List[str] = []

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

    def get_stats(self) -> Dict[str, int]:
        """
        Возвращает статистику по памяти.

        Returns:
            Словарь со статистикой
        """
        return {
            "total_records": len(self.records),
            "unique_patterns": len({r.pattern for r in self.records}),
            "avg_success_count": sum(r.success_count for r in self.records) // max(1, len(self.records)),
        }

    def get_records_by_type(self, query_type: str) -> List[RetrievalRecord]:
        """
        Возвращает записи по типу запроса.

        Args:
            query_type: Тип запроса

        Returns:
            Список записей
        """
        return [r for r in self.records if r.query_type == query_type]

    def get_most_successful(self, limit: int = 10) -> List[RetrievalRecord]:
        """
        Возвращает наиболее успешные записи.

        Args:
            limit: Максимальное количество записей

        Returns:
            Список записей
        """
        return sorted(self.records, key=lambda r: r.success_count, reverse=True)[:limit]