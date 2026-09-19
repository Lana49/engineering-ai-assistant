# core/agent_loop.py
"""
Оркестратор инженерных запросов.

Принцип маршрутизации:
1. QueryParser извлекает intent, параметры, город, код документа.
2. Полный числовой расчёт исполняется немедленно, без RAG.
3. Поиск используется для нормативных вопросов, определений,
   сравнений, свободного поиска и неполных city-зависимых расчётов.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import logging
from core.query_parser import parse_query
from core.retrieval_memory import RetrievalMemory
from core.table_extractor import extract_tables, tables_to_dicts
logger = logging.getLogger(__name__)
try:
    from core.config import PROCESSED_DIR
except ImportError:
    from pathlib import Path
    PROCESSED_DIR = Path("data/processed")


class QueryType(Enum):
    CALCULATION = "calculation"
    DEFINITION = "definition"
    SEARCH = "search"
    COMPARISON = "comparison"
    REGULATORY = "regulatory"
    GENERAL = "general"


@dataclass(slots=True)
class ReasoningStep:
    step_id: int
    description: str
    result: Any = None
    confidence: float = 0.0
    next_steps: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ContextInfo:
    query: str
    query_type: QueryType
    keywords: list[str]
    entities: dict[str, Any]
    parameters: dict[str, Any]
    chunks: list[dict[str, Any]]
    confidence: float = 0.0


class AgentLoop:
    """Маршрутизирует запросы между вычислительным и retrieval-слоями."""

    REGULATORY_MARKERS = (
        "должен",
        "должна",
        "должны",
        "требуется",
        "требуют",
        "следует",
        "необходимо",
        "не допускается",
        "допускается",
        "не менее",
        "не более",
        "принимается",
        "принимают",
        "устанавливается",
        "должно быть",
        "как правило",
        "в соответствии",
    )

    REGULATORY_QUERY_EXPANSIONS = {
        "вентиляция": [
            "системы вентиляции",
            "воздухообмен",
            "расход наружного воздуха",
            "требования к вентиляции",
        ],
        "микроклимат": [
            "параметры микроклимата",
            "температура воздуха",
            "относительная влажность",
            "скорость движения воздуха",
        ],
        "отопление": [
            "система отопления",
            "теплоснабжение",
            "отопительные приборы",
            "требования к отоплению",
        ],
        "изоляция": [
            "тепловая изоляция",
            "теплоизоляционные конструкции",
            "сопротивление теплопередаче",
        ],
        "тепловая защита": [
            "сопротивление теплопередаче",
            "ограждающие конструкции",
            "энергоэффективность здания",
        ],
    }

    def __init__(self, qa_system: Any, formula_engine: Any):
        self.qa_system = qa_system
        self.formula_engine = formula_engine

        self.messages: list[dict[str, Any]] = []
        self.reasoning_steps: list[ReasoningStep] = []
        self.context: ContextInfo | None = None
        self.last_error: str | None = None

        self.memory = RetrievalMemory(
            PROCESSED_DIR / "retrieval_memory.json"
        )

    async def run(self, user_content: str) -> dict[str, Any]:
        """Главная точка входа."""
        self.messages.append({"role": "user", "content": user_content})
        self.reasoning_steps = []
        self.context = None
        self.last_error = None

        try:
            step1 = self._analyze_query(user_content)
            self.reasoning_steps.append(step1)

            if step1.result is None:
                return self._create_error_response(
                    "Не удалось проанализировать запрос.",
                    step_id=1,
                )

            query_type: QueryType = step1.result["type"]
            keywords: list[str] = step1.result["keywords"]
            entities: dict[str, Any] = step1.result["entities"]
            parameters: dict[str, Any] = step1.result["parameters"]

            # Fast-path: прямой расчёт не должен зависеть от RAG.
            if (
                query_type == QueryType.CALCULATION
                and self.formula_engine.can_calculate_directly(
                    user_content,
                    parameters,
                )
            ):
                step2 = await self._handle_direct_calculation(
                    query=user_content,
                    parameters=parameters,
                    entities=entities,
                )
                self.reasoning_steps.append(step2)

                response = self._finalize_response(
                    result=step2.result,
                    query_type=query_type,
                    keywords=keywords,
                    original_query=user_content,
                )
                response["steps"] = len(self.reasoning_steps)
                return response

            # Retrieval нужен только если direct calculation невозможен
            # или вопрос по смыслу является нормативным/поисковым.
            step2 = self._search_chunks(
                query=user_content,
                query_type=query_type,
                entities=entities,
                keywords=keywords,
            )
            self.reasoning_steps.append(step2)

            chunks = step2.result.get("chunks", []) if step2.result else []

            step3 = self._build_context(
                query=user_content,
                query_type=query_type,
                keywords=keywords,
                entities=entities,
                parameters=parameters,
                chunks=chunks,
            )
            self.reasoning_steps.append(step3)

            if step3.result is None:
                return self._create_error_response(
                    "Не удалось сформировать контекст запроса.",
                    step_id=3,
                )

            context = step3.result
            self.context = context

            step4 = await self._dispatch(context)
            self.reasoning_steps.append(step4)

            if step4.result is None:
                return self._create_error_response(
                    "Не удалось сформировать ответ.",
                    step_id=4,
                )

            response = self._finalize_response(
                result=step4.result,
                query_type=query_type,
                keywords=keywords,
                original_query=user_content,
            )
            response["steps"] = len(self.reasoning_steps)

            return response

        except Exception as exc:
            self.last_error = str(exc)
            logger.exception("Непойманная ошибка AgentLoop.run для запроса %r", user_content[:200])
            return {
                "answer": f"❌ Внутренняя ошибка: {exc}",
                "sources": [],
                "tables": [],
                "formulas": [],
                "confidence": 0.0,
                "needs_clarification": True,
                "questions": ["Повторите запрос или уточните исходные данные."],
                "query_type": "error",
                "steps": len(self.reasoning_steps) or 1,
                "error": str(exc),
            }

    @staticmethod
    def _analyze_query(query: str) -> ReasoningStep:
        """Получает intent и сущности только через QueryParser."""
        step = ReasoningStep(
            step_id=1,
            description="Анализ запроса и извлечение сущностей",
        )

        try:
            parsed = parse_query(query)

            type_map = {
                "calculation": QueryType.CALCULATION,
                "definition": QueryType.DEFINITION,
                "search": QueryType.SEARCH,
                "comparison": QueryType.COMPARISON,
                "regulatory": QueryType.REGULATORY,
                "general": QueryType.GENERAL,
            }

            entities: dict[str, Any] = {}

            if getattr(parsed, "city", None):
                entities["city"] = parsed.city

            document_codes = list(
                getattr(parsed, "document_codes", []) or []
            )
            if document_codes:
                entities["document_codes"] = document_codes
                entities["documents"] = document_codes

            section_refs = list(
                getattr(parsed, "section_refs", []) or []
            )
            if section_refs:
                entities["section_refs"] = section_refs

            keywords = list(getattr(parsed, "keywords", []) or [])
            domain_terms = [
                word for word in keywords
                if len(word) > 2
            ]
            entities["domain_terms"] = domain_terms[:12]

            step.result = {
                "type": type_map.get(
                    getattr(parsed, "intent", "search"),
                    QueryType.SEARCH,
                ),
                "keywords": keywords,
                "entities": entities,
                "parameters": dict(
                    getattr(parsed, "parameters", {}) or {}
                ),
            }
            step.confidence = 0.9

        except (
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            logger.warning("Не удалось разобрать запрос %r: %s", query[:200], exc)
            step.result = None
            step.confidence = 0.0
            step.description = f"Ошибка анализа запроса: {exc}"

        return step

    async def _handle_direct_calculation(
        self,
        query: str,
        parameters: dict[str, Any],
        entities: dict[str, Any],
    ) -> ReasoningStep:
        """Выполняет числовой расчёт без retrieval."""
        step = ReasoningStep(
            step_id=2,
            description="Прямой детерминированный расчёт без поиска",
        )

        result = await self.formula_engine.answer_calculation(
            query=query,
            parameters=parameters,
            entities=entities,
        )

        step.result = self._normalize_calculation_result(result)
        step.confidence = step.result["confidence"]

        return step

    async def _dispatch(
        self,
        context: ContextInfo,
    ) -> ReasoningStep:
        """Выбирает обработчик по intent."""
        if context.query_type == QueryType.CALCULATION:
            return await self._handle_calculation(context)

        if context.query_type == QueryType.DEFINITION:
            return self._handle_definition(context)

        if context.query_type == QueryType.COMPARISON:
            return self._handle_comparison(context)

        if context.query_type == QueryType.REGULATORY:
            return self._handle_regulatory(context)

        return self._handle_search(context)

    async def _handle_calculation(
        self,
        context: ContextInfo,
    ) -> ReasoningStep:
        """
        Неполный или city-зависимый расчёт.

        RAG уже выполнился только потому, что direct path оказался невозможен.
        FormulaEngine теперь может использовать TableCalculator и найденные
        климатические данные через qa_system.
        """
        step = ReasoningStep(
            step_id=4,
            description="Расчёт с проверкой табличных климатических данных",
        )

        try:
            result = await self.formula_engine.answer_calculation(
                query=context.query,
                parameters=context.parameters,
                entities=context.entities,
            )

            step.result = self._normalize_calculation_result(result)
            step.confidence = step.result["confidence"]

        except (
            ArithmeticError,
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            logger.exception("Ошибка grounded-ответа")
            step.result = {
                "type": "calculation",
                "answer": f"❌ Ошибка расчёта: {exc}",
                "sources": [],
                "tables": [],
                "formulas": [],
                "confidence": 0.0,
                "needs_clarification": True,
                "questions": ["Проверьте введённые параметры."],
            }
            step.confidence = 0.0

        return step

    @staticmethod
    def _normalize_calculation_result(
        result: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "type": "calculation",
            "answer": result.get("answer", "Расчёт не выполнен."),
            "sources": result.get("sources", []),
            "tables": result.get("tables", []),
            "formulas": result.get("formulas", []),
            "formula": result.get("formula"),
            "parameters": result.get("params", {}),
            "result": result.get("result"),
            "reasoning": result.get("reasoning", ""),
            "confidence": result.get("confidence", 0.0),
            "needs_clarification": result.get(
                "needs_clarification",
                False,
            ),
            "questions": result.get("questions", []),
        }

    @staticmethod
    def _unique_keep_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []

        for item in items:
            value = str(item).strip()
            key = value.lower()

            if value and key not in seen:
                seen.add(key)
                result.append(value)

        return result

    def _expand_query(
        self,
        query: str,
        query_type: QueryType,
        entities: dict[str, Any],
        keywords: list[str],
    ) -> list[str]:
        """Строит варианты запроса для retrieval."""
        variants = [query]

        document_codes = list(
            entities.get("document_codes")
            or entities.get("documents")
            or []
        )
        section_refs = list(entities.get("section_refs") or [])
        domain_terms = list(
            entities.get("domain_terms") or keywords[:8]
        )
        city = entities.get("city")

        if document_codes:
            variants.append(
                " ".join(document_codes[:2] + domain_terms[:6]).strip()
            )

        if section_refs:
            variants.append(
                " ".join(section_refs[:2] + domain_terms[:6]).strip()
            )

        if domain_terms:
            variants.append(" ".join(domain_terms[:10]).strip())

        if city:
            variants.extend([
                f"{city} климат температура отопительный период",
                f"{city} климатические параметры t от z от",
            ])

        if query_type == QueryType.REGULATORY:
            lower_query = query.lower()

            for trigger, expansions in self.REGULATORY_QUERY_EXPANSIONS.items():
                if trigger in lower_query:
                    variants.extend(expansions)

            variants.append(
                " ".join(
                    document_codes[:2]
                    + domain_terms[:8]
                    + ["требования", "следует", "не допускается"]
                ).strip()
            )

        memory_boosts = self.memory.get_boosts(
            query,
            query_type.value,
        )
        memory_terms = memory_boosts.get("terms", [])

        if memory_terms:
            variants.append(
                " ".join(domain_terms[:6] + memory_terms[:4]).strip()
            )

        return self._unique_keep_order(
            [variant for variant in variants if variant.strip()]
        )

    @staticmethod
    def _rerank_chunks(
        chunks: list[dict[str, Any]],
        query_type: QueryType,
        entities: dict[str, Any],
        memory_boosts: dict[str, list[str]],
    ) -> list[dict[str, Any]]:
        """Повторно ранжирует найденные фрагменты."""
        preferred_sources = [
            source.lower()
            for source in memory_boosts.get("sources", [])
        ]
        document_codes = [
            code.lower()
            for code in entities.get("document_codes", [])
        ]
        section_refs = [
            ref.lower()
            for ref in entities.get("section_refs", [])
        ]
        domain_terms = [
            term.lower()
            for term in entities.get("domain_terms", [])[:12]
        ]
        city = str(entities.get("city", "")).lower().strip()

        reranked: list[dict[str, Any]] = []

        for chunk in chunks:
            doc_name = str(chunk.get("doc_name", ""))
            text = str(chunk.get("text", ""))

            lower_doc_name = doc_name.lower()
            lower_text = text.lower()

            score = float(chunk.get("score", 0.0) or 0.0)

            if any(source in lower_doc_name for source in preferred_sources):
                score += 0.20

            if any(
                code in lower_doc_name or code in lower_text
                for code in document_codes
            ):
                score += 0.45

            if any(ref in lower_text for ref in section_refs):
                score += 0.30

            if city and city in lower_text:
                score += 0.25

            term_hits = sum(
                1 for term in domain_terms
                if term and term in lower_text
            )
            score += min(0.35, term_hits * 0.04)

            if query_type == QueryType.REGULATORY:
                marker_hits = sum(
                    1
                    for marker in AgentLoop.REGULATORY_MARKERS
                    if marker in lower_text
                )
                score += min(0.50, marker_hits * 0.08)

            reranked.append({
                **chunk,
                "score": score,
            })

        reranked.sort(
            key=lambda item: item.get("score", 0.0),
            reverse=True,
        )
        return reranked

    def _search_chunks(
        self,
        query: str,
        query_type: QueryType,
        entities: dict[str, Any] | None = None,
        keywords: list[str] | None = None,
    ) -> ReasoningStep:
        """Ищет, дедуплицирует и ранжирует фрагменты индекса."""
        step = ReasoningStep(
            step_id=2,
            description="Поиск и ранжирование фрагментов документов",
        )

        entities = dict(entities or {})
        keywords = list(keywords or [])

        try:
            if (
                self.qa_system is None
                or not getattr(self.qa_system, "is_ready", False)
            ):
                step.result = {
                    "chunks": [],
                    "count": 0,
                    "expanded_queries": [],
                }
                step.confidence = 0.2
                return step

            memory_boosts = self.memory.get_boosts(
                query,
                query_type.value,
            )

            expanded_queries = self._expand_query(
                query=query,
                query_type=query_type,
                entities=entities,
                keywords=keywords,
            )

            if query_type == QueryType.REGULATORY:
                top_k = 16
            elif query_type == QueryType.CALCULATION:
                top_k = 10
            else:
                top_k = 12

            collected: list[dict[str, Any]] = []
            seen: set[tuple[str, str]] = set()

            for search_query in expanded_queries[:6]:
                found = self.qa_system.search(
                    search_query,
                    top_k=top_k,
                )

                for chunk in found or []:
                    doc_name = str(chunk.get("doc_name", ""))
                    text = str(chunk.get("text", ""))
                    fingerprint = (
                        doc_name.lower().strip(),
                        re.sub(r"\s+", " ", text[:350]).lower(),
                    )

                    if not text.strip() or fingerprint in seen:
                        continue

                    seen.add(fingerprint)
                    collected.append(chunk)

            reranked = self._rerank_chunks(
                chunks=collected,
                query_type=query_type,
                entities=entities,
                memory_boosts=memory_boosts,
            )[:top_k]

            enriched: list[dict[str, Any]] = []

            for chunk in reranked:
                text = str(chunk.get("text", ""))
                doc_name = str(chunk.get("doc_name", ""))

                tables = extract_tables(text, doc_name=doc_name)
                formulas = self._extract_formulas(text)

                enriched.append({
                    **chunk,
                    "tables": tables_to_dicts(tables),
                    "formulas": formulas,
                })

            step.result = {
                "chunks": enriched,
                "count": len(enriched),
                "expanded_queries": expanded_queries,
                "memory_boosts": memory_boosts,
            }
            step.confidence = 0.85 if enriched else 0.3

        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            step.result = {
                "chunks": [],
                "count": 0,
                "expanded_queries": [],
                "error": str(exc),
            }
            step.confidence = 0.0
            step.description += f": {exc}"

        return step

    @staticmethod
    def _extract_formulas(text: str) -> list[dict[str, Any]]:
        """Извлекает короткие формулы из найденного текста."""
        raw_formulas: list[str] = []

        equation_pattern = (
            r"(?<!\w)"
            r"([A-Za-zА-Яа-яΔδλ][\w_внтотр]*\s*="
            r"\s*[^\n.;]{3,180})"
        )
        raw_formulas.extend(re.findall(equation_pattern, text))

        result: list[dict[str, Any]] = []
        seen: set[str] = set()

        for raw in raw_formulas:
            normalized = re.sub(r"\s+", " ", raw).strip()

            if normalized in seen:
                continue

            seen.add(normalized)

            variables = re.findall(
                r"[A-Za-zА-Яа-яΔδλ][\w_внтотр]*",
                normalized,
            )

            result.append({
                "raw": normalized,
                "variables": sorted(set(variables)),
            })

        return result[:5]

    @staticmethod
    def _build_context(
        query: str,
        query_type: QueryType,
        keywords: list[str],
        entities: dict[str, Any],
        parameters: dict[str, Any],
        chunks: list[dict[str, Any]],
    ) -> ReasoningStep:
        """
        Создаёт контекст, не извлекая числа повторно.

        Параметры остаются исключительно теми, которые вернул QueryParser.
        """
        step = ReasoningStep(
            step_id=3,
            description="Формирование контекста без повторного парсинга параметров",
        )

        try:
            high_score = [
                chunk for chunk in chunks
                if float(chunk.get("score", 0.0) or 0.0) >= 0.5
            ]

            confidence = (
                len(high_score) / len(chunks)
                if chunks
                else 0.2
            )

            context = ContextInfo(
                query=query,
                query_type=query_type,
                keywords=keywords,
                entities=entities,
                parameters=parameters,
                chunks=chunks,
                confidence=max(0.2, confidence),
            )

            step.result = context
            step.confidence = context.confidence

        except (AttributeError, TypeError, ValueError) as exc:
            step.result = None
            step.confidence = 0.0
            step.description += f": {exc}"

        return step

    def _handle_definition(
        self,
        context: ContextInfo,
    ) -> ReasoningStep:
        step = ReasoningStep(
            step_id=4,
            description="Поиск определения термина",
        )

        term = self._extract_definition_term(context.query)

        if not term:
            step = ReasoningStep(step_id=4, description="Уточнение термина")
            step.result = {
                "answer": "Уточните термин, для которого нужно дать определение.",
                "sources": [],
                "tables": [],
                "formulas": [],
                "confidence": 0.2,
                "needs_clarification": True,
                "questions": ["Какой термин нужно определить?"], "grounded": True,
            }
            step.confidence = 0.2
            return step
            return self._handle_grounded(context, f"Определение термина «{term}» по базе")


    @staticmethod
    def _extract_definition_term(query: str) -> str:
        value = query.strip().lower()

        prefixes = (
            "что такое ",
            "что значит ",
            "что означает ",
            "что это ",
            "дай определение ",
            "дайте определение ",
            "определение ",
            "определи ",
            "термин ",
            "понятие ",
            "расшифруй ",
            "расшифровка ",
            "аббревиатура ",
        )

        for prefix in prefixes:
            if value.startswith(prefix):
                value = value[len(prefix):]
                break

        return value.strip(" ?!.,:;\"'«»()[]")

    def _handle_regulatory(
        self,
        context: ContextInfo,
    ) -> ReasoningStep:
        """Собирает нормы по широкому набору нормативных формулировок."""
        step = ReasoningStep(
            step_id=4,
            description="Поиск нормативных требований",
        )

        candidates: list[tuple[float, dict[str, Any]]] = []

        for chunk in context.chunks:
            text = str(chunk.get("text", "")).strip()
            lower_text = text.lower()

            if not text:
                continue

            marker_hits = sum(
                1
                for marker in self.REGULATORY_MARKERS
                if marker in lower_text
            )

            score = float(chunk.get("score", 0.0) or 0.0)
            score += min(0.6, marker_hits * 0.1)

            document_codes = context.entities.get(
                "document_codes",
                [],
            )
            if any(code.lower() in lower_text for code in document_codes):
                score += 0.3

            candidates.append((score, chunk))

        candidates.sort(key=lambda item: item[0], reverse=True)

        if not candidates:
            step.result = {
                "answer": (
                    "⚠️ В текущем индексе не найдены релевантные нормативные "
                    "фрагменты. Проверьте, что нужный СП или ГОСТ загружен и "
                    "проиндексирован."
                ),
                "sources": [],
                "tables": [],
                "formulas": [],
                "confidence": 0.15,
                "needs_clarification": True,
                "questions": [
                    "Укажите номер документа, раздел или тему требования.",
                ],
            }
            step.confidence = 0.15
            return step

        selected_context = ContextInfo(
            query=context.query,
            query_type=context.query_type,
            keywords=context.keywords,
            entities=context.entities,
            parameters=context.parameters,
            chunks=[chunk for _, chunk in candidates[:4]],
            confidence=context.confidence,
        )
        return self._handle_grounded(
            selected_context,
            "Ответ по найденным нормативным фрагментам",
        )

    def _handle_search(
        self,
        context: ContextInfo,
    ) -> ReasoningStep:
        step = ReasoningStep(
            step_id=4,
            description="Поиск информации в документах",
        )

        return self._handle_grounded(context, "Ответ по найденным документам")

    def _search_answer(
        self,
        context: ContextInfo,
        intro: str | None = None,
    ) -> dict[str, Any]:
        if not context.chunks:
            return {
                "answer": (
                    "⚠️ В проиндексированных документах ничего не найдено. "
                    "Проверьте наличие документа, тему запроса или формулировку."
                ),
                "sources": [],
                "tables": [],
                "formulas": [],
                "confidence": 0.15,
                "needs_clarification": True,
                "questions": ["Уточните документ, раздел или ключевой термин."],
            }

        result = self._build_evidence_answer(
            context.chunks[:4],
            title=intro or "Найденная информация",
            max_chars=750,
        )

        return result

    @staticmethod
    def _handle_comparison(
        context: ContextInfo,
    ) -> ReasoningStep:
        step = ReasoningStep(
            step_id=4,
            description="Подготовка сравнения",
        )

        query = context.query.lower()
        parts = re.split(r"\s+(?:и|vs|против)\s+", query)
        items = [part.strip() for part in parts if len(part.strip()) > 2]

        if len(items) < 2:
            step.result = {
                "answer": (
                    "⚠️ Не удалось определить объекты сравнения. "
                    "Напишите, например: «сравни минвату и пенополистирол»."
                ),
                "sources": [],
                "tables": [],
                "formulas": [],
                "confidence": 0.2,
                "needs_clarification": True,
                "questions": ["Что именно нужно сравнить?"],
            }
            step.confidence = 0.2
            return step

        rows: list[str] = []
        sources: list[dict[str, str]] = []

        for item in items[:3]:
            match = next(
                (
                    chunk for chunk in context.chunks
                    if item in str(chunk.get("text", "")).lower()
                ),
                None,
            )

            if match is None:
                rows.append(f"- **{item}**: данных в найденных фрагментах нет.")
                continue

            text = re.sub(
                r"\s+",
                " ",
                str(match.get("text", "")).strip(),
            )[:350]
            source = str(match.get("doc_name", ""))

            rows.append(f"- **{item}**: {text}")

            if source:
                sources.append({"doc_name": source})

        step.result = {
            "answer": "### Сравнение\n\n" + "\n\n".join(rows),
            "sources": AgentLoop._unique_sources(sources),
            "tables": [],
            "formulas": [],
            "confidence": 0.75,
            "needs_clarification": False,
            "questions": [],
        }
        step.confidence = 0.75

        return step

    @staticmethod
    def _build_evidence_answer(
        chunks: list[dict[str, Any]],
        title: str,
        max_chars: int,
    ) -> dict[str, Any]:
        parts: list[str] = []
        sources: list[dict[str, str]] = []
        tables: list[dict[str, Any]] = []
        formulas: list[dict[str, Any]] = []

        seen_text: set[str] = set()

        for chunk in chunks:
            text = re.sub(
                r"\s+",
                " ",
                str(chunk.get("text", "")).strip(),
            )

            if not text:
                continue

            fingerprint = text[:250].lower()
            if fingerprint in seen_text:
                continue

            seen_text.add(fingerprint)
            parts.append(text[:max_chars])

            doc_name = str(chunk.get("doc_name", "")).strip()
            if doc_name:
                sources.append({"doc_name": doc_name})

            tables.extend(chunk.get("tables", []))
            formulas.extend(chunk.get("formulas", []))

        if not parts:
            return {
                "answer": "⚠️ Найденные фрагменты не содержат текста для ответа.",
                "sources": [],
                "tables": [],
                "formulas": [],
                "confidence": 0.15,
                "needs_clarification": True,
                "questions": ["Уточните запрос."],
            }

        return {
            "answer": f"### {title}\n\n" + "\n\n".join(parts),
            "sources": AgentLoop._unique_sources(sources),
            "tables": AgentLoop._unique_tables(tables),
            "formulas": AgentLoop._unique_formulas(formulas),
            "confidence": min(0.9, 0.5 + len(parts) * 0.1),
            "needs_clarification": False,
            "questions": [],
        }

    @staticmethod
    def _unique_sources(
        sources: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        seen: set[str] = set()
        result: list[dict[str, str]] = []

        for source in sources:
            name = str(source.get("doc_name") or "").strip()
            key = name.lower()

            if name and key not in seen:
                seen.add(key)
                result.append({"doc_name": name})

        return result

    @staticmethod
    def _unique_tables(
        tables: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        seen: set[str] = set()
        result: list[dict[str, Any]] = []

        for table in tables:
            content = str(
                table.get("raw_text")
                or table.get("content")
                or ""
            ).strip()

            fingerprint = re.sub(
                r"\s+",
                " ",
                content[:300],
            ).lower()

            if not fingerprint or fingerprint in seen:
                continue

            seen.add(fingerprint)
            result.append(table)

        return result[:10]

    @staticmethod
    def _unique_formulas(
        formulas: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        seen: set[str] = set()
        result: list[dict[str, Any]] = []

        for formula in formulas:
            raw = str(formula.get("raw", "")).strip()
            key = re.sub(r"\s+", " ", raw).lower()

            if raw and key not in seen:
                seen.add(key)
                result.append(formula)

        return result[:5]

    def _finalize_response(
        self,
        result: dict[str, Any],
        query_type: QueryType,
        keywords: list[str],
        original_query: str,
    ) -> dict[str, Any]:
        """Строит единый ответ и сохраняет удачный retrieval в память."""
        response = {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "tables": result.get("tables", []),
            "formulas": result.get("formulas", []),
            "formula": result.get("formula"),
            "params": result.get("parameters", result.get("params", {})),
            "result": result.get("result"),
            "reasoning": result.get("reasoning", ""),
            "provider": result.get("provider", "none"),
            "used_llm": bool(result.get("used_llm", False)),
            "grounded": bool(result.get("grounded", False)),
            "confidence": result.get("confidence", 0.0),
            "needs_clarification": result.get(
                "needs_clarification",
                False,
            ),
            "questions": result.get("questions", []),
            "query_type": query_type.value,
        }

        self._save_success_to_memory(
            original_query=original_query,
            query_type=query_type,
            keywords=keywords,
            response=response,
        )

        return response

    def _save_success_to_memory(
        self,
        original_query: str,
        query_type: QueryType,
        keywords: list[str],
        response: dict[str, Any],
    ) -> None:
        """
        В память отправляются только ответы с источниками и без
        необходимости уточнения; иначе память будет усиливать ошибки.
        """
        if response.get("needs_clarification"):
            return

        if response.get("confidence", 0.0) < 0.65:
            return

        source_names: list[str] = []

        for source in response.get("sources", []):
            if isinstance(source, dict):
                name = source.get("doc_name") or source.get("source")
            else:
                name = str(source)

            if name:
                source_names.append(str(name))

        source_names = self._unique_keep_order(source_names)

        if source_names:
            self.memory.save_success(
                query=original_query,
                query_type=query_type.value,
                keywords=keywords,
                sources=source_names,
            )

    @staticmethod
    def _create_error_response(
        message: str,
        step_id: int,
    ) -> dict[str, Any]:
        return {
            "answer": f"❌ {message}",
            "sources": [],
            "tables": [],
            "formulas": [],
            "confidence": 0.0,
            "needs_clarification": True,
            "questions": ["Уточните запрос."],
            "query_type": "error",
            "steps": step_id,
        }
if  __name__ == "__main__":
    import asyncio
    from pathlib import Path
    from tempfile import TemporaryDirectory

    class _TestQA:
        is_ready = True

        def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
            return [{
                "doc_name": "Тестовый СП",
                "chunk_id": 3,
                "text": "Вентиляция должна обеспечивать требуемый воздухообмен.",
                "score": 0.9,
                "metadata": {"page": 7},
            }]

        def answer_from_results(self, question: str, results: list[dict[str, Any]]) -> dict[str, Any]:
            assert results and results[0]["doc_name"] == "Тестовый СП"
            return {
                "answer": "### Краткий ответ\n\nТребование найдено. [Источник 1]",
                "sources": [{"reference_id": 1, "doc_name": "Тестовый СП", "page": 7}],
                "tables": [], "formulas": [], "confidence": 0.9,
                "needs_clarification": False, "questions": [], "grounded": True,
            }

    class _TestFormula:
        def can_calculate_directly(self, query: str, parameters: dict[str, Any]) -> bool:
            return False

    with TemporaryDirectory() as directory:
        loop = AgentLoop(_TestQA(), _TestFormula())
        loop.memory = RetrievalMemory(Path(directory) / "memory.json")
        response = asyncio.run(loop.run("Какие требования к вентиляции?"))
        assert response["sources"][0]["doc_name"] == "Тестовый СП"
        assert response["query_type"] == "regulatory"