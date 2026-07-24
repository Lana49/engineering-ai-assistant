# core/agent_loop.py
"""
Многошаговый агент для инженерных запросов.
Совместим с async FormulaEngine.answer_calculation(...).
"""

import re
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from core.prompts import get_quick_definition


class QueryType(Enum):
    """Типы запросов."""
    CALCULATION = "calculation"
    DEFINITION = "definition"
    SEARCH = "search"
    COMPARISON = "comparison"
    REGULATORY = "regulatory"
    GENERAL = "general"


@dataclass
class ReasoningStep:
    """Шаг цепочки рассуждений."""
    step_id: int
    description: str
    result: Any = None
    confidence: float = 0.0
    next_steps: List[str] = field(default_factory=list)


@dataclass
class ContextInfo:
    """Контекстная информация по запросу."""
    query: str
    query_type: QueryType
    keywords: List[str]
    entities: Dict[str, Any]
    parameters: Dict[str, float]
    chunks: List[Dict[str, Any]]
    confidence: float = 0.0


class AgentLoop:
    """
    Многошаговый агент для инженерных запросов.
    """

    def __init__(self, qa_system, formula_engine):
        self.qa_system = qa_system
        self.formula_engine = formula_engine
        self.messages: List[Dict[str, Any]] = []
        self.reasoning_steps: List[ReasoningStep] = []
        self.context: Optional[ContextInfo] = None
        self.max_steps = 6
        self.last_error: Optional[str] = None

    async def run(self, user_content: str) -> Dict[str, Any]:
        """
        Запускает обработку запроса.
        """
        self.messages.append({"role": "user", "content": user_content})
        self.reasoning_steps = []
        self.context = None

        try:
            step1 = self._analyze_query(user_content)
            self.reasoning_steps.append(step1)
            if not step1.result:
                return self._create_error_response("Не удалось проанализировать запрос", 1)

            query_type = step1.result.get("type", QueryType.GENERAL)
            keywords = step1.result.get("keywords", [])
            entities = step1.result.get("entities", {})
            parameters = step1.result.get("parameters", {})

            step2 = self._search_chunks(user_content, query_type)
            self.reasoning_steps.append(step2)

            chunks = []
            if step2.result:
                chunks = step2.result.get("chunks", [])

            step3 = self._analyze_context(
                user_content,
                chunks,
                query_type,
                entities,
                parameters
            )
            self.reasoning_steps.append(step3)
            if not step3.result:
                return self._create_error_response("Не удалось проанализировать контекст", 3)

            context_info = step3.result
            self.context = context_info

            if query_type == QueryType.CALCULATION:
                step4 = await self._handle_calculation(context_info)
            elif query_type == QueryType.DEFINITION:
                step4 = self._handle_definition(context_info)
            elif query_type == QueryType.COMPARISON:
                step4 = self._handle_comparison(context_info)
            elif query_type == QueryType.REGULATORY:
                step4 = self._handle_regulatory(context_info)
            else:
                step4 = self._handle_search(context_info)

            self.reasoning_steps.append(step4)
            if not step4.result:
                return self._create_error_response("Не удалось обработать запрос", 4)

            step5 = self._check_completeness(step4.result, context_info)
            self.reasoning_steps.append(step5)

            step6 = self._generate_final_response(step4.result, step5.result, context_info)
            self.reasoning_steps.append(step6)

            final_response = self._format_response(step6.result, context_info, query_type)

            if step5.result and step5.result.get("needs_clarification", False):
                final_response["needs_clarification"] = True
                final_response["questions"] = step5.result.get("questions", [])

            self.messages.append({"role": "assistant", "content": final_response["answer"]})
            final_response["steps"] = len(self.reasoning_steps)
            return final_response

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            self.last_error = str(e)
            return {
                "answer": f"❌ Ошибка: {e}",
                "sources": [],
                "tables": [],
                "formulas": [],
                "confidence": 0.0,
                "query_type": "error",
                "steps": len(self.reasoning_steps) or 1,
                "error": str(e)
            }

    @staticmethod
    def _analyze_query(query: str) -> ReasoningStep:
        """Анализирует запрос и определяет его тип."""
        step = ReasoningStep(
            step_id=1,
            description="Анализ запроса и определение типа"
        )

        try:
            query_lower = query.lower()

            calc_triggers = [
                "рассчитай", "вычисли", "посчитай", "толщин", "температур",
                "потери", "формул", "вентиляц", "расход", "гсоп", "градусо",
                "определите", "найдите", "рассчитать", "вычислить"
            ]
            def_triggers = [
                "что такое", "определение", "термин", "понятие", "что значит",
                "что означает", "расшифруй", "аббревиатура", "расшифровка",
                "что это", "как понимать", "объясните", "поясните"
            ]
            comp_triggers = ["сравни", "сравнение", "отличие", "разница", "чем отличается"]
            reg_triggers = ["норма", "снип", "сп", "гост", "требование", "стандарт", "норматив"]

            is_calc = any(w in query_lower for w in calc_triggers)
            is_def = any(w in query_lower for w in def_triggers)
            is_comp = any(w in query_lower for w in comp_triggers)
            is_reg = any(w in query_lower for w in reg_triggers)

            if is_calc:
                query_type = QueryType.CALCULATION
            elif is_def:
                query_type = QueryType.DEFINITION
            elif is_comp:
                query_type = QueryType.COMPARISON
            elif is_reg:
                query_type = QueryType.REGULATORY
            else:
                query_type = QueryType.SEARCH

            keywords = re.findall(r"[а-яА-Яa-zA-Z0-9_]{2,}", query)
            keywords = [w.lower() for w in keywords if len(w) > 2]

            numbers = re.findall(r"-?\d+[.,]?\d*", query)
            parsed_numbers = []
            for num in numbers:
                try:
                    parsed_numbers.append(float(num.replace(",", ".")))
                except ValueError:
                    continue

            parameters: Dict[str, Any] = {}
            if parsed_numbers:
                parameters["numeric_values"] = parsed_numbers

            entities: Dict[str, Any] = {}
            if "°" in query or "град" in query_lower:
                entities["unit_type"] = "temperature"
            if "м²" in query or "кв.м" in query_lower:
                entities["unit_type"] = "area"
            if "м³" in query or "куб" in query_lower:
                entities["unit_type"] = "volume"

            doc_pattern = r"(СП\s?\d+\.?\d*\.?\d*|ГОСТ\s?\d+[-–]\d+|СНиП\s?[0-9.\-]+)"
            docs = re.findall(doc_pattern, query, flags=re.IGNORECASE)
            if docs:
                entities["documents"] = docs

            step.result = {
                "type": query_type,
                "keywords": keywords,
                "entities": entities,
                "parameters": parameters,
                "raw_query": query
            }
            step.confidence = 0.9

        except (re.error, ValueError, TypeError) as e:
            step.result = None
            step.confidence = 0.0
            step.description += f" (Ошибка: {e})"

        return step

    def _search_chunks(
        self,
        query: str,
        query_type: QueryType,
    ) -> ReasoningStep:
        """Ищет релевантные фрагменты в базе знаний."""
        step = ReasoningStep(
            step_id=2,
            description="Поиск релевантных фрагментов документов"
        )

        try:
            if not self.qa_system or not getattr(self.qa_system, "is_ready", False):
                step.result = {"chunks": [], "count": 0, "query_type": query_type}
                step.confidence = 0.2
                return step

            top_k = 8 if query_type == QueryType.CALCULATION else 5
            chunks = self.qa_system.search(query, top_k)

            enriched_chunks = []
            for chunk in chunks:
                text = chunk.get("text", "")
                tables = self._extract_tables(text, chunk.get("doc_name", ""))
                formulas = self._extract_formulas(text)

                enriched_chunks.append({
                    **chunk,
                    "tables": tables,
                    "formulas": formulas
                })

            step.result = {
                "chunks": enriched_chunks,
                "count": len(enriched_chunks),
                "query_type": query_type
            }
            step.confidence = 0.8 if enriched_chunks else 0.3

        except (AttributeError, TypeError, ValueError) as e:
            step.result = None
            step.confidence = 0.0
            step.description += f" (Ошибка: {e})"

        return step

    @staticmethod
    def _extract_tables(text: str, doc_name: str) -> List[Dict[str, Any]]:
        """Извлекает таблицы из текста."""
        tables = []
        lines = text.split("\n")
        in_table = False
        table_lines = []
        table_title = ""

        for i, line in enumerate(lines):
            if "[ТАБЛИЦА" in line:
                in_table = True
                table_lines = []
                if i > 0 and len(lines[i - 1].strip()) < 100:
                    table_title = lines[i - 1].strip()
                else:
                    table_title = f"Таблица {len(tables) + 1}"
                continue

            if in_table:
                if line.strip() == "":
                    if table_lines:
                        tables.append({
                            "title": table_title,
                            "content": "\n".join(table_lines),
                            "rows": [l for l in table_lines if l.strip()],
                            "doc_name": doc_name
                        })
                        table_lines = []
                        table_title = ""
                    in_table = False
                else:
                    table_lines.append(line.strip())

        if table_lines:
            tables.append({
                "title": table_title,
                "content": "\n".join(table_lines),
                "rows": [l for l in table_lines if l.strip()],
                "doc_name": doc_name
            })

        return tables

    @staticmethod
    def _extract_formulas(text: str) -> List[Dict[str, Any]]:
        """Извлекает формулы из текста."""
        formulas = []
        pattern = r"([A-Za-zА-Яа-я][_\w]*\s*[=]\s*[^;.\n]+)"

        matches = re.findall(pattern, text)
        for match in matches:
            if len(match) > 3:
                variables = re.findall(r"[A-Za-zА-Яа-я][_\w]*", match)
                formulas.append({
                    "raw": match.strip(),
                    "variables": list(set(variables)),
                    "has_operator": any(c in match for c in ["+", "-", "*", "/", "^", "="])
                })

        bracket_pattern = r"\(([^)]*[=+\-*/^][^)]*)\)"
        bracket_matches = re.findall(bracket_pattern, text)
        for match in bracket_matches:
            raw = f"({match})"
            if len(match) > 3 and raw not in [f["raw"] for f in formulas]:
                formulas.append({
                    "raw": raw,
                    "variables": list(set(re.findall(r"[A-Za-zА-Яа-я][_\w]*", match))),
                    "has_operator": True
                })

        unique_formulas = []
        seen = set()
        for formula in formulas:
            key = formula["raw"]
            if key not in seen:
                seen.add(key)
                unique_formulas.append(formula)

        return unique_formulas

    @staticmethod
    def _analyze_context(
        query: str,
        chunks: List[Dict[str, Any]],
        query_type: QueryType,
        entities: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> ReasoningStep:
        """Анализирует контекст и извлекает ключевую информацию."""
        step = ReasoningStep(
            step_id=3,
            description="Анализ контекста и извлечение ключевой информации"
        )

        try:
            combined_text = "\n".join([c.get("text", "") for c in chunks[:3]])

            extracted_params: Dict[str, float] = {}
            param_patterns = {
                "temperature": r"(-?\d+[.,]?\d*)\s*°[СC]",
                "thickness": r"(\d+[.,]?\d*)\s*мм",
                "density": r"(\d+[.,]?\d*)\s*кг/м³",
                "flow": r"(\d+[.,]?\d*)\s*м³/ч",
                "power": r"(\d+[.,]?\d*)\s*кВт",
                "resistance": r"(\d+[.,]?\d*)\s*м²·°С/Вт"
            }

            for param, pattern in param_patterns.items():
                matches = re.findall(pattern, combined_text)
                if matches:
                    try:
                        extracted_params[param] = float(matches[0].replace(",", "."))
                    except ValueError:
                        pass

            # Обновляем параметры из запроса
            numeric_params = parameters.get("numeric_values", [])
            if numeric_params and isinstance(numeric_params, list):
                for i, val in enumerate(numeric_params[:3]):
                    if i == 0 and "temperature" not in extracted_params:
                        extracted_params["temperature"] = float(val)
                    elif i == 1 and "flow" not in extracted_params:
                        extracted_params["flow"] = float(val)

            relevance_score = 0.0
            if chunks:
                good_scores = [c for c in chunks if c.get("score", 0) > 0.5]
                relevance_score = len(good_scores) / len(chunks) if chunks else 0.0

            context_info = ContextInfo(
                query=query,
                query_type=query_type,
                keywords=[w for w in query.split() if len(w) > 2],
                entities=entities,
                parameters=extracted_params,
                chunks=chunks,
                confidence=relevance_score if relevance_score > 0 else 0.5
            )

            step.result = context_info
            step.confidence = context_info.confidence

        except (ValueError, TypeError, AttributeError) as e:
            step.result = None
            step.confidence = 0.0
            step.description += f" (Ошибка: {e})"

        return step

    async def _handle_calculation(self, context: ContextInfo) -> ReasoningStep:
        """Обрабатывает расчётный запрос."""
        step = ReasoningStep(
            step_id=4,
            description="Выполнение инженерного расчета"
        )

        try:
            query = context.query
            params = context.parameters or {}

            # Формируем запрос с параметрами
            if params:
                param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                calc_query = f"{query}. Данные: {param_str}"
            else:
                calc_query = query

            raw_result = self.formula_engine.answer_calculation(calc_query)

            if asyncio.iscoroutine(raw_result):
                result = await raw_result
            else:
                result = raw_result

            # Fallback: если расчёт не дал ответа, ищем формулу в чанках
            if context.chunks and not result.get("answer"):
                for chunk in context.chunks:
                    if chunk.get("formulas"):
                        fallback_formula = chunk["formulas"][0]
                        result = {
                            "answer": f"Найдена формула: {fallback_formula.get('raw', '')}",
                            "formula": fallback_formula,
                            "params": params,
                            "result": None,
                            "reasoning": "Формула взята напрямую из найденного фрагмента документа.",
                            "source": chunk.get("doc_name", "Документ"),
                            "sources": [{"doc_name": chunk.get("doc_name", "Документ")}],
                            "tables": [],
                            "formulas": [fallback_formula]
                        }
                        break

            calc_confidence = 0.9 if result.get("answer") else 0.3

            step.result = {
                "type": "calculation",
                "answer": result.get("answer", "Расчет не удался"),
                "formula": result.get("formula"),
                "parameters": result.get("params", params),
                "result": result.get("result"),
                "reasoning": result.get("reasoning", ""),
                "source": result.get("source", ""),
                "sources": result.get("sources", []),
                "tables": result.get("tables", []),
                "formulas": result.get("formulas", []),
                "confidence": result.get("confidence", calc_confidence)
            }
            step.confidence = step.result["confidence"]

        except (ValueError, TypeError, ZeroDivisionError) as e:
            step.result = {
                "type": "calculation",
                "answer": f"❌ Ошибка расчета: {e}",
                "formula": None,
                "parameters": {},
                "result": None,
                "reasoning": f"Ошибка на этапе расчета: {e}",
                "source": "",
                "sources": [],
                "tables": [],
                "formulas": [],
                "confidence": 0.0
            }
            step.confidence = 0.0
            step.description += f" (Ошибка: {e})"

        return step

    def _handle_definition(self, context: ContextInfo) -> ReasoningStep:
        """Обрабатывает запрос на определение термина."""
        step = ReasoningStep(
            step_id=4,
            description="Поиск определения термина"
        )

        try:
            query = context.query.strip()
            query_lower = query.lower()

            # Извлекаем термин
            term = query_lower
            for prefix in [
                "что такое", "определение", "термин", "понятие",
                "что значит", "что означает", "расшифруй",
                "аббревиатура", "расшифровка", "что это",
                "как понимать", "объясните", "поясните"
            ]:
                term = term.replace(prefix, "").strip()

            # Сначала ищем в быстром словаре
            quick = get_quick_definition(term)
            if quick:
                definition = quick.get("definition", "")
                source = quick.get("source", "")
                example = quick.get("example", "")

                full_answer = definition
                if example:
                    full_answer += f"\n\nПример: {example}"
                if source:
                    full_answer += f"\n\nИсточник: {source}"

                step.result = {
                    "type": "definition",
                    "answer": full_answer,
                    "sources": [source] if source else [],
                    "tables": [],
                    "formulas": [],
                    "confidence": 0.95
                }
                step.confidence = 0.95
                return step

            # Ищем в QA системе
            definition = self.qa_system.find_definition(term)

            if not definition.get("found") and context.chunks:
                for chunk in context.chunks:
                    chunk_text = chunk.get("text", "")
                    if term in chunk_text.lower():
                        sentences = re.split(r"(?<=[.!?])\s+", chunk_text)
                        for sent in sentences:
                            if term in sent.lower():
                                definition = {
                                    "found": True,
                                    "definition": sent.strip(),
                                    "source": chunk.get("doc_name", "Документ")
                                }
                                break
                        if definition.get("found"):
                            break

            if definition.get("found"):
                step.result = {
                    "type": "definition",
                    "term": term,
                    "definition": definition.get("definition", ""),
                    "source": definition.get("source", ""),
                    "found": True,
                    "answer": (
                        f"📖 **Определение термина «{term}»:**\n\n"
                        f"{definition.get('definition', '')}\n\n"
                        f"📚 **Источник:** {definition.get('source', 'Нормативная база')}"
                    ),
                    "sources": [{"doc_name": definition.get("source", "Нормативная база")}],
                    "tables": [],
                    "formulas": [],
                    "confidence": 0.9
                }
                step.confidence = 0.9
            else:
                step.result = {
                    "type": "definition",
                    "answer": f"❌ Определение для термина «{term}» не найдено.",
                    "sources": [],
                    "tables": [],
                    "formulas": [],
                    "confidence": 0.2
                }
                step.confidence = 0.2

        except (ValueError, TypeError, AttributeError) as e:
            step.result = None
            step.confidence = 0.0
            step.description += f" (Ошибка: {e})"

        return step

    @staticmethod
    def _handle_search(context: ContextInfo) -> ReasoningStep:
        """Обрабатывает поисковый запрос."""
        step = ReasoningStep(
            step_id=4,
            description="Формирование ответа по найденным данным"
        )

        try:
            if not context.chunks:
                step.result = {
                    "type": "search",
                    "answer": "❌ Информация не найдена в базе документов.",
                    "sources": [],
                    "tables": [],
                    "formulas": [],
                    "confidence": 0.1
                }
                step.confidence = 0.1
                return step

            useful_chunks = []
            for chunk in context.chunks:
                text = chunk.get("text", "")
                if text and len(text.strip()) > 30:
                    useful_chunks.append(chunk)

            if not useful_chunks:
                useful_chunks = context.chunks[:2]

            answer_parts = []
            sources = []
            tables = []
            formulas = []

            first = useful_chunks[0]
            first_text = first.get("text", "").strip()
            if first_text:
                answer_parts.append(first_text[:1200])

            for chunk in useful_chunks[:3]:
                doc_name = chunk.get("doc_name", "")
                if doc_name and doc_name not in sources:
                    sources.append(doc_name)

                for table in chunk.get("tables", []):
                    if table not in tables:
                        tables.append(table)

                for formula in chunk.get("formulas", []):
                    if formula not in formulas:
                        formulas.append(formula)

            answer = "\n\n".join(part for part in answer_parts if part).strip()
            if not answer:
                answer = "Найдены релевантные фрагменты, но текст ответа пуст."

            if sources:
                answer += "\n\nИсточники:\n" + "\n".join(f"• {src}" for src in sources[:3])

            step.result = {
                "type": "search",
                "answer": answer,
                "sources": sources,
                "tables": tables[:5],
                "formulas": formulas[:5],
                "confidence": min(0.9, max(0.4, context.confidence))
            }
            step.confidence = step.result["confidence"]

        except (ValueError, TypeError, AttributeError) as e:
            step.result = None
            step.confidence = 0.0
            step.description += f" (Ошибка: {e})"

        return step

    @staticmethod
    def _handle_comparison(context: ContextInfo) -> ReasoningStep:
        """Обрабатывает запрос на сравнение."""
        step = ReasoningStep(
            step_id=4,
            description="Сравнение найденных данных"
        )

        try:
            # Используем context.query для получения исходного запроса
            query = context.query.lower()
            parts = re.split(r"\s+и\s+|\s+vs\s+|\s+против\s+", query)

            if len(parts) > 1:
                items = [p.strip() for p in parts if len(p.strip()) > 2]
                comparison_results = []

                for item in items:
                    # Ищем информацию по каждому объекту в чанках
                    info = None
                    for chunk in context.chunks:
                        chunk_text = chunk.get("text", "").lower()
                        if item in chunk_text:
                            info = chunk.get("text", "")[:200]
                            source = chunk.get("doc_name", "Документ")
                            break

                    if info:
                        comparison_results.append({
                            "item": item,
                            "info": info,
                            "source": source or "Документ"
                        })

                answer_lines = [f"**Сравнение:** {', '.join(items)}", ""]
                for row in comparison_results:
                    answer_lines.append(f"**{row['item']}** — {row['info']}")
                    answer_lines.append(f"Источник: {row['source']}")
                    answer_lines.append("")

                sources = [{"doc_name": r["source"]} for r in comparison_results if r.get("source")]

                step.result = {
                    "type": "comparison",
                    "items": items,
                    "results": comparison_results,
                    "answer": "\n".join(answer_lines).strip(),
                    "sources": sources,
                    "tables": [],
                    "formulas": [],
                    "confidence": 0.8 if comparison_results else 0.3
                }
                step.confidence = 0.8 if comparison_results else 0.3
            else:
                step.result = {
                    "type": "comparison",
                    "items": [],
                    "results": [],
                    "answer": "Не удалось определить объекты для сравнения. Используйте формат: «Сравни X и Y».",
                    "sources": [],
                    "tables": [],
                    "formulas": [],
                    "confidence": 0.0
                }
                step.confidence = 0.0

        except (ValueError, TypeError, AttributeError) as e:
            step.result = None
            step.confidence = 0.0
            step.description += f" (Ошибка: {e})"

        return step

    @staticmethod
    def _handle_regulatory(context: ContextInfo) -> ReasoningStep:
        """Обрабатывает запрос на поиск нормативных требований."""
        step = ReasoningStep(
            step_id=4,
            description="Поиск нормативных требований"
        )

        try:
            requirements = []
            for chunk in context.chunks:
                text_lower = chunk.get("text", "").lower()

                markers = [
                    "должен", "обязан", "требуется", "не менее", "не более",
                    "допустим", "предел", "норма", "стандарт"
                ]

                for marker in markers:
                    if marker in text_lower:
                        sentences = re.split(r"(?<=[.!?])\s+", chunk.get("text", ""))
                        for sent in sentences:
                            if marker in sent.lower():
                                requirements.append({
                                    "text": sent.strip(),
                                    "marker": marker,
                                    "source": chunk.get("doc_name", "Документ")
                                })
                                break

            if not requirements and context.chunks:
                for chunk in context.chunks[:2]:
                    requirements.append({
                        "text": chunk.get("text", "")[:300],
                        "marker": "информация",
                        "source": chunk.get("doc_name", "Документ")
                    })

            answer_lines = []
            if requirements:
                answer_lines.append("📌 **Найдены нормативные требования:**")
                answer_lines.append("")
                for req in requirements[:5]:
                    answer_lines.append(f"- {req['text']}")
                    answer_lines.append(f"  Источник: {req['source']}")
            else:
                answer_lines.append("❌ Требования не найдены")

            step.result = {
                "type": "regulatory",
                "requirements": requirements[:5],
                "answer": "\n".join(answer_lines),
                "sources": [{"doc_name": r["source"]} for r in requirements[:5]],
                "tables": [],
                "formulas": [],
                "confidence": 0.8 if requirements else 0.2
            }
            step.confidence = 0.8 if requirements else 0.2

        except (ValueError, TypeError, AttributeError) as e:
            step.result = None
            step.confidence = 0.0
            step.description += f" (Ошибка: {e})"

        return step

    @staticmethod
    def _check_completeness(result: Dict[str, Any], context: ContextInfo) -> ReasoningStep:
        """Проверяет полноту ответа."""
        step = ReasoningStep(
            step_id=5,
            description="Проверка полноты ответа"
        )

        try:
            answer = result.get("answer", "") or ""
            needs_clarification = False
            questions: List[str] = []

            if context.query_type == QueryType.CALCULATION:
                if not result.get("formula") and not result.get("formulas"):
                    needs_clarification = True
                    questions.append("Уточните формулу или тип расчёта.")
                if not context.parameters:
                    questions.append("Укажите исходные числовые данные для расчёта.")

            if len(answer.strip()) < 20:
                needs_clarification = True
                if "Уточните" not in " ".join(questions):
                    questions.append("Уточните запрос, чтобы найти более точный ответ.")

            step.result = {
                "is_complete": not needs_clarification,
                "needs_clarification": needs_clarification,
                "questions": questions,
                "completeness_score": max(0.0, 1.0 - len(questions) * 0.2) if questions else 1.0
            }
            step.confidence = 0.85 if not needs_clarification else 0.5

        except (ValueError, TypeError, AttributeError) as e:
            step.result = {
                "is_complete": False,
                "needs_clarification": True,
                "questions": [f"Ошибка проверки полноты: {e}"],
                "completeness_score": 0.0
            }
            step.confidence = 0.0
            step.description += f" (Ошибка: {e})"

        return step

    @staticmethod
    def _generate_final_response(
        result: Dict[str, Any],
        completeness: Dict[str, Any],
        context: ContextInfo
    ) -> ReasoningStep:
        """Формирует финальный ответ."""
        step = ReasoningStep(
            step_id=6,
            description="Формирование финального ответа"
        )

        try:
            answer = result.get("answer", "").strip()
            sources = result.get("sources", [])
            tables = result.get("tables", [])
            formulas = result.get("formulas", [])
            confidence = result.get("confidence", context.confidence)

            if not answer:
                answer = "❌ Не удалось сформировать ответ."

            step.result = {
                "answer": answer,
                "sources": sources,
                "tables": tables,
                "formulas": formulas,
                "confidence": confidence,
                "needs_clarification": completeness.get("needs_clarification", False),
                "questions": completeness.get("questions", [])
            }
            step.confidence = confidence if isinstance(confidence, (int, float)) else 0.7

        except (ValueError, TypeError, AttributeError) as e:
            step.result = {
                "answer": f"❌ Ошибка финализации ответа: {e}",
                "sources": [],
                "tables": [],
                "formulas": [],
                "confidence": 0.0,
                "needs_clarification": False,
                "questions": []
            }
            step.confidence = 0.0
            step.description += f" (Ошибка: {e})"

        return step

    def _format_response(
        self,
        result: Dict[str, Any],
        context: ContextInfo,
        query_type: QueryType
    ) -> Dict[str, Any]:
        """Форматирует ответ."""
        response = {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "tables": result.get("tables", []),
            "formulas": result.get("formulas", []),
            "confidence": result.get("confidence", 0.0),
            "query_type": query_type.value,
            "reasoning_steps": [
                {
                    "step": step.step_id,
                    "description": step.description,
                    "confidence": step.confidence
                }
                for step in self.reasoning_steps
            ]
        }

        if result.get("needs_clarification"):
            response["needs_clarification"] = True
            response["questions"] = result.get("questions", [])

        return response

    def _create_error_response(self, message: str, steps: int = 1) -> Dict[str, Any]:
        """Создаёт ответ с ошибкой."""
        return {
            "answer": f"❌ {message}",
            "sources": [],
            "tables": [],
            "formulas": [],
            "confidence": 0.0,
            "query_type": "error",
            "steps": steps,
            "reasoning_steps": [
                {
                    "step": step.step_id,
                    "description": step.description,
                    "confidence": step.confidence
                }
                for step in self.reasoning_steps
            ]
        }

    def get_reasoning_chain(self) -> str:
        """Возвращает цепочку рассуждений в текстовом виде."""
        chain = "🔍 **Цепочка рассуждений:**\n\n"
        for step in self.reasoning_steps:
            confidence_stars = "⭐" * int(step.confidence * 5)
            chain += f"**Шаг {step.step_id}:** {step.description}\n"
            chain += f"   Уверенность: {step.confidence:.0%} {confidence_stars}\n\n"
        return chain

    def get_reasoning_json(self) -> Dict[str, Any]:
        """Возвращает цепочку рассуждений в JSON."""
        return {
            "steps": [
                {
                    "step_id": step.step_id,
                    "description": step.description,
                    "confidence": step.confidence,
                    "result": str(step.result)[:200] if step.result else None
                }
                for step in self.reasoning_steps
            ],
            "total_steps": len(self.reasoning_steps)
        }