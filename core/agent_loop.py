import re
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class QueryType(Enum):
    CALCULATION = "calculation"
    DEFINITION = "definition"
    SEARCH = "search"
    COMPARISON = "comparison"
    REGULATORY = "regulatory"
    GENERAL = "general"


@dataclass
class ReasoningStep:
    step_id: int
    description: str
    result: Any = None
    confidence: float = 0.0
    next_steps: List[str] = field(default_factory=list)


@dataclass
class ContextInfo:
    query: str
    query_type: QueryType
    keywords: List[str]
    entities: Dict[str, Any]
    parameters: Dict[str, float]
    chunks: List[Dict]
    confidence: float = 0.0


class AgentLoop:
    """
    Многошаговый агент для инженерных запросов.
    Совместим с async FormulaEngine.answer_calculation(...).
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

            step2 = self._search_chunks(user_content, query_type, keywords)
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

        except Exception as e:
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

    def _analyze_query(self, query: str) -> ReasoningStep:
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
            parameters = {}
            parsed_numbers = []
            for num in numbers:
                try:
                    parsed_numbers.append(float(num.replace(",", ".")))
                except ValueError:
                    continue

            if parsed_numbers:
                parameters["numeric_values"] = parsed_numbers

            entities = {}
            if "°" in query or "град" in query_lower:
                entities["unit_type"] = "temperature"
            if "м²" in query or "кв.м" in query_lower:
                entities["unit_type"] = "area"
            if "м³" in query or "куб" in query_lower:
                entities["unit_type"] = "volume"

            doc_pattern = r"(СП \d+(?:\.\d+)*)|(ГОСТ \d+(?:-\d+)*)|(СНиП \d+(?:\.\d+)*)|(МСН \d+(?:\.\d+)*)"
            docs = re.findall(doc_pattern, query)
            flat_docs = []
            for doc_group in docs:
                for item in doc_group:
                    if item:
                        flat_docs.append(item)
            if flat_docs:
                entities["documents"] = flat_docs

            step.result = {
                "type": query_type,
                "keywords": keywords,
                "entities": entities,
                "parameters": parameters,
                "raw_query": query
            }
            step.confidence = 0.9

        except Exception as e:
            step.result = None
            step.confidence = 0.0
            step.description += f" (Ошибка: {e})"

        return step

    def _search_chunks(self, query: str, query_type: QueryType, keywords: List[str]) -> ReasoningStep:
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

        except Exception as e:
            step.result = None
            step.confidence = 0.0
            step.description += f" (Ошибка: {e})"

        return step

    def _extract_tables(self, text: str, doc_name: str) -> List[Dict]:
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

    def _extract_formulas(self, text: str) -> List[Dict]:
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

    def _analyze_context(
        self,
        query: str,
        chunks: List[Dict],
        query_type: QueryType,
        entities: Dict,
        parameters: Dict
    ) -> ReasoningStep:
        step = ReasoningStep(
            step_id=3,
            description="Анализ контекста и извлечение ключевой информации"
        )

        try:
            combined_text = "\n".join([c.get("text", "") for c in chunks[:3]])

            extracted_params = {}
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

            extracted_params.update(parameters)

            relevance_score = 0.0
            if chunks:
                relevance_score = len([c for c in chunks if c.get("score", 0) > 0.5]) / len(chunks)

            context_info = ContextInfo(
                query=query,
                query_type=query_type,
                keywords=[w for w in query.split() if len(w) > 2],
                entities=entities,
                parameters=extracted_params,
                chunks=chunks,
                confidence=relevance_score
            )

            step.result = context_info
            step.confidence = relevance_score if relevance_score > 0 else 0.5

        except Exception as e:
            step.result = None
            step.confidence = 0.0
            step.description += f" (Ошибка: {e})"

        return step

    async def _handle_calculation(self, context: ContextInfo) -> ReasoningStep:
        """Шаг 4a: Обработка расчета с поддержкой асинхронного вызова"""
        step = ReasoningStep(
            step_id=4,
            description="Выполнение инженерного расчета"
        )

        try:
            query = context.query
            params = context.parameters or {}

            if params:
                param_str = " ".join([f"{k}={v}" for k, v in params.items()])
                calc_query = f"{query} с параметрами: {param_str}"
            else:
                calc_query = query

            raw_result = self.formula_engine.answer_calculation(calc_query)

            if asyncio.iscoroutine(raw_result):
                result = await raw_result
            else:
                result = raw_result

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

        except Exception as e:
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
        step = ReasoningStep(
            step_id=4,
            description="Поиск определения термина"
        )

        try:
            query = context.query.lower()
            stop_words = [
                "что такое", "определение", "термин", "понятие", "что значит",
                "что означает", "расшифруй", "аббревиатура", "расшифровка",
                "что это", "как понимать", "объясните", "поясните"
            ]

            term = query
            for word in stop_words:
                term = term.replace(word, "").strip()

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

            step.result = {
                "type": "definition",
                "term": term,
                "definition": definition.get("definition", "Определение не найдено"),
                "source": definition.get("source", "Нормативная база"),
                "found": definition.get("found", False),
                "answer": (
                    f"📖 **Определение термина «{term}»:**\n\n"
                    f"{definition.get('definition', 'Определение не найдено')}\n\n"
                    f"📚 **Источник:** {definition.get('source', 'Нормативная база')}"
                ),
                "sources": [{"doc_name": definition.get("source", "Нормативная база")}],
                "tables": [],
                "formulas": [],
                "confidence": 0.9 if definition.get("found") else 0.3
            }
            step.confidence = 0.9 if definition.get("found") else 0.3

        except Exception as e:
            step.result = None
            step.confidence = 0.0
            step.description += f" (Ошибка: {e})"

        return step

    def _handle_comparison(self, context: ContextInfo) -> ReasoningStep:
        step = ReasoningStep(
            step_id=4,
            description="Сравнение параметров"
        )

        try:
            query = context.query.lower()
            parts = re.split(r"\s+и\s+|\s+vs\s+|\s+против\s+", query)

            if len(parts) > 1:
                items = [p.strip() for p in parts if len(p.strip()) > 2]
                comparison_results = []

                for item in items:
                    result = self.qa_system.answer(f"Что такое {item}?")
                    if result.get("sources"):
                        comparison_results.append({
                            "item": item,
                            "info": result.get("answer", "")[:200],
                            "source": result["sources"][0].get("doc_name", "Документ")
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
                    "answer": "Не удалось определить объекты для сравнения",
                    "sources": [],
                    "tables": [],
                    "formulas": [],
                    "confidence": 0.0
                }
                step.confidence = 0.0

        except Exception as e:
            step.result = None
            step.confidence = 0.0
            step.description += f" (Ошибка: {e})"

        return step

    def _handle_regulatory(self, context: ContextInfo) -> ReasoningStep:
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
                answer_lines.append("Требования не найдены")

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

        except Exception as e:
            step.result = None
            step.confidence = 0.0
            step.description += f" (Ошибка: {e})"

        return step

    def _handle_search(self, context: ContextInfo) -> ReasoningStep:
        step = ReasoningStep(
            step_id=4,
            description="Поиск информации"
        )

        try:
            result = self.qa_system.answer(context.query)

            step.result = {
                "type": "search",
                "answer": result.get("answer", "Информация не найдена"),
                "sources": result.get("sources", []),
                "tables": result.get("tables", []),
                "formulas": result.get("formulas", []),
                "confidence": 0.8 if result.get("sources") else 0.3
            }
            step.confidence = 0.8 if result.get("sources") else 0.3

        except Exception as e:
            step.result = None
            step.confidence = 0.0
            step.description += f" (Ошибка: {e})"

        return step

    def _check_completeness(self, result: Dict, context: ContextInfo) -> ReasoningStep:
        step = ReasoningStep(
            step_id=5,
            description="Проверка полноты ответа"
        )

        try:
            needs_clarification = False
            questions = []

            if not result or not result.get("answer"):
                needs_clarification = True
                questions.append("Пожалуйста, уточните ваш вопрос.")

            if not result.get("sources") and not result.get("source"):
                if context.query_type != QueryType.CALCULATION:
                    needs_clarification = True
                    questions.append("Не удалось найти источники. Пожалуйста, переформулируйте вопрос.")

            confidence = result.get("confidence", 0.0)
            if confidence < 0.3 and result.get("answer"):
                needs_clarification = True
                questions.append("Информация может быть неполной. Уточните параметры вашего запроса.")

            # Смягченная проверка для расчётов
            if context.query_type == QueryType.CALCULATION:
                formula_params = result.get("parameters", {})
                calc_value = result.get("result", None)
                calc_answer = result.get("answer", "")

                has_explicit_failure = (
                    "Недостаточно данных" in calc_answer
                    or "Укажите числовые параметры" in calc_answer
                    or "не удалось определить тип расчёта" in calc_answer.lower()
                )

                if not formula_params and calc_value is None and has_explicit_failure:
                    needs_clarification = True
                    questions.append("Укажите числовые параметры для расчета.")

            step.result = {
                "needs_clarification": needs_clarification,
                "questions": questions,
                "completeness_score": max(0.0, 1.0 - len(questions) * 0.2) if questions else 1.0
            }
            step.confidence = 1.0 if not needs_clarification else 0.5

        except Exception as e:
            step.result = {
                "needs_clarification": True,
                "questions": ["Произошла ошибка при проверке ответа"],
                "completeness_score": 0.0
            }
            step.confidence = 0.0
            step.description += f" (Ошибка: {e})"

        return step

    def _generate_final_response(
        self,
        result: Dict,
        completeness: Dict,
        context: ContextInfo
    ) -> ReasoningStep:
        step = ReasoningStep(
            step_id=6,
            description="Формирование финального ответа"
        )

        try:
            answer = result.get("answer", "")

            # Если это расчет - добавляем reasoning
            if result.get("type") == "calculation":
                reasoning = result.get("reasoning", "")
                if reasoning and "Ход расчёта" not in answer:
                    answer += f"\n\n**Ход расчёта:**\n```text\n{reasoning}\n```"

            sources = result.get("sources", [])
            tables = result.get("tables", [])
            formulas = result.get("formulas", [])

            if sources and "📚 **Источники:**" not in answer:
                answer += "\n\n📚 **Источники:**\n"
                for src in sources[:2]:
                    if isinstance(src, dict):
                        answer += f"• {src.get('doc_name', 'Документ')}\n"
                    else:
                        answer += f"• {src}\n"

            if tables and "📊 **Таблицы:**" not in answer:
                answer += "\n\n📊 **Таблицы:**\n"
                for table in tables[:1]:
                    content = table.get("content", "")[:300]
                    title = table.get("title", "Таблица")
                    answer += f"\n**{title}**:\n```\n{content}...\n```\n"

            if formulas and "📐 **Формулы:**" not in answer:
                answer += "\n\n📐 **Формулы:**\n"
                for formula in formulas[:2]:
                    if isinstance(formula, dict):
                        raw = formula.get("raw") or formula.get("expression") or formula.get("name", "")
                        answer += f"\n`{raw}`\n"
                    else:
                        answer += f"\n`{formula}`\n"

            if len(answer) < 100 and context.chunks and context.query_type != QueryType.CALCULATION:
                answer += "\n\n📖 **Дополнительная информация:**\n"
                for chunk in context.chunks[:1]:
                    answer += f"\n{chunk.get('text', '')[:200]}...\n"

            step.result = {
                "answer": answer,
                "sources": sources,
                "tables": tables,
                "formulas": formulas,
                "confidence": result.get("confidence", 0.5),
                "needs_clarification": completeness.get("needs_clarification", False),
                "questions": completeness.get("questions", [])
            }
            step.confidence = 0.9

        except Exception as e:
            step.result = None
            step.confidence = 0.0
            step.description += f" (Ошибка: {e})"

        return step

    def _format_response(self, result: Dict, context: ContextInfo, query_type: QueryType) -> Dict:
        response = {
            "answer": result.get("answer", "Извините, не удалось сформировать ответ."),
            "sources": result.get("sources", []),
            "tables": result.get("tables", []),
            "formulas": result.get("formulas", []),
            "confidence": result.get("confidence", 0.5),
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

    def _create_error_response(self, message: str, steps: int = 1) -> Dict:
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
        chain = "🔍 **Цепочка рассуждений:**\n\n"
        for step in self.reasoning_steps:
            confidence_stars = "⭐" * int(step.confidence * 5)
            chain += f"**Шаг {step.step_id}:** {step.description}\n"
            chain += f"   Уверенность: {step.confidence:.0%} {confidence_stars}\n\n"
        return chain

    def get_reasoning_json(self) -> Dict:
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