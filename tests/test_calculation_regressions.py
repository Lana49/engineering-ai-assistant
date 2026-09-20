"""Офлайн-регрессии intent, числовых расчётов и маршрутизации RAG."""

import asyncio
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from core.agent_loop import AgentLoop
from core.formula_engine import FormulaEngine
from core.query_parser import extract_city, extract_variables, parse_query
from core.retrieval_memory import RetrievalMemory


class CalculationRegressionTests(unittest.TestCase):
    def setUp(self):
        self.engine = FormulaEngine()

    def answer(self, query, parameters=None):
        return asyncio.run(self.engine.answer_calculation(query, parameters=parameters))

    def test_temperature_notations_and_explicit_difference(self):
        for notation in ("Δt", "delta_t", "dt", "разность температур"):
            with self.subTest(notation=notation):
                result = self.answer(f"Рассчитай теплопотери: A=12, R=3, {notation}=50")
                self.assertEqual(result["result"], 200.0)
                self.assertFalse(result["needs_clarification"])
        explicit = self.answer("Теплопотери A=12 R=3 dt=40 tв=20 tн=-30")
        self.assertTrue(explicit["needs_clarification"])
        self.assertNotIn("result", explicit)
        self.assertIn("Δt=40", explicit["answer"])
        consistent = self.answer("Теплопотери A=12 R=3 dt=50 tв=20 tн=-30")
        self.assertEqual(consistent["result"], 200.0)

    def test_temperature_difference_derived_after_api_override(self):
        result = self.answer("Теплопотери A=10 R=2 tв=20 tн=-20", {"t_n": -30.0})
        self.assertEqual(result["result"], 250.0)
        self.assertEqual(result["params"]["delta_t"], 50.0)

    def test_temperature_conflict_is_checked_after_overrides_and_parser_echo(self):
        query = "Рассчитай теплопотери: A=10, R=2, tв=20, tн=-10, delta_t=50"
        # AgentLoop передаёт извлечённые параметры повторно; это не согласие
        # на конфликтующие значения и не осознанное исправление через API.
        for parameters in (None, extract_variables(query)):
            with self.subTest(parameters=parameters):
                result = self.answer(query, parameters)
                self.assertTrue(result["needs_clarification"])
                self.assertNotIn("result", result)
                self.assertIn("30 °C", result["answer"])
        corrected_difference = self.answer(query, {"delta_t": 30.0})
        self.assertEqual(corrected_difference["result"], 150.0)
        corrected_temperature = self.answer(query, {"t_n": -30.0})
        self.assertEqual(corrected_temperature["result"], 250.0)

    def test_length_and_power_units(self):
        for thickness in ("100 мм", "10 см", "0,1 м"):
            with self.subTest(thickness=thickness):
                result = self.answer(f"Рассчитай сопротивление слоя: толщина {thickness}, λ=0,04")
                self.assertAlmostEqual(result["result"], 2.5)
        result = self.answer("Рассчитай удельный тепловой поток: Q=2 кВт, длина трубы 1000 см")
        self.assertEqual(result["result"], 200.0)
        power_name = self.answer("Рассчитай тепловой поток трубы: мощность 10 кВт, длина 5 м")
        self.assertEqual(power_name["result"], 2000.0)

    def test_ambiguous_length_and_incompatible_units_require_clarification(self):
        queries = (
            "Рассчитай тепловой поток трубы: Q=1000 Вт, длина 5 м, расход воздуха 100 м3/ч",
            "Рассчитай тепловой поток трубы: Q=1000 Вт, L=100 м3/ч",
            "Рассчитай вентиляцию: L=100 м, tв=20, tн=-10",
            "Рассчитай вентиляцию: L=100 м3/сут, tв=20, tн=-10",
            "Рассчитай теплопотери: A=10 м, R=2, dt=30",
            "Рассчитай сопротивление слоя: толщина 100 м2, λ=0,04",
            "Рассчитай тепловой поток трубы: Q=2 кВт/ч, длина 5 м",
        )
        for query in queries:
            with self.subTest(query=query):
                result = self.answer(query)
                self.assertTrue(result["needs_clarification"])
                self.assertNotIn("result", result)
                sync_result = self.engine.try_calculate(query)
                self.assertTrue(sync_result["needs_clarification"])
                self.assertNotIn("result", sync_result)

    def test_flow_units_and_russian_parameter_names(self):
        result = self.answer("Рассчитай вентиляцию: расход воздуха 0,5 м3/с, температура внутри 20, наружная температура -30")
        self.assertAlmostEqual(result["result"], 30150.0)
        self.assertEqual(result["params"]["L"], 1800.0)
        litres = self.answer("Рассчитай вентиляцию: расход воздуха 100 л/мин, tв=20, tн=-10")
        self.assertAlmostEqual(litres["params"]["L"], 6.0)
        self.assertAlmostEqual(litres["result"], 60.3)

    def test_unsupported_target_is_not_replaced_by_complete_parameters(self):
        for target in ("диаметр трубы", "диаметр вентиляции", "скорость приточного воздуха", "объем помещения"):
            with self.subTest(target=target):
                query = f"Рассчитай {target}: L=100 м3/ч, tв=20, tн=-10"
                result = self.answer(query)
                self.assertTrue(result["needs_clarification"])
                self.assertNotIn("result", result)
                self.assertFalse(self.engine.can_calculate_directly(query))
                self.assertTrue(self.engine.try_calculate(query)["needs_clarification"])
        # Неназванный предмет при единственной полной формуле остаётся допустим.
        result = self.answer("Рассчитай: A=10, R=2, dt=30")
        self.assertEqual(result["result"], 150.0)

    def test_requested_formula_has_priority_over_other_complete_params(self):
        result = self.answer("Рассчитай толщину утеплителя: толщина 100 мм, λ=0,04")
        self.assertTrue(result["needs_clarification"])
        self.assertIn("R_tr", result["answer"])
        correct = self.answer("Рассчитай толщину утеплителя: R_tr=3, λ=0,04")
        self.assertAlmostEqual(correct["result"], 0.12)

    def test_polite_calculation_requests_keep_supported_subjects(self):
        for prefix in (
            "Рассчитайте", "Помоги рассчитать", "Помогите, пожалуйста, рассчитать",
            "Вычислите", "Определите", "Найдите", "Посчитайте", "Можете рассчитать",
        ):
            with self.subTest(prefix=prefix):
                result = self.answer(f"{prefix} теплопотери: A=10, R=2, dt=30")
                self.assertEqual(result["result"], 150.0)
                unsupported = self.answer(f"{prefix} диаметр вентиляции: L=100, tв=20, tн=-10")
                self.assertTrue(unsupported["needs_clarification"])
                self.assertNotIn("result", unsupported)
        for subject in ("тепловую мощность вентиляции", "мощность приточной вентиляции"):
            with self.subTest(subject=subject):
                result = self.answer(f"Рассчитай {subject}: L=100, tв=20, tн=-10")
                self.assertAlmostEqual(result["result"], 1005.0)
        electric = self.answer("Рассчитай электрическую мощность вентиляции: L=100, tв=20, tн=-10")
        self.assertTrue(electric["needs_clarification"])
        self.assertNotIn("result", electric)

    def test_formula_validation_and_nonfinite_inputs(self):
        cases = (
            ("Теплопотери A=10 R=0 dt=20", None),
            ("Сопротивление слоя δ=0,1 λ=0", None),
            ("Удельный тепловой поток Q=100 L=0", None),
            ("Рассчитай ГСОП tв=20 tот=-8 zот=500", None),
            ("Теплопотери A=10 R=2 dt=20", {"A": math.inf}),
        )
        for query, parameters in cases:
            with self.subTest(query=query):
                with self.assertLogs("core.formula_engine", level="ERROR"):
                    result = self.answer(query, parameters)
                self.assertTrue(result["needs_clarification"])
                self.assertNotIn("result", result)

    def test_complete_calc_with_document_code_and_regulatory_question(self):
        calculate = "Рассчитай вентиляцию по СП 60.13330: L=500 tв=20 tн=-30"
        self.assertEqual(parse_query(calculate).intent, "calculation")
        self.assertEqual(parse_query("Какие требования к вентиляции по СП 60.13330?").intent, "regulatory")
        self.assertEqual(parse_query("Что такое вентиляция?").intent, "definition")
        self.assertNotEqual(parse_query("Как рассчитать вентиляцию по СП 60.13330?").intent, "calculation")
        self.assertAlmostEqual(self.answer(calculate)["result"], 8375.0)

    def test_missing_parameter_and_unambiguous_long_alias(self):
        parsed = extract_variables("требуемое сопротивление 3,5 λ=0,04")
        self.assertEqual(parsed["R_tr"], 3.5)
        self.assertNotIn("R", parsed)
        result = self.answer("Рассчитай вентиляцию L=500 tв=20")
        self.assertTrue(result["needs_clarification"])
        self.assertIn("t_n", result["answer"])

    def test_climate_enrichment_preserves_user_temperature(self):
        class ClimateStub:
            @staticmethod
            def _climate_values_from_tables(city, required):
                self.assertEqual(city.lower(), "томск")
                self.assertEqual(required, ("z_ot",))
                return {"z_ot": 225.0}, {"doc_name": "Проверочная таблица", "page": 3}

        self.engine.qa_system = object()
        self.engine._table_calculator = ClimateStub()
        result = self.answer("Рассчитай ГСОП для Томска: tв=20, tот=-5")
        self.assertEqual(result["result"], 5625.0)
        self.assertEqual(result["params"]["t_ot"], -5.0)
        self.assertEqual(result["sources"][0]["doc_name"], "Проверочная таблица")
        self.assertTrue(result["grounded"])

    def test_no_assumed_indoor_temperature_for_city_heat_loss(self):
        self.engine.qa_system = object()
        result = self.answer("Рассчитай теплопотери для Томска A=20 R=3")
        self.assertTrue(result["needs_clarification"])
        self.assertIn("t_v", result["answer"])
        self.assertIsNone(result.get("result"))
        self.assertEqual(extract_city("для Томска"), "томск")

    def test_calculation_response_is_structured_and_does_not_claim_document_evidence(self):
        result = self.answer("Рассчитай ГСОП tв=20 tот=-8,4 zот=225")
        self.assertEqual(result["result"], 6390.0)
        for heading in ("Исходные данные", "Формула", "Подстановка", "Результат"):
            self.assertIn(heading, result["answer"])
        self.assertEqual(result["sources"], [])


class AgentRoutingRegressionTests(unittest.TestCase):
    def test_search_result_object_and_each_grounded_intent(self):
        class QAStub:
            is_ready = True

            @staticmethod
            def search(query, top_k=5):
                assert query and top_k
                return [SimpleNamespace(
                    doc_name="Тестовый документ", text="Вентиляция должна обеспечивать воздухообмен.",
                    score=0.8, chunk_id=0, metadata={"page": 5},
                )]

            @staticmethod
            def answer_from_results(question, results):
                assert question and results and results[0]["metadata"]["page"] == 5
                return {"answer": "Требование [Источник 1]", "sources": [{"doc_name": "Тестовый документ"}],
                        "confidence": 0.8, "grounded": True, "used_llm": False, "needs_clarification": False}

        with TemporaryDirectory() as directory:
            agent = AgentLoop(QAStub(), FormulaEngine())
            agent.memory = RetrievalMemory(Path(directory) / "memory.json")
            for query in ("Что такое вентиляция?", "Какие требования к вентиляции?", "Сравни вентиляцию и отопление", "Найди воздухообмен"):
                with self.subTest(query=query):
                    result = asyncio.run(agent.run(query))
                    self.assertTrue(result["grounded"])
                    self.assertEqual(result["sources"][0]["doc_name"], "Тестовый документ")

    def test_direct_calculation_does_not_require_index_or_ollama(self):
        with TemporaryDirectory() as directory:
            agent = AgentLoop(None, FormulaEngine())
            agent.memory = RetrievalMemory(Path(directory) / "memory.json")
            result = asyncio.run(agent.run("Рассчитай вентиляцию по СП 60.13330 L=500 tв=20 tн=-30"))
            self.assertEqual(result["query_type"], "calculation")
            self.assertEqual(result["steps"], 2)
            self.assertAlmostEqual(result["result"], 8375.0)

    def test_agent_does_not_accept_conflicting_temperature_or_length(self):
        with TemporaryDirectory() as directory:
            agent = AgentLoop(None, FormulaEngine())
            agent.memory = RetrievalMemory(Path(directory) / "memory.json")
            for query in (
                "Рассчитай теплопотери: A=10, R=2, tв=20, tн=-10, delta_t=50",
                "Рассчитай тепловой поток трубы: Q=1000 Вт, длина 5 м, расход воздуха 100 м3/ч",
                "Рассчитай диаметр трубы: L=100 м3/ч, tв=20, tн=-10",
            ):
                with self.subTest(query=query):
                    result = asyncio.run(agent.run(query))
                    self.assertTrue(result["needs_clarification"])
                    self.assertIsNone(result.get("result"))


if __name__ == "__main__":
    unittest.main()

# ИСПРАВЛЕНО: регрессии неоднозначного L, несовместимых единиц, предмета расчёта и противоречивого Δt с API overrides.
