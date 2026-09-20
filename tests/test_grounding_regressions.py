"""Отрицательные проверки содержательной опоры, независимо от наличия Ollama."""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from core.qa_engine import QASystem, SearchResult


class GroundingRegressionTests(unittest.TestCase):
    def setUp(self):
        self.qa = QASystem(use_llm=False, use_embeddings=False, min_score=0.05)

    def answer(self, query, text, invented="Выдуманный ответ [Источник 1]"):
        result = SearchResult("СП 60.13330.2020.docx", 1, text, 0.99)
        with patch.object(self.qa, "_select_provider", return_value="ollama"), \
             patch.object(self.qa, "_ask_ollama", return_value=invented) as model:
            answer = self.qa.answer_from_results(query, [result])
        return answer, model

    def test_definition_term_does_not_include_search_instructions(self):
        for query in (
            "Что такое рабочая зона по СП 60.13330? Найди определение.",
            "Найди определение рабочая зона согласно СП 60.13330.2020.",
            "Что означает «рабочая зона»?",
        ):
            with self.subTest(query=query):
                self.assertEqual(self.qa._definition_term(query), "рабочая зона")

    def test_working_zone_bibliography_and_mentions_cannot_define_term(self):
        query = "Что такое рабочая зона по СП 60.13330? Найди определение."
        for text in (
            "2 Нормативные ссылки\nГОСТ 12.1.005-88 Общие санитарно-гигиенические требования к воздуху рабочей зоны.",
            "7.3.16 Загрязнённый воздух следует удалять из рабочей зоны помещений.",
            "Библиография\n[1] ГОСТ 12.1.005 Рабочая зона — термин и определение.",
            "Рабочая зона: см. ГОСТ 12.1.005-88.",
        ):
            with self.subTest(text=text):
                answer, model = self.answer(query, text, "Рабочая зона — высотой 2 м. [Источник 1]")
                self.assertTrue(answer["needs_clarification"])
                self.assertEqual(answer["sources"], [])
                self.assertNotIn("2 м", answer["answer"])
                model.assert_not_called()

    def test_find_definition_api_also_rejects_mentions(self):
        self.qa.build_index([{"doc_name": "СП 60.13330.2020.docx", "chunks": [{
            "chunk_id": 1, "text": "Рабочая зона должна обеспечиваться приточным воздухом.",
        }]}])
        self.assertFalse(self.qa.find_definition("рабочая зона")["found"])

    def test_positive_definition_is_exact_and_does_not_add_model_claims(self):
        text = "3.1.21 рециркуляция воздуха: Смешение воздуха из помещения с наружным воздухом."
        answer, model = self.answer("Что такое рециркуляция воздуха по СП 60.13330.2020?", text)
        self.assertFalse(answer["needs_clarification"])
        self.assertIn(text, answer["answer"])
        self.assertEqual(answer["evidence_mode"], "exact_excerpt")
        self.assertFalse(answer["used_llm"])
        self.assertEqual(answer["sources"][0]["reference_id"], 1)
        model.assert_not_called()

    def test_norm_without_direct_requirement_is_not_invented(self):
        answer, model = self.answer(
            "Какая минимальная высота рабочей зоны по СП 60.13330?",
            "Рабочая зона используется для размещения оборудования.",
            "Высота рабочей зоны должна составлять 2 м [Источник 1]",
        )
        self.assertTrue(answer["needs_clarification"])
        self.assertNotIn("2 м", answer["answer"])
        model.assert_not_called()

    def test_valid_citation_number_cannot_authorize_unsupported_norm(self):
        text = "6.4.14 Температура поверхности панелей допускается не более 60 °C. При этом следует исключать ожоги."
        answer, model = self.answer(
            "Какую температуру поверхности панелей допускает СП 60.13330.2020?", text,
            "Температура должна быть не менее 100 °C [Источник 1]",
        )
        self.assertIn(text, answer["answer"])
        self.assertNotIn("100 °C", answer["answer"])
        self.assertNotIn("не менее", answer["answer"])
        model.assert_not_called()

    def test_requested_clause_keeps_following_exceptions_and_its_own_number(self):
        text = ("7.2.1 Испытание допускается при температуре 10 °C.\n"
                "7.2.2 Падение давления должно быть не более 0,05 МПа.\n\nПри этом утечки не допускаются.\n"
                "7.2.3 Падение давления должно быть не более 0,02 МПа.")
        answer, model = self.answer("Что требует пункт 7.2.2 СП 60.13330.2020?", text)
        self.assertIn("0,05 МПа", answer["answer"])
        self.assertIn("утечки не допускаются", answer["answer"])
        self.assertNotIn("0,02 МПа", answer["answer"])
        self.assertNotIn("10 °C", answer["answer"])
        model.assert_not_called()

    def test_missing_document_revision_and_clause_cannot_be_replaced(self):
        for query in ("Что требует пункт 9.9.9 СП 60.13330.2020?",
                      "Что требует пункт 7.2.2 СП 60.13330.1999?"):
            with self.subTest(query=query):
                answer, model = self.answer(query, "7.2.2 Давление должно быть не более 0,05 МПа.")
                self.assertTrue(answer["needs_clarification"])
                self.assertEqual(answer["sources"], [])
                model.assert_not_called()

    def test_long_norm_keeps_late_exceptions_without_expanding_llm_context(self):
        text = ("7.2.2 Давление должно быть не более 0,05 МПа.\n\n"
                + "Дополнительные условия испытания. " * 65
                + "\n\nПри замерзании воды испытания запрещены.")
        answer, model = self.answer("Что требует пункт 7.2.2 СП 60.13330.2020?", text)
        self.assertIn("При замерзании воды испытания запрещены.", answer["answer"])
        self.assertLess(len(answer["context"]), 2000)
        model.assert_not_called()

    def test_norm_for_matching_operating_condition_outweighs_semantic_rank(self):
        wrong = SearchResult("СП 60.13330.2020.docx", 1,
            "5.9 В горячих цехах следует предусматривать охлаждающие панели рабочих мест. Температуру воздуха следует принимать 20 °C.", 0.99)
        right = SearchResult("СП 60.13330.2020.docx", 2,
            "6.4.14 Температуру поверхности панелей радиационного обогрева рабочих мест следует принимать не выше 60 °C.", 0.8)
        response = self.qa.answer_from_results(
            "Какую температуру поверхности панелей радиационного обогрева рабочих мест допускает СП 60.13330.2020?", [wrong, right])
        self.assertIn("60 °C", response["answer"])
        self.assertNotIn("20 °C", response["answer"])
        self.qa.build_index([{"doc_name": right.doc_name, "chunks": [wrong.to_dict(), right.to_dict()]}])
        self.qa.use_embeddings = True
        self.qa.embedding_model = Mock()
        self.qa.embedding_model.encode.return_value = np.array([[1.0, 0.0]])
        self.qa.chunk_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.qa.semantic_weight = 0.7
        self.qa.lexical_weight = 0.3
        response = self.qa.answer(
            "Какую температуру поверхности панелей радиационного обогрева рабочих мест допускает СП 60.13330.2020?", top_k=1)
        self.assertIn("60 °C", response["answer"])

    def test_unrelated_high_retrieval_score_is_not_evidence(self):
        for query in ("Какие нормы действуют для квантового телепортатора?",
                      "Объясни орбиту Нептуна"):
            with self.subTest(query=query):
                answer, model = self.answer(query, "7.3.16 Загрязнённый воздух следует удалять из помещений.")
                self.assertTrue(answer["needs_clarification"])
                self.assertEqual(answer["confidence"], 0)
                model.assert_not_called()

    def test_normal_explanation_keeps_llm_generation(self):
        answer, model = self.answer(
            "Объясни удаление загрязнённого воздуха",
            "Загрязнённый воздух удаляют вытяжными системами.",
            "Загрязнённый воздух удаляют вытяжными системами. [Источник 1]",
        )
        self.assertTrue(answer["used_llm"])
        self.assertFalse(answer["needs_clarification"])
        model.assert_called_once()


if __name__ == "__main__":
    unittest.main()

# ИСПРАВЛЕНО: библиография, неподтверждённые нормы/цифры, отсутствующие редакции/пункты и вопросы вне базы не дают выдуманный ответ.
