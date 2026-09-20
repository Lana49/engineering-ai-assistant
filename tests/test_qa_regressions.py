"""Офлайн-регрессии RAG: поиск, контракты, источники и отказ локальной модели."""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import requests

from core.qa_engine import QASystem, SearchResult
from core import prompts


DOCUMENTS = [
    {"doc_name": "Вентиляция.docx", "chunks": [{
        "chunk_id": 1,
        "text": "Вентиляция обеспечивает организованный воздухообмен в помещениях. "
                "Для помещений с выделением вредностей предусматривают вытяжку.",
        "metadata": {"page": 4},
    }]},
    {"doc_name": "Канализация.pdf", "chunks": [{
        "chunk_id": 2,
        "text": "Канализация служит для отвода сточных вод. Самотечные трубопроводы укладывают с уклоном.",
        "metadata": {"page": 12},
    }]},
]


class QARegressionTests(unittest.TestCase):
    def setUp(self):
        self.qa = QASystem(use_llm=False, use_embeddings=False, min_score=0.05)
        self.assertTrue(self.qa.build_index(DOCUMENTS))

    def test_declensions_and_document_selection(self):
        for query in ("Что такое вентиляция?", "требования к вентиляции"):
            with self.subTest(query=query):
                found = self.qa.search(query)
                self.assertTrue(found)
                self.assertEqual(found[0].doc_name, "Вентиляция.docx")
        self.assertEqual(self.qa.search("отвод сточных вод")[0].doc_name, "Канализация.pdf")
        self.assertEqual(self.qa.search(""), [])
        self.assertEqual(self.qa.search("вентиляция", top_k=0), [])

    def test_missing_embedding_model_does_not_reduce_lexical_scores(self):
        before = self.qa.search("вентиляция")[0].score
        self.qa.use_embeddings = True
        self.qa.chunk_embeddings = np.ones((2, 3), dtype=np.float32)
        self.qa.embedding_model = None
        self.assertAlmostEqual(self.qa.search("вентиляция")[0].score, before)
        self.qa.embedding_model = Mock()
        self.qa.embedding_model.encode.side_effect = RuntimeError("model unavailable")
        with self.assertLogs("core.qa_engine", level="ERROR"):
            after = self.qa.search("вентиляция")[0].score
        self.assertAlmostEqual(after, before)

    def test_semantic_only_search_works_without_lexical_overlap(self):
        self.qa.use_embeddings = True
        self.qa.embedding_model = Mock()
        self.qa.embedding_model.encode.return_value = np.array([[0.0, 1.0]])
        self.qa.chunk_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        found = self.qa.search("unseenenglishterm")
        self.assertEqual(found[0].doc_name, "Канализация.pdf")
        self.assertEqual(found[0].score, 1.0)

    def test_embedding_build_failure_keeps_usable_index(self):
        self.qa.use_embeddings = True
        self.qa.embedding_model = Mock()
        self.qa.embedding_model.encode.side_effect = RuntimeError("out of memory")
        with self.assertLogs("core.qa_engine", level="ERROR"):
            self.assertTrue(self.qa.build_index(DOCUMENTS))
        self.assertIsNone(self.qa.chunk_embeddings)
        self.assertTrue(self.qa.search("сточные воды"))
        self.assertIn("embedding_error", self.qa.last_index_diagnostics)

    def test_saved_index_does_not_change_runtime_settings_or_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "index.pkl"
            self.qa.chunk_embeddings = np.ones((2, 3), dtype=np.float32)
            self.assertTrue(self.qa.save_index(path))
            target = QASystem(use_llm=False, use_embeddings=False, top_k=2, min_score=0.07,
                              ollama_model="local-only", embedding_model_name="different")
            with self.assertLogs("core.qa_engine", level="WARNING"):
                self.assertTrue(target.load_index(path))
            self.assertEqual((target.top_k, target.min_score), (2, 0.07))
            self.assertEqual(target.ollama_model, "local-only")
            self.assertEqual(target.embedding_model_name, "different")
            self.assertFalse(target.use_embeddings)
            self.assertIsNone(target.chunk_embeddings)
            self.assertTrue(target.search("вентиляция"))

    def test_corrupt_index_does_not_replace_working_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.pkl"
            with path.open("wb") as handle:
                pickle.dump({"documents": [], "chunks": [None]}, handle)
            with self.assertLogs("core.qa_engine", level="ERROR"):
                self.assertFalse(self.qa.load_index(path))
            self.assertEqual(self.qa.search("вентиляция")[0].doc_name, "Вентиляция.docx")

    def test_matrix_without_vectorizer_is_not_a_ready_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "incomplete.pkl"
            with path.open("wb") as handle:
                pickle.dump({"documents": DOCUMENTS, "chunks": self.qa.chunks,
                             "tfidf_matrix": self.qa.tfidf_matrix}, handle)
            with self.assertLogs("core.qa_engine", level="ERROR"):
                self.assertFalse(self.qa.load_index(path))
            self.assertTrue(self.qa.search("вентиляция"))

    def test_mapping_adapter_preserves_pages_and_citation_numbers(self):
        result = self.qa.search("вентиляция")[0]
        self.assertEqual(SearchResult.from_value(result.to_dict()), result)
        response = self.qa.answer_from_results("вентиляция", [result.to_dict(), result])
        self.assertEqual(len(response["sources"]), 1)
        self.assertEqual(response["sources"][0]["reference_id"], 1)
        self.assertEqual(response["sources"][0]["page"], 4)
        self.assertIn("страница 4", response["context"])
        self.assertIn("[Источник 1]", response["answer"])

    def test_context_is_bounded_and_keeps_relevant_window(self):
        text = "Вступление.\n" * 1000 + "Вентиляция — организованный воздухообмен.\n" + "Приложение.\n" * 1000
        result = SearchResult("Пример.pdf", 1, text, 0.8, metadata={"page": 8})
        response = self.qa.answer_from_results("вентиляция", [result] * 8)
        self.assertLess(len(response["context"]), 12000)
        self.assertIn("Вентиляция — организованный воздухообмен", response["context"])
        self.assertEqual(result.text, text)

    def test_model_response_requires_existing_citations(self):
        results = self.qa.search("вентиляция")
        with patch.object(self.qa, "_select_provider", return_value="ollama"), \
             patch.object(self.qa, "_ask_ollama", return_value="Выдуманная норма [Источник 99]"):
            with self.assertLogs("core.qa_engine", level="WARNING"):
                rejected = self.qa.answer_from_results("вентиляция", results)
        self.assertFalse(rejected["used_llm"])
        self.assertNotIn("Выдуманная норма", rejected["answer"])
        with patch.object(self.qa, "_select_provider", return_value="ollama"), \
             patch.object(self.qa, "_ask_ollama", return_value="Вентиляция обеспечивает воздухообмен. [Источник 1]"):
            accepted = self.qa.answer_from_results("вентиляция", results)
        self.assertTrue(accepted["used_llm"])
        self.assertTrue(accepted["answer"].startswith("### Краткий ответ"))

    def test_ollama_failure_is_logged_and_fallback_is_available(self):
        with patch("core.qa_engine.requests.post", side_effect=requests.ConnectionError("offline")):
            with self.assertLogs("core.qa_engine", level="ERROR"):
                self.assertIsNone(self.qa._ask_ollama("тест"))
        self.assertIn("ConnectionError", self.qa.last_llm_error)
        self.assertTrue(self.qa.answer("вентиляция")["sources"])

    def test_prompts_and_empty_context(self):
        prompts._run_self_tests()
        response = self.qa.answer_from_results("неизвестное условие", [])
        self.assertTrue(response["needs_clarification"])
        self.assertEqual(response["sources"], [])
        self.assertEqual(response["confidence"], 0.0)

    def test_ollama_server_without_requested_model_is_not_available(self):
        response = Mock()
        response.json.return_value = {"models": [{"name": "llama3.1:8b"}]}
        self.qa.ollama_model = "phi3:mini"
        self.qa.use_llm = True
        with patch("core.qa_engine.requests.get", return_value=response):
            self.assertFalse(self.qa.is_ollama_available())
            self.assertEqual(self.qa.get_selected_provider(), "none")
            self.assertIn("phi3:mini", self.qa.last_llm_error)
            self.qa.ollama_model = "llama3.1:8b"
            self.assertTrue(self.qa.is_ollama_available())
            self.assertEqual(self.qa.get_selected_provider(), "ollama")

    def test_failed_save_preserves_previous_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "index.pkl"
            self.assertTrue(self.qa.save_index(path))
            previous = path.read_bytes()
            with patch("core.qa_engine.pickle.dump", side_effect=OSError("disk full")):
                with self.assertLogs("core.qa_engine", level="ERROR"):
                    self.assertFalse(self.qa.save_index(path))
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(temp_dir).glob("*.tmp")), [])

    def test_explicit_document_and_clause_are_retrieval_constraints(self):
        docs = [
            {"doc_name": "СП 60.13330.2020.docx", "chunks": [{"chunk_id": 0,
                "text": "3.1.21 рециркуляция воздуха: Смешение воздуха из помещения с наружным воздухом."}]},
            {"doc_name": "СП 50.13330.2024.docx", "chunks": [{"chunk_id": 0,
                "text": "Рециркуляция воздуха применяется по СП 60.13330.2020, приложение В."}]},
            {"doc_name": "СП 73.13330.2016.docx", "chunks": [
                {"chunk_id": 0, "text": "7.1.3 Испытания гидростатическим методом систем водоснабжения."},
                {"chunk_id": 1, "text": "7.2.2 Система считается выдержавшей испытание, если падение давления не более 0,05 МПа."},
            ]},
        ]
        self.assertTrue(self.qa.build_index(docs))
        defined = self.qa.answer("Что такое рециркуляция воздуха по СП 60.13330.2020?")
        self.assertIn("Смешение воздуха", defined["answer"])
        self.assertTrue(all(source["doc_name"] == "СП 60.13330.2020.docx" for source in defined["sources"]))
        clause = self.qa.answer("Что требует пункт 7.2.2 СП 73.13330.2016 при испытании водоснабжения?")
        self.assertIn("0,05 МПа", clause["answer"])
        self.assertEqual(self.qa.search("Рециркуляция по СП 60.13330.1999"), [])
        self.assertEqual(self.qa.search("Что требует пункт 9.9.9 СП 73.13330.2016?"), [])
        reordered = self.qa.answer_from_results("Что такое рециркуляция воздуха по СП 60.13330.2020?", [
            SearchResult("СП 50.13330.2024.docx", 0, docs[1]["chunks"][0]["text"], 1.5),
            SearchResult("СП 60.13330.2020.docx", 0, docs[0]["chunks"][0]["text"], 0.8),
        ])
        self.assertIn("Смешение воздуха", reordered["answer"])
        self.assertEqual(len(reordered["sources"]), 1)


if __name__ == "__main__":
    unittest.main()

# ИСПРАВЛЕНО: регрессионные проверки RAG без скачивания моделей, внешних API и сетевых запросов.
