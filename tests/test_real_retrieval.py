"""Интеграционные проверки на реальных файлах корпуса (без Ollama и эмбеддингов).

Тесты пропускаются, если исходные документы Hugging Face ещё не скачаны.
"""

import contextlib
import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from core.agent_loop import AgentLoop
from core.parser import DocumentParser
from core.qa_engine import QASystem
from core.retrieval_memory import RetrievalMemory


RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FILENAMES = (
    "СП 60.13330.2020.docx", "СП 50.13330.2024.docx",
    "СП 131.13330.2025.docx", "СП 73.13330.2016.docx",
    "СП 74.13330.2023.RTF", "СП 40.13330.2012.doc", "СП 132.13330.2011.pdf",
)
QUESTIONS = (
    ("Что такое рециркуляция воздуха по СП 60.13330.2020?", "СП 60.13330.2020.docx", "3.1.21", "Смешение воздуха"),
    ("Какую температуру поверхности панелей радиационного обогрева рабочих мест допускает СП 60.13330.2020?", "СП 60.13330.2020.docx", "6.4.14", "60 °C"),
    ("Для чего следует предусматривать воду технического качества по СП 60.13330.2020?", "СП 60.13330.2020.docx", "12.2", "мокрых пылеуловителей"),
    ("Какое минимальное расстояние до горючих конструкций требуется для трубопроводов с теплоносителем выше 100 °C по СП 60.13330.2020?", "СП 60.13330.2020.docx", "14.9", "100 мм"),
    ("Что требует пункт 7.2.2 СП 73.13330.2016 при гидростатическом испытании системы водоснабжения?", "СП 73.13330.2016.docx", "7.2.2", "0,05 МПа"),
)


@unittest.skipUnless(all((RAW / name).is_file() for name in FILENAMES), "Скачайте документы корпуса для интеграционных тестов")
class RealRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            parser = DocumentParser(chunk_size=1200, chunk_overlap=200)
            cls.documents = [parser.parse_file(RAW / name) for name in FILENAMES]
            cls.qa = QASystem(use_llm=False, use_embeddings=False, top_k=5)
            if not cls.qa.build_index(cls.documents):
                raise AssertionError("Не удалось построить индекс реальных документов")

    def test_all_selected_formats_have_chunks(self):
        for document in self.documents:
            with self.subTest(document=document["doc_name"]):
                self.assertTrue(document["chunks"], document["metadata"])

    def test_five_ground_truth_questions(self):
        for question, document, clause, expected in QUESTIONS:
            with self.subTest(question=question):
                results = self.qa.search(question)
                self.assertTrue(results)
                self.assertEqual(results[0].doc_name, document)
                self.assertTrue(any(clause in item.text and expected in item.text for item in results))
                response = self.qa.answer(question)
                self.assertIn(expected, response["answer"])
                self.assertIn("[Источник ", response["answer"])

    def test_explicit_clause_selects_requested_document(self):
        results = self.qa.search("Что требует пункт 7.2.2 СП 73.13330.2016?")
        self.assertTrue(results)
        self.assertEqual(results[0].doc_name, "СП 73.13330.2016.docx")
        self.assertIn("7.2.2 Система считается", results[0].text)

    def test_agent_expansion_keeps_original_document_and_clause(self):
        formula = Mock()
        formula.can_calculate_directly.return_value = False
        with tempfile.TemporaryDirectory() as directory:
            agent = AgentLoop(self.qa, formula)
            agent.memory = RetrievalMemory(Path(directory) / "memory.json")
            for question, document, _, expected in (QUESTIONS[0], QUESTIONS[-1]):
                with self.subTest(question=question):
                    response = asyncio.run(agent.run(question))
                    self.assertIn(expected, response["answer"])
                    self.assertTrue(response["sources"])
                    self.assertTrue(all(source["doc_name"] == document for source in response["sources"]))


def probe() -> None:
    """Печатает выдачу, чтобы вручную проверить источник, пункт и выдержку."""
    RealRetrievalTests.setUpClass()
    qa = RealRetrievalTests.qa
    print("CHUNKS", len(qa.chunks))
    for question, _, _, _ in QUESTIONS:
        print("\nQUESTION", question)
        for result in qa.search(question):
            print("RESULT", result.doc_name, result.chunk_id, round(result.score, 3), result.text[:260].replace("\n", " "))
        print("ANSWER", qa.answer(question)["answer"])


if __name__ == "__main__":
    import sys
    if "--probe" in sys.argv:
        probe()
    else:
        unittest.main()

# ИСПРАВЛЕНО: интеграционные проверки пяти вопросов с эталонными пунктами из реального корпуса, фильтра документов и чтения DOCX/RTF/DOC/PDF.
