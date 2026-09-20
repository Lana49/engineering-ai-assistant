"""Регрессии запуска Streamlit, малого индекса и экспорта русского ответа."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import app
from core.qa_engine import QASystem


class AppRegressionTests(unittest.TestCase):
    def test_unified_query_route_and_sources(self):
        app._run_self_tests()

    def test_optional_downloader_is_not_called_when_missing(self):
        with patch.object(app, "snapshot_download", None):
            self.assertFalse(app.sync_hf_dataset_to_raw())

    def test_partial_dataset_is_resumed_and_force_flag_is_forwarded(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            (raw / "already.rtf").write_text("test", encoding="ascii")
            with patch.object(app, "RAW_DIR", raw), patch.object(app, "snapshot_download") as downloader:
                self.assertTrue(app.sync_hf_dataset_to_raw())
                self.assertEqual(downloader.call_count, 1)
                self.assertFalse(downloader.call_args.kwargs["force_download"])
                self.assertTrue(app.sync_hf_dataset_to_raw(force=True))
                self.assertTrue(downloader.call_args.kwargs["force_download"])

    def test_history_write_failure_keeps_answer(self):
        with patch.object(app.st, "session_state", SimpleNamespace(messages=[{"content": "Ответ"}])), patch.object(Path, "open", side_effect=PermissionError("read-only")), patch.object(app.logger, "exception") as log:
            app.save_history()
            self.assertTrue(log.called)

    def test_table_renders_cells_and_settings_handle_bad_values(self):
        with patch.object(app.st, "markdown"), patch.object(app.st, "dataframe") as render:
            app._render_table({"title": "Климат", "headers": ["Город", "t"], "rows": [["Томск", "-25"]]})
            frame = render.call_args.args[0]
            self.assertEqual(frame.iloc[0, 0], "Томск")
            self.assertEqual(frame.iloc[0, 1], "-25")
        app.settings._run_self_tests()

    def test_valid_small_index_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            document = raw / "sample.rtf"
            document.write_text(r"{\rtf1 Test}", encoding="ascii")
            docs = [{"name": "sample.rtf", "filename": "sample.rtf", "chunks": [
                {"text": "Вентиляция обеспечивает воздухообмен в помещении."}
            ]}]
            qa = QASystem(use_llm=False, use_embeddings=False)
            with patch.multiple(app, RAW_DIR=raw, PROCESSED_DIR=root, INDEX_FILE=root / "index.pkl"), patch.object(app, "parse_directory", return_value=docs):
                self.assertTrue(app.force_rebuild_index(qa))
                self.assertLess((root / "index.pkl").stat().st_size, 1_000_000)

    def test_pdf_keeps_cyrillic_and_escapes_text(self):
        import pymupdf
        with tempfile.TemporaryDirectory() as temporary, patch.object(app, "PROCESSED_DIR", Path(temporary)):
            result = app.export_to_pdf("Теплопотери < 500 Вт & сопротивление > 2", [{"doc_name": "Тест & проверка"}])
            self.assertIsNotNone(result)
            with pymupdf.open(result) as document:
                text = "".join(page.get_text() for page in document)
            self.assertIn("Теплопотери < 500", text)
            self.assertIn("Тест & проверка", text)

    def test_streamlit_question_and_calculation(self):
        from streamlit.testing.v1 import AppTest
        # Настоящий Streamlit исполняет интерфейс, а QA получает маленький локальный корпус.
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {
            "USE_LLM": "false", "USE_EMBEDDINGS": "false", "AUTO_SYNC_DATASET": "false", "AUTO_REBUILD_INDEX": "false",
        }):
            script = f'''
from pathlib import Path
import app
from core.qa_engine import QASystem
app.HISTORY_FILE = Path({str(Path(temporary) / "history.json")!r})
app.INDEX_FILE = Path({str(Path(temporary) / "index.pkl")!r})
app.RAW_DIR = Path({temporary!r})
def create_qa():
    qa = QASystem(use_llm=False, use_embeddings=False, min_score=0.05)
    qa.build_index([{{"name": "Тестовый СП", "chunks": [{{"text": "Вентиляция — организованный обмен воздуха в помещении. Система вентиляции удаляет загрязнённый воздух и подаёт наружный воздух."}}]}}])
    return qa
app.get_cached_qa_system = create_qa
app.main()
'''
            test_app = AppTest.from_string(script, default_timeout=30).run()
            self.assertEqual(len(test_app.exception), 0, str(test_app.exception))
            test_app.chat_input[0].set_value("Что такое вентиляция?").run()
            self.assertEqual(len(test_app.exception), 0, str(test_app.exception))
            response = test_app.session_state["messages"][-1]
            self.assertTrue(response["sources"])
            self.assertRegex(response["content"], r"\[(?:Источник )?1\]")
            test_app.chat_input[0].set_value("Рассчитай вентиляцию: L=100 м³/ч, t_в=20, t_н=-25").run()
            self.assertEqual(len(test_app.exception), 0, str(test_app.exception))
            answer = test_app.session_state["messages"][-1]["content"]
            self.assertTrue("1507" in answer or "1 507" in answer, answer)


if __name__ == "__main__":
    unittest.main()

# ИСПРАВЛЕНО: тесты единого маршрута, None downloader, малого индекса, русского PDF и живого исполнения Streamlit.
