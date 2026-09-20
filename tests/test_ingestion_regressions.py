"""Проверки новых регрессий чтения, климатических таблиц и памяти."""
import unittest
from unittest.mock import Mock, patch

from core import parser, table_calculator, table_extractor, retrieval_memory, error_handler


class IngestionRegressionTests(unittest.TestCase):
    def test_pdf_locations_rtf_objects_and_chunk_boundaries(self):
        parser._run_self_tests()

    def test_markdown_cells_table_metadata_and_city_boundaries(self):
        table_extractor._run_self_tests()

    def test_numeric_climate_validation(self):
        table_calculator._run_self_tests()

    def test_atomic_memory_and_corrupt_records(self):
        retrieval_memory._run_self_tests()

    def test_errors_are_logged_outside_except_as_well(self):
        error_handler._run_self_tests()

    def test_one_row_climate_table_is_not_discarded(self):
        qa = Mock()
        qa.search.return_value = [{
            "doc_name": "Контрольная таблица (не норматив)", "score": 1.0,
            "metadata": {"page": 4},
            "text": "[ТАБЛИЦА: Климат]\n| Город | t_от | z_от |\n| --- | --- | --- |\n| Томск | -8,4 | 225 |",
        }]
        with patch.dict("os.environ", {"ALLOW_BUILTIN_CLIMATE_FALLBACK": "false"}):
            response = table_calculator.TableCalculator(qa).calculate_gsop_from_table("Томск", 20)
        self.assertFalse(response["needs_clarification"])
        self.assertEqual(response["result"], 6390.0)
        self.assertEqual(response["sources"][0]["metadata"]["page"], 4)


if __name__ == "__main__":
    unittest.main()

# ИСПРАВЛЕНО: регрессии новых обработчиков документов/таблиц/памяти подключены к общей команде unittest.
