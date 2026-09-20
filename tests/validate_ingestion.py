"""Ручная проверка каждого файла корпуса; отчёт не подменяет проверку OCR/таблиц человеком."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.parser import DocumentParser


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = DocumentParser()
    files = sorted(p for p in (root / "data/raw").rglob("*") if p.suffix.lower() in {".pdf", ".docx", ".rtf", ".doc"})
    documents = []
    rows = []
    started = time.monotonic()
    for index, path in enumerate(files, 1):
        before = time.monotonic()
        try:
            parsed = parser.parse_file(path)
            text = parsed.get("text", "")
            chunks = parsed.get("chunks", [])
            error = "" if text and chunks else str(parsed.get("error") or "empty_text_or_chunks")
            if not error:
                documents.append(parsed)
            rows.append({"file": path.name, "format": path.suffix.lower(), "bytes": path.stat().st_size, "characters": len(text), "chunks": len(chunks), "error": error, "seconds": round(time.monotonic() - before, 2)})
        except Exception as exc:
            rows.append({"file": path.name, "format": path.suffix.lower(), "error": f"{type(exc).__name__}: {exc}", "seconds": round(time.monotonic() - before, 2)})
        print(f"CHECK {index}/{len(files)} {path.name}: {rows[-1].get('characters', 0)} chars {rows[-1]['error']}", flush=True)
    report = {"files": len(files), "read_nonempty": len(documents), "failed": len(files) - len(documents), "formats": dict(Counter(row["format"] for row in rows)), "chunks": sum(len(doc["chunks"]) for doc in documents), "elapsed_seconds": round(time.monotonic() - started, 2), "details": rows}
    output = root / "data/processed"
    output.mkdir(parents=True, exist_ok=True)
    (output / "ingestion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "parsed_documents.json").write_text(json.dumps(documents, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, ensure_ascii=False, indent=2))
    assert files, "Сначала скачайте базу"
    assert not report["failed"], "Есть непрочитанные файлы, см. ingestion_report.json"


if __name__ == "__main__":
    main()

# ИСПРАВЛЕНО: проверка всех четырёх форматов на реальном корпусе с пофайловым отчётом и кэшем извлечённого текста.
