"""Построение полного индекса из уже прочитанных документов, без повторного OCR.

Запускать из корня проекта. JSON должен быть результатом своего DocumentParser.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import config
from core.qa_engine import QASystem


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--parsed", type=Path, required=True)
    cli.add_argument("--output", type=Path, default=config.PROCESSED_DIR / "faiss_index.pkl")
    cli.add_argument("--batch-size", type=int, default=64)
    args = cli.parse_args()
    started = time.monotonic()
    documents = json.loads(args.parsed.read_text(encoding="utf-8"))
    assert isinstance(documents, list) and documents, "Нет распознанных документов"
    for document in documents:
        # Индекс остаётся переносимым: в нём нет путей к прошлой рабочей сессии.
        name = document["doc_name"]
        assert Path(name).name == name, f"Ожидалось имя файла: {name}"
        assert (config.RAW_DIR / name).is_file(), f"Не найден документ: {name}"
        relative_path = str(Path("data/raw") / name)
        document["filepath"] = relative_path
        for chunk in document.get("chunks", []):
            chunk["filepath"] = relative_path
    qa = QASystem(
        use_llm=False, use_embeddings=True,
        embedding_model_name=config.EMBEDDING_MODEL,
        embedding_batch_size=args.batch_size,
        top_k=config.TOP_K, min_score=config.MIN_SCORE,
        semantic_weight=config.SEMANTIC_WEIGHT, lexical_weight=config.LEXICAL_WEIGHT,
    )
    assert qa.embedding_model is not None, "Модель эмбеддингов недоступна"
    assert qa.build_index(documents), qa.last_index_diagnostics
    assert qa.chunk_embeddings is not None, "Полный семантический индекс не построен"
    assert len(qa.chunks) == sum(len(doc["chunks"]) for doc in documents)
    assert qa.save_index(args.output), qa.last_save_diagnostics
    report = {
        "documents": len(documents), "chunks": len(qa.chunks),
        "embedding_shape": list(qa.chunk_embeddings.shape),
        "tfidf_shape": list(qa.tfidf_matrix.shape),
        "index_bytes": args.output.stat().st_size,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        "document_paths": "relative_to_project_root",
        "diagnostics": qa.last_index_diagnostics,
    }
    report_path = args.output.with_name("full_index_build.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# ИСПРАВЛЕНО: полный индекс из сохранённого чтения; переносимые пути и измерения.
