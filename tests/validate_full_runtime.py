"""Ручная содержательная проверка сохранённого полного индекса и async API."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import resource
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import config
from core.agent_loop import AgentLoop
from core.formula_engine import FormulaEngine
from core.qa_engine import QASystem
from core.retrieval_memory import RetrievalMemory
from test_real_retrieval import QUESTIONS


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--index", type=Path, default=config.PROCESSED_DIR / "faiss_index.pkl")
    cli.add_argument("--output", type=Path, default=config.PROCESSED_DIR / "full_runtime_validation.json")
    cli.add_argument("--model", default=config.OLLAMA_MODEL)
    cli.add_argument("--no-llm", action="store_true")
    args = cli.parse_args()
    started = time.monotonic()
    qa = QASystem(
        use_embeddings=True, use_llm=not args.no_llm,
        ollama_model=args.model, ollama_base_url=config.OLLAMA_BASE_URL,
        embedding_model_name=config.EMBEDDING_MODEL,
        top_k=config.TOP_K, min_score=config.MIN_SCORE,
        semantic_weight=config.SEMANTIC_WEIGHT, lexical_weight=config.LEXICAL_WEIGHT,
    )
    assert qa.embedding_model is not None
    assert qa.load_index(args.index), qa.last_load_diagnostics
    assert len(qa.documents) == 151 and len(qa.chunks) == 35309
    assert qa.chunk_embeddings is not None and qa.chunk_embeddings.shape == (35309, 384)
    assert all((config.BASE_DIR / doc["filepath"]).is_file() for doc in qa.documents)
    report = {"documents": len(qa.documents), "chunks": len(qa.chunks),
              "embedding_shape": list(qa.chunk_embeddings.shape),
              "load_seconds": round(time.monotonic() - started, 3), "checks": []}
    with tempfile.TemporaryDirectory() as temporary:
        agent = AgentLoop(qa, FormulaEngine(qa))
        agent.memory = RetrievalMemory(Path(temporary) / "retrieval_memory.json")
        for question, document, clause, expected in QUESTIONS:
            tick = time.monotonic()
            result = asyncio.run(agent.run(question))
            assert not result.get("error"), result
            assert not result.get("needs_clarification"), result["answer"]
            assert expected.casefold() in result["answer"].casefold(), result["answer"]
            assert clause in result["answer"], result["answer"]
            assert any(source["doc_name"] == document for source in result["sources"])
            report["checks"].append({"question": question, "seconds": round(time.monotonic() - tick, 3),
                                     "answer": result["answer"], "sources": result["sources"], "passed": True})
        negatives = (
            "Что такое рабочая зона по СП 60.13330? Найди определение.",
            "Что такое квантовая телепортация по СП 60.13330?",
            "Что требует пункт 99.99.99 СП 60.13330.2020?",
            "Что такое вентиляция по СП 60.13330.2099?",
            "Рассчитай тепловой поток трубы: Q=1000 Вт, длина 5 м, расход воздуха 100 м3/ч",
            "Рассчитай диаметр трубы: L=100 м3/ч, tв=20, tн=-10",
            "Рассчитай теплопотери: A=10, R=2, tв=20, tн=-10, delta_t=50",
            "Рассчитай вентиляцию: L=100 м3/ч",
        )
        for question in negatives:
            tick = time.monotonic()
            result = asyncio.run(agent.run(question))
            assert not result.get("error"), result
            assert result.get("needs_clarification"), result["answer"]
            report["checks"].append({"question": question, "seconds": round(time.monotonic() - tick, 3),
                                     "answer": result["answer"], "passed": True})
        result = asyncio.run(agent.run("Рассчитай вентиляцию: L=100 м³/ч, t_в=20, t_н=-25"))
        assert not result.get("needs_clarification") and "1507.5" in result["answer"].replace(",", ".")
        report["calculation"] = result["answer"]
        if not args.no_llm:
            question = "Как обеспечивается удаление загрязнённого воздуха из помещений?"
            tick = time.monotonic()
            result = qa.answer(question)
            assert result["used_llm"], qa.last_llm_error
            answer = result["answer"].casefold()
            assert "зон" in answer and "загрязн" in answer, result["answer"]
            assert any(term in answer for term in ("наиболее", "наибольш", "максималь")), result["answer"]
            report["llm"] = {"question": question, "seconds": round(time.monotonic() - tick, 3),
                             "answer": result["answer"], "sources": result["sources"], "used_llm": True}
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("checks", "llm")}, ensure_ascii=False, indent=2))
    print(f"OK: {len(report['checks'])} проверок, расчёт и загрузка полного индекса")


if __name__ == "__main__":
    main()

# ИСПРАВЛЕНО: проверка полного индекса после перезапуска и содержательных регрессий.
