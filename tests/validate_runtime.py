"""Ручная интеграционная проверка настоящей модели на файле из базы (не unit-тест)."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.agent_loop import AgentLoop
from core.formula_engine import FormulaEngine
from core.parser import DocumentParser
from core.qa_engine import QASystem


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--document", type=Path, required=True)
    cli.add_argument("--model", default="llama3.1:8b")
    cli.add_argument("--no-llm", action="store_true")
    cli.add_argument("--output", type=Path, default=Path("data/processed/runtime_validation.json"))
    args = cli.parse_args()
    started = time.monotonic()
    document = DocumentParser().parse_file(args.document)
    assert document and document.get("chunks"), "Документ не прочитан"
    qa = QASystem(use_llm=not args.no_llm, use_embeddings=True, ollama_model=args.model, top_k=3)
    assert qa.embedding_model is not None, "Настоящая модель эмбеддингов не загружена"
    assert qa.build_index([document])
    assert qa.chunk_embeddings is not None, "Не построены настоящие эмбеддинги"
    question = "Как обеспечивается удаление загрязнённого воздуха из помещений?"
    results = qa.search(question, top_k=3)
    assert results and any(item.semantic_score > 0 for item in results)
    assert any("7.3.16" in item.text and "наиболее загрязнен" in item.text for item in results), "Не найдено прямое правило удаления воздуха"
    # Этот пример должен воздержаться от определения, даже когда LLM доступна.
    negative = qa.answer("Что такое рабочая зона по СП 60.13330? Найди определение.")
    assert negative["needs_clarification"] and not negative["used_llm"], negative["answer"]
    assert not negative["sources"], "Косвенные упоминания представлены как доказательство определения"
    definition = qa.answer("Что такое рециркуляция воздуха по СП 60.13330.2020?")
    assert definition["evidence_mode"] == "exact_excerpt", definition["answer"]
    assert "Смешение воздуха" in definition["answer"] and "[Источник 1]" in definition["answer"]
    norm = qa.answer("Какую температуру поверхности панелей радиационного обогрева рабочих мест допускает СП 60.13330.2020?")
    assert norm["evidence_mode"] == "exact_excerpt" and "60 °C" in norm["answer"], norm["answer"]
    # Вызов Ollama проверяется отдельно на обычном объяснении с прямой опорой.
    answer = qa.answer_from_results(question, results)
    assert answer["sources"] and not answer["needs_clarification"], answer["answer"]
    lower = answer["answer"].lower().replace("ё", "е")
    assert "зон" in lower and "загрязн" in lower and re.search(r"наиболее|наибольш|максималь", lower), answer["answer"]
    assert qa._has_valid_citations(answer["answer"], len(answer["sources"]))
    # Числа сами по себе не доказывают истинность, но новых числовых норм быть не должно.
    factual_answer = re.sub(r"\[Источник\s+\d+\]|(?m:^\s*\d+[.)]\s+)", "", answer["answer"])
    numbers = lambda value: {number.replace(",", ".") for number in re.findall(r"\d+(?:[.,]\d+)*", value)}
    assert numbers(factual_answer) <= numbers(answer["context"]), "Ответ добавил неподтверждённые числа"
    engine = FormulaEngine(qa)
    agent = AgentLoop(qa, engine)
    calculation = asyncio.run(agent.run("Рассчитай вентиляцию: L=100 м³/ч, t_в=20, t_н=-25"))
    assert not calculation.get("needs_clarification"), calculation["answer"]
    assert re.search(r"1\s?507[,.]5", calculation["answer"]), calculation["answer"]
    if not args.no_llm:
        assert answer["used_llm"], f"Модель не сформировала проверяемый ответ: {qa.last_llm_error}"
    report = {
        "document": args.document.name,
        "chunks": len(qa.chunks),
        "embedding_shape": list(qa.chunk_embeddings.shape),
        "retrieval": [{"source": item.doc_name, "score": item.score, "semantic_score": item.semantic_score, "excerpt": item.text[:350]} for item in results],
        "model": args.model,
        "used_llm": answer["used_llm"],
        "llm_error": qa.last_llm_error,
        "answer": answer["answer"],
        "sources": answer["sources"],
        "definition_rejection": negative,
        "supported_definition": definition,
        "supported_norm": norm,
        "content_checks_passed": True,
        "calculation": calculation["answer"],
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# ИСПРАВЛЕНО: воспроизводимая проверка реального чтения, эмбеддингов, Ollama и async-калькулятора.
