# -*- coding: utf-8 -*-
"""
Конфигурация проекта.
Поддерживает локальный запуск и Hugging Face Space.
"""

from __future__ import annotations
import logging
import math
import os
import sys
from pathlib import Path
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env", override=False)
    load_dotenv(BASE_DIR / "op.env", override=False)
except ImportError:
    logger.info("python-dotenv не установлен: используются переменные окружения")

def _project_path(name: str, default: str | Path) -> Path:
    """Относительные пути в настройках отсчитываются от проекта, не от cwd."""
    path = Path(os.getenv(name, "").strip() or default).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


DATA_DIR = _project_path("ENGINEERING_DATA_DIR", "data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Подхватываем перенесённый локальный кэш, сохраняя обычный HF-кэш в Docker.
# Явная настройка всегда приоритетнее автоматически найденной папки.
EMBEDDINGS_CACHE_DIR = _project_path("SENTENCE_TRANSFORMERS_HOME", DATA_DIR / "embeddings")
if os.getenv("SENTENCE_TRANSFORMERS_HOME", "").strip() or EMBEDDINGS_CACHE_DIR.is_dir():
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(EMBEDDINGS_CACHE_DIR)

HF_DATASET_REPO_ID = os.getenv("HF_DATASET_REPO_ID", "Lana49/engineering-docs").strip()

IS_HF_SPACE = bool(
    os.getenv("SPACE_ID")
    or os.getenv("HF_SPACE_ID")
    or os.getenv("SYSTEM") == "spaces"
)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
if LLM_PROVIDER not in {"ollama", "none"}:
    logger.warning("Провайдер %s не поддерживается; выбран Ollama", LLM_PROVIDER)
    LLM_PROVIDER = "ollama"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini").strip()


def _env_number(name: str, default: float, minimum: float = 0.0) -> float:
    """Ошибка в настройке не должна мешать запуску приложения."""
    try:
        value = float(os.getenv(name, str(default)).replace(",", "."))
        if not math.isfinite(value) or value < minimum:
            raise ValueError("значение вне диапазона")
        return value
    except ValueError:
        logger.warning("Некорректная настройка %s; используется %s", name, default)
        return default


TOP_K = int(_env_number("TOP_K", 5, 1))
MIN_SCORE = min(1.0, _env_number("MIN_SCORE", 0.15))

USE_EMBEDDINGS = os.getenv("USE_EMBEDDINGS", "true").lower() == "true"
SEMANTIC_WEIGHT = _env_number("SEMANTIC_WEIGHT", 0.7)
LEXICAL_WEIGHT = _env_number("LEXICAL_WEIGHT", 0.3)

CHUNK_SIZE = int(_env_number("CHUNK_SIZE", 1200, 300))
CHUNK_OVERLAP = min(CHUNK_SIZE - 1, int(_env_number("CHUNK_OVERLAP", 200)))
MIN_CHUNK_SIZE = min(CHUNK_SIZE, int(_env_number("MIN_CHUNK_SIZE", 120, 1)))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").strip()

def _startup_environment() -> str:
    """Только нужные start.sh настройки; однострочные значения читаются как данные."""
    values = {
        "USE_LLM": os.getenv("USE_LLM", "true").lower(),
        "LLM_PROVIDER": LLM_PROVIDER,
        "OLLAMA_BASE_URL": OLLAMA_BASE_URL,
        "OLLAMA_MODEL": OLLAMA_MODEL,
        "STREAMLIT_SERVER_PORT": os.getenv("STREAMLIT_SERVER_PORT", "7860"),
    }
    for name, value in values.items():
        if any(character in value for character in "\r\n\x00"):
            raise ValueError(f"Настройка {name} должна занимать одну строку")
    return "\n".join(f"{name}={value}" for name, value in values.items())


if not (__name__ == "__main__" and sys.argv[1:] == ["--startup-env"]):
    print("✅ config.py загружен")
    print(f"📁 Папка документов: {RAW_DIR}")
    print(f"📁 Папка индексов: {PROCESSED_DIR}")
    print(f"📦 Dataset: {HF_DATASET_REPO_ID}")
    print(f"🚀 Режим Space: {'ON' if IS_HF_SPACE else 'OFF'}")


def _run_self_tests() -> None:
    from unittest.mock import patch
    with patch.dict(os.environ, {"TEST_SETTING": "not-a-number"}):
        assert _env_number("TEST_SETTING", 5) == 5
    with patch.dict(os.environ, {"TEST_SETTING": "NaN"}):
        assert _env_number("TEST_SETTING", 5) == 5
    with patch.dict(os.environ, {"TEST_SETTING": "0,7"}):
        assert _env_number("TEST_SETTING", 5) == 0.7
    assert 0 <= CHUNK_OVERLAP < CHUNK_SIZE


if __name__ == "__main__":
    if sys.argv[1:] == ["--startup-env"]:
        print(_startup_environment())
    else:
        _run_self_tests()

# ИСПРАВЛЕНО: единые .env/op.env/environment для shell и Python; переносимые пути данных/кэша.
