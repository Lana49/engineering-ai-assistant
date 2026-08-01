# -*- coding: utf-8 -*-
"""
Конфигурация проекта.
Поддерживает локальный запуск и Hugging Face Space.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Заглушка, если python-dotenv не установлен
    def load_dotenv() -> bool:
        """Пустая заглушка для load_dotenv."""
        return False

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

HF_DATASET_REPO_ID = os.getenv("HF_DATASET_REPO_ID", "Lana49/engineering-docs").strip()

IS_HF_SPACE = bool(
    os.getenv("SPACE_ID")
    or os.getenv("HF_SPACE_ID")
    or os.getenv("SYSTEM") == "spaces"
)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b").strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()

TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.15"))

USE_EMBEDDINGS = os.getenv("USE_EMBEDDINGS", "true").lower() == "true"
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.7"))
LEXICAL_WEIGHT = float(os.getenv("LEXICAL_WEIGHT", "0.3"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "120"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").strip()

print("✅ config.py загружен")
print(f"📁 Папка документов: {RAW_DIR}")
print(f"📁 Папка индексов: {PROCESSED_DIR}")
print(f"📦 Dataset: {HF_DATASET_REPO_ID}")
print(f"🚀 Режим Space: {'ON' if IS_HF_SPACE else 'OFF'}")