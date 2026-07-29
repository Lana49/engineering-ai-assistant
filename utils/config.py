# -*- coding: utf-8 -*-
"""
Конфигурация проекта.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Hugging Face Dataset
HF_DATASET_REPO_ID = "Lana49/engineering-docs"

# Создаём папки
for dir_path in [RAW_DIR, PROCESSED_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

print(f"✅ config.py загружен")
print(f"📁 Папка документов: {RAW_DIR}")
print(f"📁 Папка индексов: {PROCESSED_DIR}")
print(f"📦 Dataset: {HF_DATASET_REPO_ID}")