
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Создаем папки
for dir_path in [RAW_DIR, PROCESSED_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

print(f"✅ config.py загружен")
print(f"📁 Папка документов: {RAW_DIR}")