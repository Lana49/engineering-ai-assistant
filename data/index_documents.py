import sys
from pathlib import Path

# Добавляем текущую папку
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import RAW_DIR

print("=" * 50)
print("🚀 ИНДЕКСАЦИЯ ДОКУМЕНТОВ")
print("=" * 50)

# Ищем файлы
files = list(RAW_DIR.glob("*"))
print(f"\n📁 Папка: {RAW_DIR}")
print(f"📄 Найдено файлов: {len(files)}")

for f in files:
    size = f.stat().st_size
    print(f"  • {f.name} ({size:,} байт)")

if not files:
    print(f"\n⚠️ Положите документы в папку:")
    print(f"   {RAW_DIR}")