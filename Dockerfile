# ============================================
# Dockerfile для Engineering AI Assistant
# Hugging Face Docker Space
# ============================================

FROM python:3.10-slim

# ============================================
# НАСТРОЙКИ ОКРУЖЕНИЯ
# ============================================
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_ORIGINS=* \
    OLLAMA_MODELS=/app/data/models \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HOME=/app/.cache/huggingface \
    USE_LLM=true \
    LLM_PROVIDER=ollama \
    OLLAMA_MODEL=phi3:mini \
    AUTO_SYNC_DATASET=true \
    AUTO_REBUILD_INDEX=true \
    PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# ============================================
# СИСТЕМНЫЕ ЗАВИСИМОСТИ
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    ca-certificates \
    procps \
    build-essential \
    libmagic1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    fonts-dejavu-core \
    zstd \
    antiword \
    catdoc \
    poppler-utils \
    unrtf \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# УСТАНОВКА OLLAMA
# ============================================
RUN curl -fsSL https://ollama.com/install.sh | sh

# ============================================
# СОЗДАНИЕ ДИРЕКТОРИЙ
# ============================================
RUN mkdir -p /app/data/models /app/data/raw /app/data/processed /app/.cache/huggingface && \
    chmod -R 777 /app/data /app/.cache

# ============================================
# УСТАНОВКА PYTHON ЗАВИСИМОСТЕЙ
# ============================================
COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r requirements.txt

# ============================================
# ПРЕДВАРИТЕЛЬНАЯ ЗАГРУЗКА EMBEDDING МОДЕЛИ
# ============================================
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')" || true

# ============================================
# КОПИРОВАНИЕ КОДА
# ============================================
COPY . .

RUN chmod +x /app/start.sh

# ============================================
# ТОМА ДЛЯ ДАННЫХ
# ============================================
VOLUME ["/app/data"]

# ============================================
# ПОРТЫ
# ============================================
EXPOSE 7860
EXPOSE 11434

# ============================================
# HEALTHCHECK
# ============================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
  CMD curl --fail http://127.0.0.1:7860/_stcore/health || exit 1

# ============================================
# ЗАПУСК
# ============================================
CMD ["/app/start.sh"]

# ИСПРАВЛЕНО: кириллический шрифт для экспорта отчётов; сохранён запуск Ollama и загрузка базы во время работы.
