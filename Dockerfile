# ============================================
# Dockerfile для Engineering AI Assistant
# Порт: 7860 (Hugging Face Docker Space)
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
    HF_HUB_ENABLE_HF_TRANSFER=1

# ============================================
# УСТАНОВКА СИСТЕМНЫХ ЗАВИСИМОСТЕЙ
# ============================================
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    ca-certificates \
    procps \
    build-essential \
    libmagic1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    zstd \
    # ========== УТИЛИТЫ ДЛЯ ПАРСИНГА ДОКУМЕНТОВ ==========
    antiword \
    catdoc \
    poppler-utils \
    unrtf \
    # ========== OCR ДЛЯ СКАНИРОВАННЫХ PDF ==========
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# УСТАНОВКА OLLAMA
# ============================================
RUN curl -fsSL https://ollama.com/install.sh | sh

# Создаём директории для данных
RUN mkdir -p /app/data/{models,raw,processed}

# ============================================
# УСТАНОВКА PYTHON ЗАВИСИМОСТЕЙ
# ============================================
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

# ============================================
# КОПИРУЕМ ИСХОДНЫЙ КОД
# ============================================
COPY . .

RUN chmod +x /app/start.sh

# ============================================
# НАСТРОЙКА ТОМОВ
# ============================================
VOLUME ["/app/data"]

# ============================================
# ОТКРЫВАЕМ ПОРТЫ
# ============================================
EXPOSE 7860
EXPOSE 11434

# ============================================
# ЗАПУСК
# ============================================
CMD ["/app/start.sh"]