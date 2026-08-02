# ============================================
# Dockerfile для Engineering AI Assistant
# С поддержкой Ollama и постоянным хранилищем
# ============================================

FROM python:3.10-slim

# НАСТРОЙКИ ОКРУЖЕНИЯ
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    \
    # Ollama
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_ORIGINS=* \
    OLLAMA_MODELS=/app/data/models \
    \
    # Streamlit
    STREAMLIT_SERVER_PORT=7860\
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    \
    # Hugging Face
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_DATASET_REPO_ID=Lana49/engineering-docs

# УСТАНОВКА СИСТЕМНЫХ ЗАВИСИМОСТЕЙ
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
    && rm -rf /var/lib/apt/lists/*

# УСТАНОВКА OLLAMA
RUN curl -fsSL https://ollama.com/install.sh | sh

# Создаём директории для данных
RUN mkdir -p /app/data/{models,raw,processed}

# УСТАНОВКА PYTHON ЗАВИСИМОСТЕЙ
WORKDIR /app

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

# КОПИРУЕМ ИСХОДНЫЙ КОД
COPY . .

# Даём права на выполнение скриптов
RUN chmod +x /app/start.sh

# НАСТРОЙКА ТОМОВ (для постоянного хранилища)
VOLUME ["/app/data"]

# ОТКРЫВАЕМ ПОРТЫ
EXPOSE 7860
EXPOSE 11434

# ЗАПУСК ПРИЛОЖЕНИЯ
CMD ["/app/start.sh"]