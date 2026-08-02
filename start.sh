#!/usr/bin/env bash
set -euo pipefail

OLLAMA_URL="http://localhost:11434"
OLLAMA_MODEL="${OLLAMA_MODEL:-phi3:mini}"
OLLAMA_START_TIMEOUT="${OLLAMA_START_TIMEOUT:-120}"
STREAMLIT_PORT="${STREAMLIT_SERVER_PORT:-7860}"

echo "🚀 ЗАПУСК ИНЖЕНЕРНОГО ПОМОЩНИКА"
echo "OLLAMA_MODEL: ${OLLAMA_MODEL}"
echo "STREAMLIT_PORT: ${STREAMLIT_PORT}"

echo "==> Очищаем .lock файлы..."
find /app/data/raw -name "*.lock" -type f -delete 2>/dev/null || true
echo "✅ Очистка .lock файлов завершена"

echo "==> Запускаем Ollama..."
ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!

# Функция для корректного завершения
cleanup() {
  echo "==> Останавливаем сервисы..."
  if kill -0 "${OLLAMA_PID}" 2>/dev/null; then
    kill "${OLLAMA_PID}" || true
  fi
}
trap cleanup EXIT

echo "==> Ожидаем запуска Ollama API..."
started="false"
for i in $(seq 1 "${OLLAMA_START_TIMEOUT}"); do
  if curl -sf "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
    started="true"
    echo "✅ Ollama готов (${i} сек.)"
    break
  fi
  echo -n "."
  sleep 1
done

if [ "${started}" != "true" ]; then
  echo ""
  echo "❌ ОШИБКА: Ollama не запустился за ${OLLAMA_START_TIMEOUT} секунд"
  echo "---- /tmp/ollama.log ----"
  cat /tmp/ollama.log || true
  exit 1
fi

echo ""
echo "==> Проверяем модель ${OLLAMA_MODEL}..."
if ! ollama list | grep -q "^${OLLAMA_MODEL} "; then
  echo "==> Скачиваем модель ${OLLAMA_MODEL} (это может занять время)..."
  ollama pull "${OLLAMA_MODEL}"
  echo "✅ Модель ${OLLAMA_MODEL} загружена"
else
  echo "✅ Модель ${OLLAMA_MODEL} уже существует"
fi

echo ""
echo "==> Запускаем Streamlit на порту ${STREAMLIT_PORT}..."
exec streamlit run app.py \
  --server.port="${STREAMLIT_PORT}" \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false