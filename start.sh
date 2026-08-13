#!/usr/bin/env bash
set -e

echo "🚀 ЗАПУСК ИНЖЕНЕРНОГО ПОМОЩНИКА"
echo "USE_LLM: ${USE_LLM:-false}"
echo "STREAMLIT_PORT: ${STREAMLIT_SERVER_PORT:-7860}"

# Очистка .lock файлов
find /tmp -name "*.lock" -type f -delete 2>/dev/null || true
find /app -name "*.lock" -type f -delete 2>/dev/null || true
echo "✅ Очистка .lock файлов завершена"

# Запуск Ollama только если USE_LLM=true
if [ "${USE_LLM:-false}" = "true" ]; then
  echo "==> Запускаем Ollama..."
  ollama serve > /tmp/ollama.log 2>&1 &

  echo "==> Ожидаем запуска Ollama API..."
  for i in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
      echo "✅ Ollama готов (${i} сек.)"
      break
    fi
    sleep 1
    echo -n "."
  done
  echo

  MODEL="${OLLAMA_MODEL:-phi3:mini}"
  echo "==> Проверяем модель ${MODEL}..."
  if ! ollama list | grep -q "${MODEL}"; then
    echo "==> Скачиваем модель ${MODEL}..."
    ollama pull "${MODEL}"
  else
    echo "✅ Модель ${MODEL} уже есть"
  fi
else
  echo "⏭️ Ollama отключён (USE_LLM=false)"
fi

echo "==> Запускаем Streamlit на порту ${STREAMLIT_SERVER_PORT:-7860}..."
exec streamlit run app.py --server.port="${STREAMLIT_SERVER_PORT:-7860}" --server.address="0.0.0.0"