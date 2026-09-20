#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
echo "Запуск инженерного помощника"

if [ -x .venv/bin/python ]; then
  ENGINEERING_PYTHON="$PWD/.venv/bin/python"
else
  ENGINEERING_PYTHON="python"
fi

# Читаем dotenv через тот же Python и config.py, что использует приложение.
# Вывод содержит только разрешённые имена; значения не исполняются как shell-код.
ENGINEERING_STARTUP_ENV="$("${ENGINEERING_PYTHON}" core/config.py --startup-env)"
while IFS= read -r ENGINEERING_SETTING; do
  case "${ENGINEERING_SETTING%%=*}" in
    USE_LLM|LLM_PROVIDER|OLLAMA_BASE_URL|OLLAMA_MODEL|STREAMLIT_SERVER_PORT)
      export "${ENGINEERING_SETTING}"
      ;;
    *)
      echo "Некорректная настройка запуска" >&2
      exit 1
      ;;
  esac
done <<< "${ENGINEERING_STARTUP_ENV}"
unset ENGINEERING_STARTUP_ENV ENGINEERING_SETTING

# Ошибка модели не должна выключать поиск и детерминированный калькулятор.
if [ "${USE_LLM:-true}" = "true" ] && [ "${LLM_PROVIDER:-ollama}" != "none" ]; then
  OLLAMA_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
  if ! curl -fsS --max-time 2 "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
    case "${OLLAMA_URL}" in
      http://127.0.0.1:11434|http://localhost:11434)
        if command -v ollama >/dev/null 2>&1; then
          ollama serve > "${TMPDIR:-/tmp}/engineering-ollama.log" 2>&1 &
          for attempt in $(seq 1 30); do
            if curl -fsS --max-time 2 "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
              break
            fi
            sleep 1
          done
        fi
        ;;
    esac
  fi
  if curl -fsS --max-time 2 "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
    if command -v ollama >/dev/null 2>&1; then
      ENGINEERING_MODEL="${OLLAMA_MODEL:-phi3:mini}"
      if ! OLLAMA_HOST="${OLLAMA_URL}" ollama show "${ENGINEERING_MODEL}" >/dev/null 2>&1; then
        echo "Загрузка модели ${ENGINEERING_MODEL}..."
        if ! OLLAMA_HOST="${OLLAMA_URL}" ollama pull "${ENGINEERING_MODEL}"; then
          echo "Модель не скачана. Доступны поиск, выдержки из документов и калькулятор."
        fi
      fi
    fi
  else
    echo "Ollama недоступна. Доступны поиск, выдержки из документов и калькулятор."
  fi
fi

exec "${ENGINEERING_PYTHON}" -m streamlit run app.py --server.port="${STREAMLIT_SERVER_PORT:-7860}" --server.address="0.0.0.0"

# ИСПРАВЛЕНО: единая конфигурация и Python из .venv; сбой Ollama не останавливает UI.
