# core/error_handler.py
"""
Обработчик ошибок с понятными сообщениями.
Адаптировано для Streamlit-приложений.
"""

from __future__ import annotations

import json
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Awaitable


class ErrorHandler:
    """
    Обработчик ошибок с понятными сообщениями.
    Адаптировано для Streamlit-приложений.
    """

    _VALID_LOG_LEVELS = {"debug", "info", "error"}

    def __init__(self, log_level: str = "info", log_file: str | None = None):
        """
        Инициализация обработчика ошибок.

        Args:
            log_level: Уровень логирования ("info", "debug", "error")
            log_file: Путь к файлу для записи логов (опционально)
        """
        normalized_level = (log_level or "info").strip().lower()
        self.log_level = normalized_level if normalized_level in self._VALID_LOG_LEVELS else "info"
        self.log_file = log_file
        self.errors: list[dict[str, Any]] = []

    def handle(self, error: Exception, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Обрабатывает ошибку и возвращает понятный ответ для пользователя.
        """
        error_info: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {},
        }

        self.errors.append(error_info)
        self._log_error(error_info)
        return self._format_error(error_info)

    def _log_error(self, error_info: dict[str, Any]) -> None:
        """
        Логирует ошибку в консоль и/или файл.
        """
        message = (
            f"[ERROR] {error_info.get('timestamp', '')} "
            f"{error_info.get('type', 'Unknown')}: {error_info.get('message', '')}"
        )

        print(message)

        if self.log_level == "debug":
            trace = str(error_info.get("traceback", "") or "")
            if trace.strip():
                print(trace)

        if self.log_file:
            try:
                log_path = Path(self.log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)

                with log_path.open("a", encoding="utf-8") as file:
                    file.write(message + "\n")
                    if self.log_level == "debug":
                        trace = str(error_info.get("traceback", "") or "")
                        if trace.strip():
                            file.write(trace + "\n")
                    file.write("-" * 80 + "\n")
            except OSError:
                pass

    def _format_error(self, error_info: dict[str, Any]) -> dict[str, Any]:
        """
        Форматирует ошибку для пользователя.
        """
        error_type = str(error_info.get("type", "Exception"))
        message = str(error_info.get("message", ""))

        error_messages = {
            "ModuleNotFoundError": (
                f"⚠️ Отсутствует необходимая библиотека. "
                f"Установите: pip install {self._extract_module_name(message)}"
            ),
            "FileNotFoundError": f"📁 Файл не найден: {message}",
            "KeyError": f"🔑 Отсутствует ключ в данных: {message}",
            "ValueError": f"📊 Ошибка в данных: {message}",
            "ConnectionError": "🌐 Ошибка подключения. Проверьте интернет-соединение.",
            "TimeoutError": "⏰ Превышено время ожидания. Попробуйте позже.",
            "PermissionError": f"🔒 Нет доступа к файлу: {message}",
            "JSONDecodeError": "📄 Ошибка парсинга JSON. Проверьте формат данных.",
            "ZeroDivisionError": "⚠️ Деление на ноль. Проверьте исходные данные.",
            "TypeError": f"⚠️ Неверный тип данных: {message}",
            "AttributeError": f"⚠️ Ошибка доступа к атрибуту: {message}",
            "ImportError": f"⚠️ Ошибка импорта: {message}",
            "RuntimeError": f"⚠️ Ошибка выполнения: {message}",
            "MemoryError": "⚠️ Недостаточно памяти. Попробуйте уменьшить объём данных.",
            "IndexError": "⚠️ Выход за границы списка. Проверьте индексы.",
            "NotImplementedError": "⚠️ Функция ещё не реализована.",
            "StopIteration": "⚠️ Итерация завершена.",
            "OverflowError": "⚠️ Число слишком велико для обработки.",
            "RecursionError": "⚠️ Превышена глубина рекурсии.",
            "KeyboardInterrupt": "⏹️ Операция прервана пользователем.",
            "OSError": f"⚠️ Системная ошибка: {message}",
            "IOError": f"⚠️ Ошибка ввода/вывода: {message}",
            "UnicodeDecodeError": "⚠️ Ошибка декодирования. Проверьте кодировку файла.",
            "UnicodeEncodeError": "⚠️ Ошибка кодирования. Проверьте текстовые данные.",
        }

        message_lower = message.lower()

        if "FormulaEngine" in error_type or "calculation" in message_lower:
            user_message = f"📐 Ошибка расчёта: {message}"
        elif "QASystem" in error_type or "search" in message_lower:
            user_message = f"🔍 Ошибка поиска: {message}"
        elif "AgentLoop" in error_type:
            user_message = f"🤖 Ошибка агента: {message}"
        else:
            user_message = error_messages.get(error_type, f"❌ Ошибка: {message or 'Неизвестная ошибка'}")

        return {
            "user_message": user_message,
            "debug_info": error_info if self.log_level == "debug" else None,
            "is_error": True,
            "type": error_type,
            "message": message,
            "timestamp": str(error_info.get("timestamp", "") or ""),
        }

    @staticmethod
    def _extract_module_name(message: str) -> str:
        """Извлекает имя модуля из сообщения об ошибке."""
        match = re.search(r"'([^']+)'", message)
        return match.group(1) if match else "название-библиотеки"

    def get_last_error(self) -> dict[str, Any] | None:
        """Возвращает последнюю ошибку."""
        return self.errors[-1] if self.errors else None

    def get_all_errors(self) -> list[dict[str, Any]]:
        """Возвращает все ошибки."""
        return list(self.errors)

    def clear_errors(self) -> None:
        """Очищает список ошибок."""
        self.errors.clear()

    @staticmethod
    def format_for_ui(error_info: dict[str, Any]) -> str:
        """Форматирует ошибку для отображения в UI."""
        if error_info.get("is_error"):
            return str(error_info.get("user_message", "❌ Неизвестная ошибка"))
        return f"ℹ️ {error_info.get('message', 'Неизвестная ошибка')}"

    def get_error_summary(self) -> dict[str, Any]:
        """Возвращает сводку по ошибкам."""
        if not self.errors:
            return {"total": 0, "types": {}}

        types: dict[str, int] = {}
        for error in self.errors:
            error_type = str(error.get("type", "Unknown"))
            types[error_type] = types.get(error_type, 0) + 1

        last_error = self.errors[-1] if self.errors else None

        return {
            "total": len(self.errors),
            "types": types,
            "last_error": last_error,
            "last_timestamp": last_error.get("timestamp") if last_error else None,
        }

    @staticmethod
    def log_to_file(error_info: dict[str, Any], file_path: str) -> None:
        """Записывает ошибку в отдельный файл лога в формате JSON."""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with path.open("a", encoding="utf-8") as file:
                json.dump(error_info, file, ensure_ascii=False)
                file.write("\n" + "-" * 80 + "\n")
        except OSError as exc:
            print(f"⚠️ Не удалось записать лог: {exc}")

    def get_user_friendly_message(self, error: Exception) -> str:
        """Быстрый метод для получения понятного сообщения об ошибке."""
        result = self.handle(error)
        return str(result["user_message"])

    @staticmethod
    def is_critical(error_info: dict[str, Any]) -> bool:
        """Определяет, является ли ошибка критической."""
        critical_types = {
            "MemoryError",
            "RecursionError",
            "KeyboardInterrupt",
            "SystemError",
        }
        return str(error_info.get("type")) in critical_types


def safe_execute(
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> tuple[Any | None, Exception | None]:
    """Безопасное выполнение функции с обработкой ошибок."""
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as exc:
        return None, exc


async def safe_execute_async(
    async_func: Callable[..., Awaitable[Any]],
    *args: Any,
    **kwargs: Any,
) -> tuple[Any | None, Exception | None]:
    """Безопасное выполнение асинхронной функции с обработкой ошибок."""
    try:
        result = await async_func(*args, **kwargs)
        return result, None
    except Exception as exc:
        return None, exc