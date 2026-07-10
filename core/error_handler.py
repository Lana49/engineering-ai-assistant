import traceback
from typing import Optional, Dict, Any, List
from datetime import datetime


class ErrorHandler:
    """
    Обработчик ошибок с понятными сообщениями.
    Адаптировано для Streamlit-приложений.
    """

    def __init__(self, log_level: str = "info", log_file: Optional[str] = None):
        """
        Инициализация обработчика ошибок.

        Args:
            log_level: Уровень логирования ("info", "debug", "error")
            log_file: Путь к файлу для записи логов (опционально)
        """
        self.log_level = log_level
        self.log_file = log_file
        self.errors: List[Dict[str, Any]] = []

    def handle(self, error: Exception, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Обрабатывает ошибку и возвращает понятный ответ для пользователя.

        Args:
            error: Исключение
            context: Дополнительный контекст (опционально)

        Returns:
            Dict с полями:
            - user_message: понятное сообщение для пользователя
            - debug_info: детальная информация (если log_level == "debug")
            - is_error: всегда True
            - type: тип ошибки
            - message: исходное сообщение ошибки
        """
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }

        self.errors.append(error_info)
        self._log_error(error_info)

        return self._format_error(error_info)

    def _log_error(self, error_info: Dict[str, Any]):
        """
        Логирует ошибку в консоль и/или файл.
        """
        message = (
            f"[ERROR] {error_info['timestamp']} "
            f"{error_info['type']}: {error_info['message']}"
        )

        # Вывод в консоль
        print(message)
        if self.log_level == "debug":
            print(error_info['traceback'])

        # Запись в файл
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(message + "\n")
                    if self.log_level == "debug":
                        f.write(error_info['traceback'] + "\n")
                        f.write("-" * 80 + "\n")
            except Exception:
                pass  # Игнорируем ошибки записи в лог

    def _format_error(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Форматирует ошибку для пользователя.
        """
        error_type = error_info['type']
        message = error_info['message']

        # Понятные сообщения для разных типов ошибок
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
            "UnicodeDecodeError": "⚠️ Ошибка кодировки. Проверьте формат файла.",
            "UnicodeEncodeError": "⚠️ Ошибка кодировки при сохранении.",
        }

        # Для ошибок из formula_engine
        if "FormulaEngine" in error_type or "calculation" in str(message).lower():
            user_message = f"📐 Ошибка расчёта: {message}"
        # Для ошибок из QA системы
        elif "QASystem" in error_type or "search" in str(message).lower():
            user_message = f"🔍 Ошибка поиска: {message}"
        # Для ошибок из агента
        elif "AgentLoop" in error_type:
            user_message = f"🤖 Ошибка агента: {message}"
        else:
            user_message = error_messages.get(error_type, f"❌ Ошибка: {message}")

        return {
            "user_message": user_message,
            "debug_info": error_info if self.log_level == "debug" else None,
            "is_error": True,
            "type": error_type,
            "message": message,
            "timestamp": error_info.get("timestamp", "")
        }

    def _extract_module_name(self, message: str) -> str:
        """
        Извлекает имя модуля из сообщения об ошибке.
        """
        import re
        match = re.search(r"'([^']+)'", message)
        return match.group(1) if match else ""

    def get_last_error(self) -> Optional[Dict[str, Any]]:
        """
        Возвращает последнюю ошибку.
        """
        return self.errors[-1] if self.errors else None

    def get_all_errors(self) -> List[Dict[str, Any]]:
        """
        Возвращает все ошибки.
        """
        return self.errors

    def clear_errors(self):
        """
        Очищает список ошибок.
        """
        self.errors.clear()

    def format_for_ui(self, error_info: Dict[str, Any]) -> str:
        """
        Форматирует ошибку для отображения в UI.
        """
        if error_info.get('is_error'):
            return error_info.get('user_message', '❌ Неизвестная ошибка')
        return f"ℹ️ {error_info.get('message', 'Неизвестная ошибка')}"

    def get_error_summary(self) -> Dict[str, Any]:
        """
        Возвращает сводку по ошибкам.
        """
        if not self.errors:
            return {"total": 0, "types": {}}

        types = {}
        for error in self.errors:
            error_type = error.get('type', 'Unknown')
            types[error_type] = types.get(error_type, 0) + 1

        return {
            "total": len(self.errors),
            "types": types,
            "last_error": self.errors[-1] if self.errors else None,
            "last_timestamp": self.errors[-1].get("timestamp") if self.errors else None
        }

    def log_to_file(self, error_info: Dict[str, Any], file_path: str):
        """
        Записывает ошибку в отдельный файл лога в формате JSON.
        """
        try:
            import json
            with open(file_path, 'a', encoding='utf-8') as f:
                json.dump(error_info, f, ensure_ascii=False, indent=2)
                f.write("\n" + "-" * 80 + "\n")
        except Exception as e:
            print(f"⚠️ Не удалось записать лог: {e}")

    def get_user_friendly_message(self, error: Exception) -> str:
        """
        Быстрый метод для получения понятного сообщения об ошибке.

        Args:
            error: Исключение

        Returns:
            Понятное сообщение для пользователя
        """
        result = self.handle(error)
        return result['user_message']

    def is_critical(self, error_info: Dict[str, Any]) -> bool:
        """
        Определяет, является ли ошибка критической.

        Критические ошибки: MemoryError, RecursionError, KeyboardInterrupt
        """
        critical_types = [
            "MemoryError",
            "RecursionError",
            "KeyboardInterrupt",
            "SystemError"
        ]
        return error_info.get('type') in critical_types


# ========== УТИЛИТНЫЕ ФУНКЦИИ ==========

def safe_execute(func, *args, **kwargs):
    """
    Безопасное выполнение функции с обработкой ошибок.

    Args:
        func: Функция для выполнения
        *args: Аргументы функции
        **kwargs: Ключевые аргументы функции

    Returns:
        Tuple (результат, ошибка или None)
    """
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        return None, e


async def safe_execute_async(async_func, *args, **kwargs):
    """
    Безопасное выполнение асинхронной функции с обработкой ошибок.

    Args:
        async_func: Асинхронная функция для выполнения
        *args: Аргументы функции
        **kwargs: Ключевые аргументы функции

    Returns:
        Tuple (результат, ошибка или None)
    """
    try:
        result = await async_func(*args, **kwargs)
        return result, None
    except Exception as e:
        return None, e