

from typing import Optional, Dict, Any
import traceback


class ErrorHandler:
    """
    Обработчик ошибок с понятными сообщениями
    """

    def __init__(self, log_level: str = "info"):
        self.log_level = log_level
        self.errors: List[Dict[str, Any]] = []

    def handle(self, error: Exception, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Обрабатывает ошибку и возвращает понятный ответ"""
        error_info = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }

        self.errors.append(error_info)
        self._log_error(error_info)

        return self._format_error(error_info)

    def _log_error(self, error_info: Dict[str, Any]):
        """Логирует ошибку"""
        print(f"[ERROR] {error_info['type']}: {error_info['message']}")
        if self.log_level == "debug":
            print(error_info['traceback'])

    def _format_error(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Форматирует ошибку для пользователя"""
        error_type = error_info['type']
        message = error_info['message']

        if "ModuleNotFoundError" in error_type:
            user_message = "⚠️ Отсутствует библиотека. Установите: " + message.split("'")[1]
        elif "FileNotFoundError" in error_type:
            user_message = "📁 Файл не найден: " + message
        elif "KeyError" in error_type:
            user_message = "🔑 Отсутствует ключ в данных: " + message
        elif "ValueError" in error_type:
            user_message = "📊 Ошибка в данных: " + message
        else:
            user_message = f"❌ Ошибка: {message}"

        return {
            "user_message": user_message,
            "debug_info": error_info if self.log_level == "debug" else None,
            "is_error": True
        }

    def get_last_error(self) -> Optional[Dict[str, Any]]:
        """Возвращает последнюю ошибку"""
        return self.errors[-1] if self.errors else None

    def clear_errors(self):
        """Очищает список ошибок"""
        self.errors.clear()