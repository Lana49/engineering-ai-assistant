from typing import List, Dict, Any, Optional
import re


class AgentLoop:
    """
    Агентский цикл для последовательной обработки запросов.
    Адаптировано из Gramax runAgentTurn
    """

    def __init__(self, qa_system, formula_engine):
        self.qa_system = qa_system
        self.formula_engine = formula_engine
        self.messages: List[Dict[str, Any]] = []
        self.max_steps = 5
        self.last_error: Optional[str] = None

    async def run(self, user_content: str) -> Dict[str, Any]:
        """
        Запуск агентского цикла
        """
        self.messages.append({"role": "user", "content": user_content})

        for step in range(self.max_steps):
            try:
                # 1. Определяем тип запроса
                query_type = self._detect_query_type(user_content)

                # 2. Обрабатываем запрос
                if query_type == "calculation":
                    result = self._handle_calculation(user_content)
                elif query_type == "definition":
                    result = self._handle_definition(user_content)
                elif query_type == "table":
                    result = self._handle_table(user_content)
                else:
                    result = self._handle_search(user_content)

                # 3. Проверяем результат
                if self._is_complete(result):
                    self.messages.append({"role": "assistant", "content": result})
                    return {
                        "answer": result,
                        "sources": self._extract_sources(result),
                        "steps": step + 1
                    }

                # 4. Если нужны уточнения
                if self._need_clarification(result):
                    return {
                        "answer": result,
                        "sources": [],
                        "steps": step + 1,
                        "needs_clarification": True
                    }

            except Exception as e:
                self.last_error = str(e)
                return {
                    "answer": f"❌ Ошибка: {e}",
                    "sources": [],
                    "steps": step + 1,
                    "error": str(e)
                }

        return {
            "answer": "❌ Превышено максимальное количество шагов",
            "sources": [],
            "steps": self.max_steps
        }

    def _detect_query_type(self, query: str) -> str:
        """Определяет тип запроса"""
        lower = query.lower()

        calc_triggers = [
            'рассчитай', 'вычисли', 'посчитай',
            'гсоп', 'градусо-сутки',
            'толщин', 'изоляци',
            'потери', 'теплопотер',
            'вентиляц', 'расход'
        ]
        if any(w in lower for w in calc_triggers):
            return "calculation"

        def_triggers = [
            'что такое', 'определение', 'термин',
            'что значит', 'что означает',
            'расшифруй', 'аббревиатура'
        ]
        if any(w in lower for w in def_triggers):
            return "definition"

        table_triggers = ['таблиц', 'табл', 'покажи таблиц', 'выведи таблиц']
        if any(w in lower for w in table_triggers):
            return "table"

        return "search"

    def _handle_calculation(self, query: str) -> str:
        """Обработка расчётного запроса"""
        result = self.formula_engine.answer_calculation(query)
        return result.get('answer', '❌ Не удалось выполнить расчёт')

    def _handle_definition(self, query: str) -> str:
        """Обработка запроса определения"""
        clean = query
        for trigger in ['что такое', 'определение', 'термин', 'что значит', 'что означает']:
            clean = clean.replace(trigger, "").strip()

        result = self.qa_system.find_definition(clean)
        if result['found']:
            return f"📖 **Определение термина «{clean}»:**\n\n{result['definition']}\n\n📚 **Источник:** {result['source']}"
        return f"⚠️ Определение для термина «{clean}» не найдено."

    def _handle_table(self, query: str) -> str:
        """Обработка запроса таблицы"""
        result = self.qa_system.answer(query)
        if result.get('sources'):
            for source in result['sources']:
                if 'ТАБЛИЦА' in source['text']:
                    return self._format_table(source)
        return "❌ Таблица не найдена."

    def _handle_search(self, query: str) -> str:
        """Обработка поискового запроса"""
        result = self.qa_system.answer(query)
        return result.get('answer', '❌ Информация не найдена')

    def _format_table(self, source: Dict) -> str:
        """Форматирует таблицу из текста"""
        text = source['text']
        lines = text.split('\n')
        table_lines = []
        in_table = False

        for line in lines:
            if '[ТАБЛИЦА]' in line:
                in_table = True
                continue
            if in_table and line.strip() == '':
                break
            if in_table:
                table_lines.append(line)

        if table_lines:
            return f"📊 **Найдена таблица:**\n\n" + "\n".join(table_lines[:15]) + \
                   f"\n\n📚 **Источник:** {source['doc_name']}"
        return "❌ Таблица не найдена."

    def _is_complete(self, response: str) -> bool:
        """Проверяет, завершён ли ответ"""
        return len(response) > 100 and '❌' not in response

    def _need_clarification(self, response: str) -> bool:
        """Проверяет, нужно ли уточнение"""
        if '❌' in response or 'не найдено' in response:
            return True
        if len(response) < 50:
            return True
        return False

    def _extract_sources(self, response: str) -> List[str]:
        """Извлекает источники из ответа"""
        sources = []
        if 'Источник:' in response:
            lines = response.split('\n')
            for line in lines:
                if 'Источник:' in line:
                    sources.append(line.replace('Источник:', '').strip())
        return sources