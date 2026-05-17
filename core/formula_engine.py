
import re
from typing import Dict, List


class FormulaEngine:
    """Движок для работы с формулами из строительной документации."""

    def __init__(self):
        self.formulas = self._init_formulas()

    def _init_formulas(self) -> Dict[str, Dict]:
        """Инициализация базы формул."""
        return {
            "thickness": {
                "name": "Толщина тепловой изоляции",
                "expression": "δ = λ × ((tв - tн) / q - Rн)",
                "variables": ["tв", "tн", "λ", "q", "Rн"],
                "description": "Расчет толщины изоляции по нормированной плотности теплового потока",
            },
            "heat_loss": {
                "name": "Тепловые потери",
                "expression": "q = (tв - tн) / (Rиз + Rн)",
                "variables": ["tв", "tн", "Rиз", "Rн"],
                "description": "Расчет плотности теплового потока",
            },
            "surface_temp": {
                "name": "Температура на поверхности",
                "expression": "tпов = tн + (tв - tн) × Rн / (Rиз + Rн)",
                "variables": ["tв", "tн", "Rиз", "Rн"],
                "description": "Расчет температуры наружной поверхности изоляции",
            },
            "resistance": {
                "name": "Термическое сопротивление",
                "expression": "R = δ / λ",
                "variables": ["δ", "λ"],
                "description": "Расчет термического сопротивления слоя изоляции",
            },
        }

    def get_available_formulas(self) -> List[Dict]:
        """Список доступных формул."""
        return [
            {
                "name": f["name"],
                "expression": f["expression"],
                "variables": f["variables"],
                "description": f["description"],
            }
            for f in self.formulas.values()
        ]

    def answer_calculation(self, question: str) -> Dict[str, str]:
        """Ответ на расчетный вопрос."""
        q_lower = question.lower()

        numbers = re.findall(r"(\d+(?:[.,]\d+)?)\s*(°c|мм|м|вт)?", q_lower)
        temp_values = []
        for num, unit in numbers:
            try:
                value = float(num.replace(",", "."))
                if unit == "°c" or "градус" in q_lower:
                    temp_values.append(value)
            except ValueError:
                continue

        if "толщин" in q_lower:
            return self._calculate_thickness(temp_values, question)
        if "потер" in q_lower or "теплопотер" in q_lower:
            return self._calculate_heat_loss(temp_values)
        if "температур" in q_lower and "поверхн" in q_lower:
            return self._calculate_surface_temp(temp_values)

        return self._general_calculation_guide()

    def _calculate_thickness(self, temps: List[float], question: str) -> Dict[str, str]:
        """Расчет толщины изоляции."""
        t_in = temps[0] if temps else 150
        t_out = 20
        lam = 0.045
        q_norm = 40
        R_outer = 0.1

        delta = lam * ((t_in - t_out) / q_norm - R_outer)
        delta_mm = delta * 1000

        answer_text = f"""
**📐 Расчет толщины тепловой изоляции**

**Исходные данные:**
- Температура теплоносителя: **{t_in}°C**
- Температура окружающей среды: **{t_out}°C**
- Теплопроводность изоляции: **0,045 Вт/(м·°C)**
- Нормированная плотность теплового потока: **40 Вт/м²**
- Сопротивление теплоотдаче: **0,1 м²·°C/Вт**

**Формула расчета (СП 61.13330.2012, формула В.18):**
δ = λ × ((tв - tн) / q - Rн)

text

**Подстановка:**
δ = {lam} × (({t_in} - {t_out}) / {q_norm} - {R_outer})
δ = {lam} × ({t_in - t_out} / {q_norm} - {R_outer})
δ = {lam} × ({(t_in - t_out) / q_norm:.3f} - {R_outer})
δ = {lam} × {((t_in - t_out) / q_norm - R_outer):.4f}

text

**Результат:**
- Расчетная толщина: **{delta:.3f} м** = **{delta_mm:.0f} мм**
- Рекомендуемая толщина: **{round(delta_mm/10)*10} мм**
"""
        return {"answer": answer_text}

    def _calculate_heat_loss(self, temps: List[float]) -> Dict[str, str]:
        """Расчет тепловых потерь."""
        t_in = temps[0] if temps else 150
        t_out = 20
        R_ins = 2.0
        R_outer = 0.15

        q = (t_in - t_out) / (R_ins + R_outer)

        answer_text = f"""
**📊 Расчет тепловых потерь**

**Исходные данные:**
- Температура теплоносителя: **{t_in}°C**
- Температура окружающей среды: **{t_out}°C**
- Термическое сопротивление изоляции: **{R_ins} м²·°C/Вт**
- Сопротивление теплоотдаче: **{R_outer} м²·°C/Вт**

**Формула (СП 61.13330.2012, формула В.17):**
q = (tв - tн) / (Rиз + Rн)

text

**Подстановка:**
q = ({t_in} - {t_out}) / ({R_ins} + {R_outer})
q = {t_in - t_out} / {R_ins + R_outer}
q = {q:.1f} Вт/м

text

**Результат:**
- Тепловые потери: **{q:.1f} Вт/м**
"""
        return {"answer": answer_text}

    def _calculate_surface_temp(self, temps: List[float]) -> Dict[str, str]:
        """Расчет температуры на поверхности."""
        t_in = temps[0] if temps else 150
        t_out = 20
        R_ins = 2.0
        R_outer = 0.15

        t_surf = t_out + (t_in - t_out) * R_outer / (R_ins + R_outer)

        answer_text = f"""
**🌡️ Расчет температуры на поверхности изоляции**

**Исходные данные:**
- Температура теплоносителя: **{t_in}°C**
- Температура окружающей среды: **{t_out}°C**
- Термическое сопротивление изоляции: **{R_ins} м²·°C/Вт**
- Сопротивление теплоотдаче: **{R_outer} м²·°C/Вт**

**Формула (СП 61.13330.2012, формула В.29):**
tпов = tн + (tв - tн) × Rн / (Rиз + Rн)

text

**Подстановка:**
tпов = {t_out} + ({t_in} - {t_out}) × {R_outer} / ({R_ins} + {R_outer})
tпов = {t_out} + {t_in - t_out} × {R_outer} / {R_ins + R_outer}
tпов = {t_out} + {t_in - t_out} × {(R_outer/(R_ins+R_outer)):.3f}
tпов = {t_surf:.1f}°C

text

**Результат:**
- Температура на поверхности: **{t_surf:.1f}°C**
"""
        return {"answer": answer_text}

    def _general_calculation_guide(self) -> Dict[str, str]:
        """Общее руководство по расчетам."""
        answer_text = """
**📐 Как рассчитать тепловую изоляцию**

**Основные формулы:**

1. **Толщина изоляции:** δ = λ × ((tв - tн) / q - Rн)
2. **Тепловые потери:** q = (tв - tн) / (Rиз + Rн)
3. **Температура поверхности:** tпов = tн + (tв - tн) × Rн / (Rиз + Rн)

**Примеры вопросов:**
- "Рассчитай толщину изоляции при температуре 150°C"
- "Какие тепловые потери при 200°C?"

**Для точного расчета укажите:**
- Температуру теплоносителя (°C)
- Температуру окружающей среды (°C)
- Тип поверхности (плоская или цилиндрическая)
"""
        return {"answer": answer_text}