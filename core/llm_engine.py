
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
import re


class LLMEngine:
    """Движок для работы с локальной языковой моделью"""

    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", use_gpu: bool = None):
        """
        Инициализация LLM
        model_name: название модели (можно заменить на русскую Saiga)
        use_gpu: использовать ли GPU (автоопределение если None)
        """
        self.model_name = model_name
        self.use_gpu = use_gpu if use_gpu is not None else torch.cuda.is_available()
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

        # Альтернативные русские модели
        self.russian_models = [
            "IlyaGusev/saiga_mistral_7b",
            "IlyaGusev/saiga_llama3_8b",
            "mistralai/Mistral-7B-Instruct-v0.2"
        ]

    def load_model(self):
        """Загружает модель в память"""
        if self.is_loaded:
            return True

        print(f"🔄 Загрузка модели {self.model_name}...")

        try:
            # Загрузка токенизатора
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Настройка паддинга
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Загрузка модели
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                device_map="auto" if self.use_gpu else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            if not self.use_gpu:
                self.model = self.model.to("cpu")

            self.is_loaded = True
            device = "GPU" if self.use_gpu else "CPU"
            print(f"✅ Модель загружена на {device}")
            return True

        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False

    def generate(self, prompt: str, context: str = None,
                 max_new_tokens: int = 500,
                 temperature: float = 0.7,
                 top_p: float = 0.95) -> str:
        """
        Генерация ответа на основе промпта и контекста
        """
        if not self.is_loaded:
            if not self.load_model():
                return "Ошибка: модель не загружена"

        # Формирование полного промпта
        if context:
            full_prompt = f"""Ты — эксперт по строительной документации, специализирующийся на СП 61.13330.2012 "Тепловая изоляция оборудования и трубопроводов".

Контекст из документации:
{context}

Вопрос: {prompt}

Ответ (кратко, четко, со ссылками на нормативы):"""
        else:
            full_prompt = f"""Ты — эксперт по строительной документации. Отвечай кратко и по делу.

Вопрос: {prompt}

Ответ:"""

        # Токенизация
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        if self.use_gpu:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Генерация
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Декодирование
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Извлекаем только ответ (после "Ответ:")
        if "Ответ:" in response:
            response = response.split("Ответ:")[-1].strip()

        # Очистка от повторений
        response = self._clean_response(response)

        return response

    def _clean_response(self, text: str) -> str:
        """Очистка ответа от мусора"""
        # Убираем повторяющиеся строки
        lines = text.split('\n')
        unique_lines = []
        seen = set()
        for line in lines:
            line = line.strip()
            if line and line not in seen and len(line) > 5:
                seen.add(line)
                unique_lines.append(line)

        return '\n'.join(unique_lines[:20])

    def answer_with_context(self, question: str, context: str,
                            sources: List[Dict] = None) -> Dict:
        """
        Генерация ответа с использованием контекста из документации
        """
        if not self.is_loaded:
            self.load_model()

        # Формируем контекст
        context_text = context
        if sources:
            context_text += "\n\nИсточники:\n"
            for src in sources[:3]:
                context_text += f"- {src['doc_name']} (релевантность: {src['score']:.2f})\n"
                context_text += f"  {src['text'][:200]}...\n"

        # Генерируем ответ
        answer = self.generate(
            prompt=question,
            context=context_text,
            max_new_tokens=500,
            temperature=0.3
        )

        return {
            'question': question,
            'answer': answer,
            'sources': sources or [],
            'model': self.model_name
        }

    def unload(self):
        """Выгружает модель из памяти"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            self.is_loaded = False
            print("✅ Модель выгружена")