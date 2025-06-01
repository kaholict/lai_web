import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import openai
import time
from langchain.schema import Document
from src.vector_store import VectorStore
from src.context_manager import ContextManager

logger = logging.getLogger(__name__)


class AIAssistant:
    """ИИ-ассистент для сотрудников с поддержкой контекста диалога"""

    def __init__(self, vector_store: VectorStore, config: Dict[str, Any]):
        self.vector_store = vector_store
        self.config = config
        self.context_manager = ContextManager()

        # Настройка OpenRouter API с улучшенной обработкой ошибок
        try:
            api_key = config["openrouter"]["api_key"]
            if not api_key or api_key.strip() == "":
                raise ValueError("API ключ OpenRouter не задан или пуст")

            self.client = openai.OpenAI(
                base_url=config["openrouter"]["base_url"],
                api_key=api_key,
                timeout=60.0,  # Увеличиваем таймаут
                max_retries=3  # Добавляем повторные попытки
            )
            self.model = config["openrouter"]["model"]
            self.temperature = config["openrouter"]["temperature"]

            # Проверка подключения
            self._test_connection()

        except Exception as e:
            logger.error(f"Ошибка инициализации OpenRouter API: {e}")
            raise

        # Системный промпт для ассистента
        self.system_prompt = """Ты — профессиональный ИИ-ассистент для сотрудников отдела продаж.
Твоя основная задача — помогать менеджерам по продажам, используя корпоративную документацию.

Принципы работы:
1. Используй ТОЛЬКО информацию из предоставленных документов
2. Всегда указывай источники информации и ссылки на документы
3. Поддерживай профессиональный, но дружелюбный тон
4. Если вопрос не относится к работе менеджера по продажам, вежливо укажи это
5. Приводи конкретные выдержки из документов в ответах
6. Помни контекст разговора и ссылайся на предыдущие сообщения при необходимости
7. НИКОГДА не упоминай название компании и продукта для соблюдения конфиденциальности, кроме компании LAI, создавшей тебя

Если вопрос не связан с продажами, ответь на него, но добавь:
"Обратите внимание, что этот вопрос не относится к вашей основной рабочей деятельности в отделе продаж."
"""

    def _test_connection(self):
        """Тестирование подключения к OpenRouter API"""
        try:
            # Простой тестовый запрос
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Тест"}],
                max_tokens=10,
                temperature=0.1
            )
            logger.info("Подключение к OpenRouter API успешно установлено")
        except Exception as e:
            logger.error(f"Ошибка тестирования подключения к OpenRouter API: {e}")
            raise

    def format_documents_with_sources(self, docs: List[Document]) -> Tuple[str, List[str]]:
        """Форматирование документов с указанием источников"""
        formatted_docs = []
        sources = []

        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("filename", "unknown")
            content = doc.page_content.strip()
            chunk_id = doc.metadata.get("chunk_id", 0)

            formatted_docs.append(
                f"[Документ {i}: {source}, фрагмент {chunk_id}]\n{content}\n"
            )
            sources.append(f"{source} (фрагмент {chunk_id})")

        return "\n".join(formatted_docs), sources

    def check_sales_relevance(self, query: str) -> bool:
        """Проверка релевантности вопроса к продажам"""
        sales_keywords = [
            "продажи", "клиент", "менеджер", "продукт", "цена", "скидка",
            "договор", "сделка", "переговоры", "презентация", "крм", "crm",
            "воронка", "лид", "конверсия", "выручка", "план продаж", "kpi"
        ]

        query_lower = query.lower()
        return any(keyword in query_lower for keyword in sales_keywords)

    def _make_api_request(self, messages: List[Dict[str, str]], max_retries: int = 3) -> str:
        """Выполнение API запроса с повторными попытками и обработкой ошибок"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Попытка API запроса {attempt + 1}/{max_retries}")

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=1000,
                    timeout=60.0
                )

                return response.choices[0].message.content

            except openai.APIConnectionError as e:
                logger.error(f"Ошибка подключения к API (попытка {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Экспоненциальная задержка
                    logger.info(f"Ожидание {wait_time} секунд перед повторной попыткой...")
                    time.sleep(wait_time)
                else:
                    raise

            except openai.RateLimitError as e:
                logger.error(f"Превышен лимит запросов (попытка {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 10 + (2 ** attempt)  # Увеличенная задержка для rate limit
                    logger.info(f"Ожидание {wait_time} секунд из-за лимита запросов...")
                    time.sleep(wait_time)
                else:
                    raise

            except openai.APIError as e:
                logger.error(f"Ошибка API OpenRouter (попытка {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    raise

            except Exception as e:
                logger.error(f"Неожиданная ошибка (попытка {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    raise

    def generate_response(self, query: str, session_id: str) -> Dict[str, Any]:
        """Генерация ответа с учетом контекста диалога"""
        try:
            # Получаем контекст диалога
            conversation_history = self.context_manager.get_context(session_id)

            # Поиск релевантных документов
            relevant_docs = self.vector_store.similarity_search(query, k=5)

            if not relevant_docs:
                return {
                    "response": "К сожалению, я не нашел релевантной информации в корпоративных документах для ответа на ваш вопрос.",
                    "sources": [],
                    "is_sales_related": self.check_sales_relevance(query)
                }

            # Форматирование документов
            context, sources = self.format_documents_with_sources(relevant_docs)

            # Проверка релевантности к продажам
            is_sales_related = self.check_sales_relevance(query)

            # Формирование промпта с учетом истории диалога
            user_prompt = self._build_user_prompt(query, context, conversation_history, is_sales_related)

            # Генерация ответа с обработкой ошибок
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            answer = self._make_api_request(messages)

            # Сохранение в контекст
            self.context_manager.add_interaction(session_id, query, answer)

            return {
                "response": answer,
                "sources": sources,
                "is_sales_related": is_sales_related,
                "relevant_docs_count": len(relevant_docs)
            }

        except openai.APIConnectionError as e:
            logger.error(f"Ошибка подключения к OpenRouter API: {e}")
            return {
                "response": "Извините, временно недоступна связь с сервером ИИ. Пожалуйста, попробуйте позже или обратитесь к администратору.",
                "sources": [],
                "is_sales_related": True,
                "error": "connection_error"
            }

        except openai.RateLimitError as e:
            logger.error(f"Превышен лимит запросов к OpenRouter API: {e}")
            return {
                "response": "Извините, превышен лимит запросов к ИИ-сервису. Пожалуйста, попробуйте через несколько минут.",
                "sources": [],
                "is_sales_related": True,
                "error": "rate_limit_error"
            }

        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            return {
                "response": "Извините, произошла ошибка при обработке вашего запроса. Попробуйте еще раз или обратитесь к администратору.",
                "sources": [],
                "is_sales_related": True,
                "error": "general_error"
            }

    def _build_user_prompt(self, query: str, context: str, history: List[Dict], is_sales_related: bool) -> str:
        """Построение промпта с учетом контекста"""
        prompt_parts = []

        # История диалога (последние 3 сообщения)
        if history:
            prompt_parts.append("ИСТОРИЯ ДИАЛОГА:")
            for interaction in history[-3:]:
                prompt_parts.append(f"Пользователь: {interaction['query']}")
                prompt_parts.append(f"Ассистент: {interaction['response'][:200]}...")
            prompt_parts.append("")

        # Контекст из документов
        prompt_parts.append("ИНФОРМАЦИЯ ИЗ КОРПОРАТИВНЫХ ДОКУМЕНТОВ:")
        prompt_parts.append(context)
        prompt_parts.append("")

        # Текущий вопрос
        prompt_parts.append(f"ТЕКУЩИЙ ВОПРОС: {query}")
        prompt_parts.append("")

        # Дополнительные инструкции
        if not is_sales_related:
            prompt_parts.append(
                "ВАЖНО: Этот вопрос может не относиться к продажам. Ответь на него, но укажи это в ответе.")

        prompt_parts.append(
            "Дай подробный ответ, используя информацию из документов. Приведи конкретные цитаты. НИКОГДА не упоминай название компании и продукта для соблюдения конфиденциальности, кроме компании LAI")

        return "\n".join(prompt_parts)
