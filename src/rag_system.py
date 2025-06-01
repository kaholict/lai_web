# import logging
# from typing import List, Dict, Any, Optional
# import openai
# from langchain.schema import Document
# from src.vector_store import VectorStore

# logger = logging.getLogger(__name__)


# class RAGSystem:
#     """Система Retrieval Augmented Generation с использованием OpenRouter API"""

#     def __init__(self, vector_store: VectorStore, config: Dict[str, Any]):
#         self.vector_store = vector_store
#         self.config = config

#         # Настройка OpenRouter API[1][2]
#         self.client = openai.OpenAI(
#             base_url=config["openrouter"]["base_url"],
#             api_key=config["openrouter"]["api_key"]
#         )
#         self.model = config["openrouter"]["model"]
#         self.temperature = config["openrouter"]["temperature"]

#     def format_documents(self, docs: List[Document]) -> str:
#         """Форматирование документов для использования в промпте"""
#         formatted_docs = []
#         for i, doc in enumerate(docs, 1):
#             source = doc.metadata.get("filename", "unknown")
#             content = doc.page_content.strip()
#             formatted_docs.append(f"Документ {i} (источник: {source}):\n{content}")

#         return "\n\n".join(formatted_docs)

#     def generate_response(self, query: str, context_docs: List[Document]) -> str:
#         """Генерация ответа с использованием контекста из документов"""

#         context = self.format_documents(context_docs)

#         system_prompt = """Ты — эксперт по обучению и развитию персонала, специализирующийся на создании образовательных программ для отдела продаж. 

# Твоя задача — создавать качественный образовательный контент на основе предоставленной документации по онбордингу.

# Принципы работы:
# 1. Используй только информацию из предоставленных документов
# 2. Структурируй ответы логично и последовательно
# 3. Адаптируй контент для новых сотрудников
# 4. Включай практические примеры и упражнения
# 5. Отвечай на русском языке

# Если в документах нет достаточной информации для ответа, честно об этом сообщи."""

#         user_prompt = f"""
# Контекст из документов онбординга:
# {context}

# Запрос: {query}

# Ответ:"""

#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt}
#                 ],
#                 temperature=self.temperature,
#                 max_tokens=8000
#             )

#             return response.choices[0].message.content

#         except Exception as e:
#             logger.error(f"Ошибка при генерации ответа: {e}")
#             return "Извините, произошла ошибка при генерации ответа."

#     # def query(self, question: str, k: int = 5) -> str:
#     #     """Основной метод для запросов к RAG системе"""
#     #     try:
#     #         # Поиск релевантных документов
#     #         relevant_docs = self.vector_store.similarity_search(question, k=k)

#     #         if not relevant_docs:
#     #             return "К сожалению, не найдено релевантной информации в документах."

#     #         # Генерация ответа
#     #         response = self.generate_response(question, relevant_docs)
#     #         return response

#     #     except Exception as e:
#     #         logger.error(f"Ошибка при обработке запроса: {e}")
#     #         return "Произошла ошибка при обработке запроса."


#     def query(self, question: str, k: Optional[int] = None) -> str:
#         """Основной метод для запросов к RAG системе"""
#         try:
#             # Поиск релевантных документов
#             relevant_docs = self.vector_store.similarity_search(question, k=k)
#             if not relevant_docs:
#                 return "К сожалению, не найдено релевантной информации в документах."
#             # Генерация ответа
#             response = self.generate_response(question, relevant_docs)
#             return response
#         except Exception as e:
#             self.logger.error(f"Ошибка при обработке запроса: {e}")
#             return "Произошла ошибка при обработке запроса."


import logging
from typing import List, Dict, Any, Optional
import openai
import json
import time
from langchain.schema import Document
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

class RAGSystem:
    """Система Retrieval Augmented Generation с использованием OpenRouter API"""
    
    def __init__(self, vector_store: VectorStore, config: Dict[str, Any]):
        self.vector_store = vector_store
        self.config = config
        
        # Настройка OpenRouter API[1][2]
        self.client = openai.OpenAI(
            base_url=config["openrouter"]["base_url"],
            api_key=config["openrouter"]["api_key"]
        )
        self.model = config["openrouter"]["model"]
        self.temperature = config["openrouter"]["temperature"]
    
    def format_documents(self, docs: List[Document]) -> str:
        """Форматирование документов для использования в промпте"""
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("filename", "unknown")
            content = doc.page_content.strip()
            formatted_docs.append(f"Документ {i} (источник: {source}):\n{content}")
        
        return "\n\n".join(formatted_docs)
    
    def generate_response(self, query: str, context_docs: List[Document], max_tokens: int = 2000, response_format: str = "text") -> str:
        """Генерация ответа с использованием контекста из документов"""
        
        context = self.format_documents(context_docs)
        
        system_prompt = """Ты — эксперт по обучению и развитию персонала, специализирующийся на создании образовательных программ для отдела продаж. 
        
Твоя задача — создавать качественный образовательный контент на основе предоставленной документации по онбордингу.

Принципы работы:
1. Используй только информацию из предоставленных документов
2. Структурируй ответы логично и последовательно
3. Адаптируй контент для новых сотрудников
4. Включай практические примеры и упражнения
5. Отвечай на русском языке

Если в документах нет достаточной информации для ответа, честно об этом сообщи."""

        # Дополнительные инструкции для JSON ответов
        if response_format == "json":
            system_prompt += "\n\nВАЖНО: Твой ответ должен быть валидным JSON объектом. Убедись, что JSON полный и корректно закрывается."

        user_prompt = f"""
Контекст из документов онбординга:
{context}

Запрос: {query}

Ответ:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            return "Извините, произошла ошибка при генерации ответа."
    
    def generate_structured_json_response(self, query: str, context_docs: List[Document], max_retries: int = 3) -> Dict[str, Any]:
        """Генерация структурированного JSON ответа с повторными попытками при неполном JSON"""
        
        # Начинаем с базового количества токенов и увеличиваем при необходимости
        max_tokens_attempts = [4000, 6000, 8000, 12000, 16000, 20000]
        
        for attempt in range(max_retries):
            try:
                # Выбираем количество токенов для текущей попытки
                current_max_tokens = max_tokens_attempts[min(attempt, len(max_tokens_attempts) - 1)]
                
                logger.info(f"Попытка {attempt + 1}: генерация JSON с max_tokens={current_max_tokens}")
                
                response_text = self.generate_response(
                    query, 
                    context_docs, 
                    max_tokens=current_max_tokens,
                    response_format="json"
                )
                
                # Пытаемся извлечь JSON из ответа
                json_result = self._extract_and_validate_json(response_text)
                
                if json_result:
                    logger.info(f"Успешно получен валидный JSON на попытке {attempt + 1}")
                    return json_result
                else:
                    logger.warning(f"Попытка {attempt + 1}: не удалось получить валидный JSON")
                    
            except Exception as e:
                logger.error(f"Ошибка на попытке {attempt + 1}: {e}")
                
            # Небольшая пауза между попытками
            time.sleep(1)
        
        # Если все попытки не удались, возвращаем базовую структуру
        logger.error("Не удалось получить валидный JSON после всех попыток")
        return None
    
    def _extract_and_validate_json(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Извлечение и валидация JSON из текста ответа"""
        
        if not response_text or not response_text.strip():
            return None
            
        # Пытаемся найти JSON в ответе
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            logger.warning("Не найдены JSON скобки в ответе")
            return None
        
        json_str = response_text[start_idx:end_idx + 1]
        
        try:
            # Пытаемся парсить JSON
            json_result = json.loads(json_str)
            
            # Проверяем, что JSON содержит ожидаемые поля для структуры курса
            if self._validate_course_structure(json_result):
                return json_result
            else:
                logger.warning("JSON не содержит ожидаемую структуру курса")
                return None
                
        except json.JSONDecodeError as e:
            logger.warning(f"Ошибка парсинга JSON: {e}")
            
            # Пытаемся "исправить" неполный JSON
            fixed_json = self._attempt_json_fix(json_str)
            if fixed_json:
                return fixed_json
                
        return None
    
    def _attempt_json_fix(self, json_str: str) -> Optional[Dict[str, Any]]:
        """Попытка исправить неполный JSON"""
        
        # Удаляем неполные строки в конце
        lines = json_str.split('\n')
        for i in range(len(lines) - 1, -1, -1):
            try:
                # Пытаемся добавить недостающие закрывающие скобки
                test_json = '\n'.join(lines[:i+1])
                
                # Подсчитываем открытые и закрытые скобки
                open_braces = test_json.count('{')
                close_braces = test_json.count('}')
                open_brackets = test_json.count('[')
                close_brackets = test_json.count(']')
                
                # Добавляем недостающие закрывающие символы
                missing_brackets = open_brackets - close_brackets
                missing_braces = open_braces - close_braces
                
                fixed_json = test_json + ']' * missing_brackets + '}' * missing_braces
                
                result = json.loads(fixed_json)
                if self._validate_course_structure(result):
                    logger.info("Успешно исправлен неполный JSON")
                    return result
                    
            except (json.JSONDecodeError, IndexError):
                continue
                
        return None
    
    def _validate_course_structure(self, json_obj: Dict[str, Any]) -> bool:
        """Валидация структуры курса в JSON"""
        
        required_fields = ["course_title", "description", "modules"]
        
        # Проверяем наличие обязательных полей
        for field in required_fields:
            if field not in json_obj:
                return False
        
        # Проверяем, что modules - это список
        if not isinstance(json_obj["modules"], list):
            return False
            
        # Проверяем, что есть хотя бы один модуль
        if len(json_obj["modules"]) == 0:
            return False
            
        # Проверяем структуру первого модуля
        first_module = json_obj["modules"][0]
        module_required_fields = ["module_title", "module_description", "lessons"]
        
        for field in module_required_fields:
            if field not in first_module:
                return False
                
        return True

    def query(self, question: str, k: Optional[int] = None, max_tokens: int = 2000, response_format: str = "text") -> str:
        """Основной метод для запросов к RAG системе"""
        try:
            # Поиск релевантных документов
            relevant_docs = self.vector_store.similarity_search(question, k=k)
            
            if not relevant_docs:
                return "К сожалению, не найдено релевантной информации в документах."
            
            # Генерация ответа
            response = self.generate_response(question, relevant_docs, max_tokens=max_tokens, response_format=response_format)
            return response
            
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {e}")
            return "Произошла ошибка при обработке запроса."
    
    def query_json(self, question: str, k: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Метод для получения структурированного JSON ответа"""
        try:
            # Поиск релевантных документов
            relevant_docs = self.vector_store.similarity_search(question, k=k)
            
            if not relevant_docs:
                logger.warning("Не найдено релевантных документов для JSON запроса")
                return None
            
            # Генерация структурированного JSON ответа
            return self.generate_structured_json_response(question, relevant_docs)
            
        except Exception as e:
            logger.error(f"Ошибка при обработке JSON запроса: {e}")
            return None
