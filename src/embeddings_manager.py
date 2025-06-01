import logging
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import torch

logger = logging.getLogger(__name__)


class EmbeddingsManager:
    """Класс для управления эмбеддингами документов"""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", device: str = "cpu"):
        """
        Инициализация менеджера эмбеддингов
        Используем HuggingFace embeddings из langchain-huggingface
        """
        self.model_name = model_name
        self.device = device

        try:
            # Устанавливаем torch в режим CPU для стабильности
            torch.set_default_device('cpu')

            # Используем обновленный импорт из langchain-huggingface
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': True
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32
                },
                show_progress=False
            )

            logger.info(f"Инициализирована модель эмбеддингов: {model_name}")

        except Exception as e:
            logger.error(f"Ошибка при инициализации модели эмбеддингов: {e}")
            # Fallback к более простой конфигурации
            try:
                logger.info("Попытка инициализации с упрощенными параметрами...")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Более легкая модель
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info("Успешно инициализирована fallback модель эмбеддингов")
            except Exception as fallback_error:
                logger.error(f"Критическая ошибка при инициализации эмбеддингов: {fallback_error}")
                raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Создание эмбеддингов для списка текстов"""
        try:
            if not texts:
                logger.warning("Пустой список текстов для создания эмбеддингов")
                return []

            # Обрабатываем тексты батчами для стабильности
            batch_size = 16
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.embeddings.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)

            logger.info(f"Создано {len(all_embeddings)} эмбеддингов")
            return all_embeddings

        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддингов документов: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Создание эмбеддинга для запроса"""
        try:
            if not text or not text.strip():
                logger.warning("Пустой текст для создания эмбеддинга запроса")
                return []

            embedding = self.embeddings.embed_query(text)
            return embedding

        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддинга запроса: {e}")
            raise
