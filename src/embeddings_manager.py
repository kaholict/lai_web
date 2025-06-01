import logging
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

logger = logging.getLogger(__name__)


class EmbeddingsManager:
    """Класс для управления эмбеддингами документов"""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", device: str = "cpu"):
        """
        Инициализация менеджера эмбеддингов
        Используем HuggingFace embeddings из langchain-community
        """
        self.model_name = model_name
        self.device = device

        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': "cpu"},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"Инициализирована модель эмбеддингов: {model_name}")
        except Exception as e:
            logger.error(f"Ошибка при инициализации модели эмбеддингов: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Создание эмбеддингов для списка текстов"""
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Создано {len(embeddings)} эмбеддингов")
            return embeddings
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддингов документов: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Создание эмбеддинга для запроса"""
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддинга запроса: {e}")
            raise
