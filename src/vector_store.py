import logging
from typing import List, Optional
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from src.embeddings_manager import EmbeddingsManager

logger = logging.getLogger(__name__)


class VectorStore:
    """Класс для управления векторным хранилищем"""

    def __init__(self, embeddings_manager: EmbeddingsManager, persist_directory: str = "./data/vector_store"):
        self.embeddings_manager = embeddings_manager
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.vector_store: Optional[FAISS] = None
        self.logger = logging.getLogger(__name__)

    def create_vector_store(self, documents: List[Document]) -> None:
        """Создание векторного хранилища из документов"""
        try:
            if not documents:
                logger.warning("Нет документов для создания векторного хранилища")
                return

            logger.info(f"Создание векторного хранилища из {len(documents)} документов")
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings_manager.embeddings
            )

            # Сохраняем векторное хранилище[12]
            self.save_vector_store()
            logger.info("Векторное хранилище успешно создано и сохранено")

        except Exception as e:
            logger.error(f"Ошибка при создании векторного хранилища: {e}")
            raise

    def save_vector_store(self) -> None:
        """Сохранение векторного хранилища на диск"""
        if self.vector_store is None:
            logger.warning("Нет векторного хранилища для сохранения")
            return

        try:
            self.vector_store.save_local(str(self.persist_directory))
            logger.info(f"Векторное хранилище сохранено в {self.persist_directory}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении векторного хранилища: {e}")
            raise

    def load_vector_store(self) -> bool:
        """Загрузка существующего векторного хранилища"""
        try:
            index_path = self.persist_directory / "index.faiss"
            if not index_path.exists():
                logger.info("Существующее векторное хранилище не найдено")
                return False

            self.vector_store = FAISS.load_local(
                str(self.persist_directory),
                self.embeddings_manager.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Векторное хранилище успешно загружено")
            return True

        except Exception as e:
            logger.error(f"Ошибка при загрузке векторного хранилища: {e}")
            return False

    # def similarity_search(self, query: str, k: int = 5) -> List[Document]:
    #     """Поиск наиболее похожих документов"""
    #     if self.vector_store is None:
    #         logger.error("Векторное хранилище не инициализировано")
    #         return []

    #     try:
    #         docs = self.vector_store.similarity_search(query, k=k)
    #         logger.info(f"Найдено {len(docs)} релевантных документов")
    #         return docs
    #     except Exception as e:
    #         logger.error(f"Ошибка при поиске: {e}")
    #         return []

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Поиск наиболее похожих документов. Если k=None — возвращает все документы."""
        if self.vector_store is None:
            self.logger.error("Векторное хранилище не инициализировано")
            return []
        try:
            if k is None:
                # Возвращаем все документы
                all_docs = list(self.vector_store.docstore._dict.values())
                self.logger.info(f"Возвращены все {len(all_docs)} документов для запроса")
                return all_docs
            else:
                return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            self.logger.error(f"Ошибка в similarity_search: {e}")
            return []


    def add_documents(self, documents: List[Document]) -> None:
        """Добавление новых документов в существующее хранилище"""
        if self.vector_store is None:
            logger.error("Векторное хранилище не инициализировано")
            return

        try:
            self.vector_store.add_documents(documents)
            self.save_vector_store()
            logger.info(f"Добавлено {len(documents)} документов в векторное хранилище")
        except Exception as e:
            logger.error(f"Ошибка при добавлении документов: {e}")
            raise
