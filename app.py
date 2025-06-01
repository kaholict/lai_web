#!/usr/bin/env python3
"""
Главный файл для деплоя в Streamlit Cloud
"""

import logging
import sys
import os
from pathlib import Path
import streamlit as st

# Добавляем текущую директорию в путь
sys.path.append(str(Path(__file__).parent))


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def cloud_init():
    """Автоматическая инициализация для Streamlit Cloud"""
    try:
        from src.config_manager import ConfigManager
        from src.document_processor import DocumentProcessor
        from src.embeddings_manager import EmbeddingsManager
        from src.vector_store import VectorStore

        config = ConfigManager()
        docs_path = Path("data/raw")

        # Проверка и создание векторного хранилища
        embeddings_manager = EmbeddingsManager(
            model_name=config.get("embeddings.model_name"),
            device="cpu"
        )

        vector_store = VectorStore(
            embeddings_manager=embeddings_manager,
            persist_directory=config.get("vector_store.persist_directory")
        )

        if not vector_store.load_vector_store():
            logging.info("Векторное хранилище не найдено, начинаем обработку документов")
            processor = DocumentProcessor(
                chunk_size=config.get("document_processing.chunk_size"),
                chunk_overlap=config.get("document_processing.chunk_overlap")
            )

            if docs_path.exists():
                documents = processor.process_documents(str(docs_path))
                if documents:
                    vector_store.create_vector_store(documents)
                    logging.info(f"Обработано {len(documents)} документов")
                else:
                    logging.warning("Документы не найдены")
            else:
                logging.warning("Папка с документами не найдена")

        return True
    except Exception as e:
        logging.error(f"Ошибка инициализации: {e}")
        return False


def main():
    setup_logging()

    # Проверяем переменные окружения
    if not os.getenv("OPENROUTER_API_KEY"):
        st.error("Не задан API ключ OpenRouter. Установите переменную окружения OPENROUTER_API_KEY")
        st.stop()

    # Инициализация
    if cloud_init():
        # Запуск веб-интерфейса
        from src.web_interface import main as web_main
        web_main()
    else:
        st.error("Ошибка инициализации системы")


if __name__ == "__main__":
    main()
