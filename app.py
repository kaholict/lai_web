#!/usr/bin/env python3
"""
Главный файл для деплоя в Streamlit Cloud
"""

import logging
import sys
from pathlib import Path
import streamlit.web.cli as stcli
from src.config_manager import ConfigManager
from src.document_processor import DocumentProcessor
from src.embeddings_manager import EmbeddingsManager
from src.vector_store import VectorStore


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def cloud_init():
    """Автоматическая инициализация для Streamlit Cloud"""
    config = ConfigManager()
    docs_path = Path("data/raw")

    # Проверка и создание векторного хранилища
    vector_store = VectorStore(
        embeddings_manager=EmbeddingsManager(
            model_name=config.get("embeddings.model_name"),
            device="cpu"
        ),
        persist_directory=config.get("vector_store.persist_directory")
    )

    if not vector_store.load_vector_store():
        logging.info("Векторное хранилище не найдено, начинаем обработку документов")
        processor = DocumentProcessor(
            chunk_size=config.get("document_processing.chunk_size"),
            chunk_overlap=config.get("document_processing.chunk_overlap")
        )
        documents = processor.process_documents(str(docs_path))
        vector_store.create_vector_store(documents)
        logging.info(f"Обработано {len(documents)} документов")


def main():
    setup_logging()
    cloud_init()

    # Запуск веб-интерфейса
    sys.argv = [
        "streamlit", "run", "src/web_interface.py",
        "--server.address", "0.0.0.0",
        "--server.port", "8501"
    ]
    stcli.main()


if __name__ == "__main__":
    main()
