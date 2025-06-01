#!/usr/bin/env python3

"""
Главный файл для деплоя в Streamlit Cloud
"""

import logging
import sys
import os
import argparse
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


def setup_mode():
    """Режим автоматической настройки системы"""
    try:
        from src.config_manager import ConfigManager
        from src.document_processor import DocumentProcessor
        from src.embeddings_manager import EmbeddingsManager
        from src.vector_store import VectorStore

        print("🚀 Запуск автоматической настройки системы...")

        config = ConfigManager()
        docs_path = Path("data/raw")

        # Создание необходимых папок
        docs_path.mkdir(parents=True, exist_ok=True)

        print("📁 Создание менеджера эмбеддингов...")
        embeddings_manager = EmbeddingsManager(
            model_name=config.get("embeddings.model_name"),
            device="cpu"
        )

        print("🗄️ Инициализация векторного хранилища...")
        vector_store = VectorStore(
            embeddings_manager=embeddings_manager,
            persist_directory=config.get("vector_store.persist_directory")
        )

        if not vector_store.load_vector_store():
            print("📄 Векторное хранилище не найдено, начинаем обработку документов...")
            processor = DocumentProcessor(
                chunk_size=config.get("document_processing.chunk_size"),
                chunk_overlap=config.get("document_processing.chunk_overlap")
            )

            if docs_path.exists() and any(docs_path.iterdir()):
                documents = processor.process_documents(str(docs_path))
                if documents:
                    vector_store.create_vector_store(documents)
                    print(f"✅ Обработано {len(documents)} документов")
                else:
                    print("⚠️ Документы найдены, но не удалось их обработать")
            else:
                print("⚠️ Папка с документами пуста или не найдена")
                print("📝 Создание пустого векторного хранилища...")
                # Создаем пустое хранилище для инициализации
                from langchain.schema import Document
                dummy_doc = Document(page_content="Инициализация системы", metadata={"source": "system"})
                vector_store.create_vector_store([dummy_doc])
        else:
            print("✅ Векторное хранилище успешно загружено")

        print("🎉 Автоматическая настройка завершена успешно!")
        return True

    except Exception as e:
        print(f"❌ Ошибка при автоматической настройке: {e}")
        logging.error(f"Ошибка инициализации: {e}")
        return False


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
                    # Создаем пустое хранилище
                    from langchain.schema import Document
                    dummy_doc = Document(page_content="Инициализация системы", metadata={"source": "system"})
                    vector_store.create_vector_store([dummy_doc])
            else:
                logging.warning("Папка с документами не найдена")
                # Создаем папку и пустое хранилище
                docs_path.mkdir(parents=True, exist_ok=True)
                from langchain.schema import Document
                dummy_doc = Document(page_content="Инициализация системы", metadata={"source": "system"})
                vector_store.create_vector_store([dummy_doc])

        return True

    except Exception as e:
        logging.error(f"Ошибка инициализации: {e}")
        return False


def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='LAI Assistant')
    parser.add_argument('--mode', choices=['setup', 'run'], default='run',
                        help='Режим запуска: setup для настройки, run для запуска веб-интерфейса')

    # Проверяем, запущен ли скрипт из командной строки или через Streamlit
    if len(sys.argv) > 1 and '--mode' in ' '.join(sys.argv):
        args = parser.parse_args()
        setup_logging()

        if args.mode == 'setup':
            success = setup_mode()
            sys.exit(0 if success else 1)

    # Обычный запуск для Streamlit
    setup_logging()

    # Проверяем переменные окружения только если это не setup режим
    if not os.getenv("OPENROUTER_API_KEY") and 'setup' not in sys.argv:
        st.error(
            "❌ Не задан API ключ OpenRouter. Установите переменную окружения OPENROUTER_API_KEY или добавьте ключ в secrets.toml")
        st.info("💡 Для локальной разработки создайте файл .streamlit/secrets.toml с вашим API ключом")
        st.stop()

    # Инициализация
    if cloud_init():
        # Запуск веб-интерфейса
        from src.web_interface import main as web_main
        web_main()
    else:
        st.error("❌ Ошибка инициализации системы")


if __name__ == "__main__":
    main()
