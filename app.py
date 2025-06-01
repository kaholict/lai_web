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


def check_api_key():
    """Проверка наличия API ключа"""
    api_key = None

    # Проверяем Streamlit secrets
    try:
        if hasattr(st, 'secrets') and st.secrets:
            api_key = st.secrets.get("OPENROUTER_API_KEY")
            if api_key:
                return True
    except Exception:
        pass

    # Проверяем переменные окружения
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        return True

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
        try:
            embeddings_manager = EmbeddingsManager(
                model_name=config.get("embeddings.model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                device="cpu"
            )
        except Exception as e:
            logging.error(f"Ошибка инициализации эмбеддингов: {e}")
            return False

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
    setup_logging()

    # Проверяем API ключ
    if not check_api_key():
        st.error("❌ Не задан API ключ OpenRouter")
        st.info("💡 Для работы приложения необходимо:")
        st.markdown("""
        1. **Для Streamlit Cloud**: Добавьте `OPENROUTER_API_KEY` в секреты приложения
        2. **Для локальной разработки**: 
           - Создайте файл `.streamlit/secrets.toml` с содержимым:
           ```
           OPENROUTER_API_KEY = "ваш_api_ключ"
           ```
           - Или установите переменную окружения `OPENROUTER_API_KEY`
        """)
        st.markdown("🔑 Получить API ключ можно на [openrouter.ai/keys](https://openrouter.ai/keys)")
        st.stop()

    # Инициализация
    if cloud_init():
        # Запуск веб-интерфейса
        from src.web_interface import main as web_main
        web_main()
    else:
        st.error("❌ Ошибка инициализации системы")
        st.info("🔧 Попробуйте перезапустить приложение")


if __name__ == "__main__":
    main()
