#!/usr/bin/env python3
"""
Главный файл ИИ-ассистента для сотрудников
"""

import argparse
import logging
import sys
from pathlib import Path
import streamlit.web.cli as stcli

# Добавляем текущую директорию в путь
sys.path.append(str(Path(__file__).parent))

from src.config_manager import ConfigManager
from src.document_processor import DocumentProcessor
from src.embeddings_manager import EmbeddingsManager
from src.vector_store import VectorStore


def setup_logging():
    """Настройка системы логирования"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/assistant.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def process_documents(config_manager: ConfigManager, documents_path: str) -> bool:
    """Обработка документов и создание векторного хранилища"""
    logging.info("Начинаем обработку документов")

    try:
        # Инициализация компонентов
        doc_processor = DocumentProcessor(
            chunk_size=config_manager.get("document_processing.chunk_size", 1000),
            chunk_overlap=config_manager.get("document_processing.chunk_overlap", 200)
        )

        embeddings_manager = EmbeddingsManager(
            model_name=config_manager.get("embeddings.model_name"),
            device=config_manager.get("embeddings.device", "cpu")
        )

        vector_store = VectorStore(
            embeddings_manager=embeddings_manager,
            persist_directory=config_manager.get("vector_store.persist_directory")
        )

        # Обработка документов
        documents = doc_processor.process_documents(documents_path)
        if not documents:
            logging.error("Не найдено документов для обработки")
            return False

        # Создание векторного хранилища
        vector_store.create_vector_store(documents)
        logging.info(f"Успешно обработано {len(documents)} фрагментов документов")
        return True

    except Exception as e:
        logging.error(f"Ошибка при обработке документов: {e}")
        return False


def run_web_interface():
    """Запуск веб-интерфейса"""
    logging.info("Запуск веб-интерфейса")

    # Запуск Streamlit приложения
    sys.argv = ["streamlit", "run", "src/web_interface.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
    stcli.main()


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description="ИИ-ассистент для сотрудников")
    parser.add_argument("--mode", choices=["web", "process", "setup"],
                        default="web", help="Режим работы")
    parser.add_argument("--documents", default="data/raw",
                        help="Путь к папке с документами")
    parser.add_argument("--config", default="config.yaml",
                        help="Путь к файлу конфигурации")

    args = parser.parse_args()

    # Настройка логирования
    setup_logging()

    try:
        # Загрузка конфигурации
        config_manager = ConfigManager(args.config)

        if args.mode == "setup":
            print("🔧 Настройка проекта...")

            # Проверка наличия документов
            documents_path = Path(args.documents)
            if not documents_path.exists() or not any(documents_path.iterdir()):
                print(f"❌ Папка с документами пуста или не существует: {documents_path}")
                print("Скопируйте ваши PDF и DOCX файлы в папку data/raw/")
                return 1

            # Обработка документов
            print("📄 Обработка документов...")
            if process_documents(config_manager, args.documents):
                print("✅ Документы успешно обработаны")
                print("🚀 Теперь можно запустить веб-интерфейс: python app.py --mode web")
                return 0
            else:
                print("❌ Ошибка при обработке документов")
                return 1

        elif args.mode == "process":
            print("📄 Обработка документов...")
            if process_documents(config_manager, args.documents):
                print("✅ Документы успешно обработаны")
                return 0
            else:
                print("❌ Ошибка при обработке документов")
                return 1

        elif args.mode == "web":
            # Проверка готовности системы
            vector_store_path = Path(config_manager.get("vector_store.persist_directory"))
            if not (vector_store_path / "index.faiss").exists():
                print("❌ Векторное хранилище не найдено.")
                print("Сначала запустите: python app.py --mode setup")
                return 1

            print("🚀 Запуск веб-интерфейса...")
            print("📝 Интерфейс будет доступен по адресу: http://localhost:8501")
            run_web_interface()

        return 0

    except Exception as e:
        logging.error(f"Критическая ошибка: {e}")
        print(f"❌ Критическая ошибка: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
