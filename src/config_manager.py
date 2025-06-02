import yaml
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st

logger = logging.getLogger(__name__)


class ConfigManager:
    """Менеджер конфигурации с проверкой настроек и поддержкой secrets"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Загрузка конфигурации из файла"""
        try:
            if not self.config_path.exists():
                logger.info(f"Файл конфигурации не найден: {self.config_path}, используем настройки по умолчанию")
                return self._get_default_config()

            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Загружаем секреты из Streamlit secrets или переменных окружения
            self._load_secrets(config)

            logger.info(f"Конфигурация загружена из {self.config_path}")
            return config

        except Exception as e:
            logger.warning(f"Ошибка при загрузке конфигурации: {e}, используем настройки по умолчанию")
            config = self._get_default_config()
            self._load_secrets(config)
            return config

    def _load_secrets(self, config: Dict[str, Any]) -> None:
        """Загрузка секретов из Streamlit secrets или переменных окружения"""
        try:
            api_key = None

            # Пытаемся загрузить из Streamlit secrets
            try:
                if hasattr(st, 'secrets') and st.secrets:
                    api_key = st.secrets.get("OPENROUTER_API_KEY")
                    if api_key:
                        logger.info("API ключ загружен из Streamlit secrets")
            except Exception as e:
                logger.debug(f"Не удалось загрузить из Streamlit secrets: {e}")

            # Если не найден в secrets, пытаемся загрузить из переменных окружения
            if not api_key:
                api_key = os.getenv("OPENROUTER_API_KEY")
                if api_key:
                    logger.info("API ключ загружен из переменных окружения")

            # Устанавливаем API ключ в конфигурацию
            if api_key:
                config["openrouter"]["api_key"] = api_key
            else:
                logger.warning("API ключ не найден ни в secrets, ни в переменных окружения")
                # Не используем хардкоженный ключ из config.yaml для безопасности
                config["openrouter"]["api_key"] = ""

        except Exception as e:
            logger.error(f"Ошибка при загрузке секретов: {e}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Получение конфигурации по умолчанию"""
        return {
            "openrouter": {
                "api_key": "",  # Будет загружен из secrets или переменных окружения
                "base_url": "https://openrouter.ai/api/v1",
                "model": "meta-llama/llama-3.3-70b-instruct:free",
                "temperature": 0.7
            },
            "embeddings": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cpu"
            },
            "document_processing": {
                "chunk_size": 1000,
                "chunk_overlap": 200
            },
            "vector_store": {
                "index_name": "sales_assistant",
                "persist_directory": "./data/vector_store"
            },
            "assistant": {
                "max_context_length": 10,
                "session_timeout_hours": 24,
                "max_response_tokens": 1000
            }
        }

    def _validate_config(self) -> None:
        """Валидация конфигурации"""
        required_sections = ["openrouter", "embeddings", "vector_store"]
        for section in required_sections:
            if section not in self.config:
                logger.error(f"Отсутствует обязательная секция конфигурации: {section}")
                self.config[section] = self._get_default_config()[section]

        # Проверка API ключа
        api_key = self.config["openrouter"]["api_key"]
        if not api_key:
            logger.error("API ключ OpenRouter не задан")
            raise ValueError(
                "API ключ OpenRouter не найден. Установите переменную окружения OPENROUTER_API_KEY или добавьте ключ в secrets.toml")
        elif not api_key.startswith("sk-or-"):
            logger.warning(f"API ключ может быть некорректным (не начинается с 'sk-or-'): {api_key[:10]}...")

        # Создание директорий
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Создание необходимых директорий"""
        directories = [
            "data/raw",
            "data/processed",
            "data/sessions",
            self.config["vector_store"]["persist_directory"],
        ]

        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Не удалось создать директорию {directory}: {e}")

    @property
    def config(self) -> Dict[str, Any]:
        """Получение конфигурации"""
        return self._config

    @config.setter
    def config(self, value: Dict[str, Any]) -> None:
        """Установка конфигурации"""
        self._config = value

    def get(self, key_path: str, default: Any = None) -> Any:
        """Получение значения конфигурации по пути"""
        keys = key_path.split('.')
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
