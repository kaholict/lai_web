import streamlit as st
import uuid
import logging
from datetime import datetime
from pathlib import Path
import yaml
from src.ai_assistant import AIAssistant
from src.document_processor import DocumentProcessor
from src.embeddings_manager import EmbeddingsManager
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Загрузка конфигурации"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Ошибка загрузки конфигурации: {e}")
        return {}


def init_assistant() -> AIAssistant:
    """Инициализация ассистента"""
    if 'assistant' not in st.session_state:
        config = load_config()

        # Инициализация компонентов
        embeddings_manager = EmbeddingsManager(
            model_name=config["embeddings"]["model_name"],
            device="cpu"
        )

        vector_store = VectorStore(
            embeddings_manager=embeddings_manager,
            persist_directory=config["vector_store"]["persist_directory"]
        )

        # Загрузка векторного хранилища
        if not vector_store.load_vector_store():
            st.error("Не удалось загрузить векторное хранилище. Убедитесь, что документы обработаны.")
            st.stop()

        st.session_state.assistant = AIAssistant(vector_store, config)

    return st.session_state.assistant


def init_session():
    """Инициализация сессии"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'first_visit' not in st.session_state:
        st.session_state.first_visit = True


def display_chat_message(message: dict):
    """Отображение сообщения в чате"""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Показываем источники для ответов ассистента
        if message["role"] == "assistant" and "sources" in message:
            if message["sources"]:
                with st.expander("📚 Источники информации"):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"{i}. {source}")


def main():
    """Главная функция веб-интерфейса"""
    # Настройка страницы согласно дизайну
    st.set_page_config(
        page_title="LAI - ИИ-ассистент для сотрудников",
        page_icon="👨‍💻",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Инициализация
    init_session()
    assistant = init_assistant()

    # Боковая панель с информацией
    with st.sidebar:
        st.title("LAI Assistant")
        st.markdown("**Ваш ИИ-помощник по продажам**")

        # Информация о сессии
        st.markdown("---")
        st.markdown("### 📊 Информация о сессии")
        session_info = assistant.context_manager.get_session_info(st.session_state.session_id)
        if session_info["exists"]:
            st.write(f"**Сообщений:** {session_info['interactions_count']}")
            st.write(f"**Начало:** {session_info['created_at'].strftime('%H:%M')}")

        # Кнопка очистки контекста
        if st.button("🗑️ Очистить историю"):
            assistant.context_manager.clear_context(st.session_state.session_id)
            st.session_state.messages = []
            st.rerun()

        # Справочная информация
        st.markdown("---")
        st.markdown("### ℹ️ Справка")
        st.markdown("""
        **Возможности ассистента:**
        - Ответы на вопросы по продажам
        - Поиск в корпоративной документации
        - Память контекста диалога
        - Указание источников информации

        **Совет:** Задавайте конкретные вопросы, связанные с вашей работой в отделе продаж.
        """)

    # Основная область чата
    st.title("💬 Чат с ИИ-ассистентом")

    # Приветственное сообщение
    if st.session_state.first_visit:
        with st.chat_message("assistant"):
            st.markdown("""
            👋 Привет! Я LAI — ваш ИИ-ассистент для работы в отделе продаж.

            Я могу помочь вам с:
            - Вопросами по продуктам и услугам
            - Процедурами и регламентами продаж
            - Работой с клиентами
            - Поиском информации в корпоративных документах

            Задайте любой вопрос, связанный с вашей работой!
            """)
        st.session_state.first_visit = False

    # Отображение истории сообщений
    for message in st.session_state.messages:
        display_chat_message(message)

    # Поле ввода сообщения
    if prompt := st.chat_input("Введите ваш вопрос..."):
        # Добавляем сообщение пользователя
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        display_chat_message(user_message)

        # Генерируем ответ ассистента
        with st.chat_message("assistant"):
            with st.spinner("Обрабатываю ваш запрос..."):
                response_data = assistant.generate_response(prompt, st.session_state.session_id)

            # Отображаем ответ
            st.markdown(response_data["response"])

            # Показываем источники
            if response_data["sources"]:
                with st.expander("📚 Источники информации"):
                    for i, source in enumerate(response_data["sources"], 1):
                        st.write(f"{i}. {source}")

            # Предупреждение для вопросов не по теме
            if not response_data["is_sales_related"]:
                st.warning("⚠️ Обратите внимание: этот вопрос не относится к основной деятельности отдела продаж.")

        # Добавляем ответ в историю
        assistant_message = {
            "role": "assistant",
            "content": response_data["response"],
            "sources": response_data["sources"]
        }
        st.session_state.messages.append(assistant_message)


if __name__ == "__main__":
    main()
