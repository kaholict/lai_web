import streamlit as st
import uuid
import logging
from src.ai_assistant import AIAssistant
from src.config_manager import ConfigManager
from src.embeddings_manager import EmbeddingsManager
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


def apply_cloud_style():
    st.markdown("""
    <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        .stChatInputContainer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background: white;
            padding: 1rem;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)


def init_assistant():
    if 'assistant' not in st.session_state:
        config = ConfigManager()
        embeddings = EmbeddingsManager(
            model_name=config.get("embeddings.model_name"),
            device="cpu"
        )
        vector_store = VectorStore(
            embeddings_manager=embeddings,
            persist_directory=config.get("vector_store.persist_directory")
        )
        vector_store.load_vector_store()
        st.session_state.assistant = AIAssistant(vector_store, config.config)


def main():
    st.set_page_config(
        page_title="LAI Assistant",
        page_icon="🤖",
        layout="centered"
    )
    apply_cloud_style()

    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    init_assistant()

    st.title("LAI - Нейросетевой ассистент")

    if prompt := st.chat_input("Введите ваш вопрос..."):
        response = st.session_state.assistant.generate_response(
            prompt,
            st.session_state.session_id
        )
        st.chat_message("user").markdown(prompt)
        with st.chat_message("assistant"):
            st.markdown(response["response"])
            if response.get("sources"):
                with st.expander("Источники"):
                    st.write("\n".join(response["sources"]))


if __name__ == "__main__":
    main()
