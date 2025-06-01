import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class ContextManager:
    """Менеджер контекста диалогов для поддержания памяти между сообщениями"""

    def __init__(self, max_history_length: int = 10, session_timeout_hours: int = 24):
        self.max_history_length = max_history_length
        self.session_timeout = timedelta(hours=session_timeout_hours)
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.storage_path = Path("../data/sessions")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Загружаем существующие сессии
        self._load_sessions()

    def get_context(self, session_id: str) -> List[Dict[str, Any]]:
        """Получение контекста диалога для сессии"""
        self._cleanup_expired_sessions()

        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "created_at": datetime.now(),
                "last_activity": datetime.now()
            }

        # Обновляем время последней активности
        self.sessions[session_id]["last_activity"] = datetime.now()

        return self.sessions[session_id]["history"]

    def add_interaction(self, session_id: str, query: str, response: str) -> None:
        """Добавление взаимодействия в контекст"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "created_at": datetime.now(),
                "last_activity": datetime.now()
            }

        interaction = {
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }

        self.sessions[session_id]["history"].append(interaction)
        self.sessions[session_id]["last_activity"] = datetime.now()

        # Ограничиваем длину истории
        if len(self.sessions[session_id]["history"]) > self.max_history_length:
            self.sessions[session_id]["history"] = self.sessions[session_id]["history"][-self.max_history_length:]

        # Сохраняем сессию
        self._save_session(session_id)

    def clear_context(self, session_id: str) -> None:
        """Очистка контекста сессии"""
        if session_id in self.sessions:
            self.sessions[session_id]["history"] = []
            self._save_session(session_id)

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Получение информации о сессии"""
        if session_id not in self.sessions:
            return {"exists": False}

        session = self.sessions[session_id]
        return {
            "exists": True,
            "created_at": session["created_at"],
            "last_activity": session["last_activity"],
            "interactions_count": len(session["history"])
        }

    def _cleanup_expired_sessions(self) -> None:
        """Очистка истекших сессий"""
        current_time = datetime.now()
        expired_sessions = []

        for session_id, session_data in self.sessions.items():
            if current_time - session_data["last_activity"] > self.session_timeout:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self.sessions[session_id]
            session_file = self.storage_path / f"session_{session_id}.json"
            if session_file.exists():
                session_file.unlink()

    def _save_session(self, session_id: str) -> None:
        """Сохранение сессии в файл"""
        if session_id not in self.sessions:
            return

        session_file = self.storage_path / f"session_{session_id}.json"
        session_data = self.sessions[session_id].copy()

        # Конвертируем datetime в строки для JSON
        session_data["created_at"] = session_data["created_at"].isoformat()
        session_data["last_activity"] = session_data["last_activity"].isoformat()

        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка при сохранении сессии {session_id}: {e}")

    def _load_sessions(self) -> None:
        """Загрузка сессий из файлов"""
        for session_file in self.storage_path.glob("session_*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)

                session_id = session_file.stem.replace("session_", "")

                # Конвертируем строки обратно в datetime
                session_data["created_at"] = datetime.fromisoformat(session_data["created_at"])
                session_data["last_activity"] = datetime.fromisoformat(session_data["last_activity"])

                self.sessions[session_id] = session_data

            except Exception as e:
                logger.error(f"Ошибка при загрузке сессии из {session_file}: {e}")
