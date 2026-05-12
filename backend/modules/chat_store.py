"""
Chat session persistence with MongoDB primary + in-memory fallback.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    from pymongo import MongoClient
except Exception:  # pragma: no cover - optional dependency safety
    MongoClient = None


class ChatStore:
    def __init__(self) -> None:
        self.mongo_uri = os.getenv("MONGODB_URI")
        self.mongo_db = os.getenv("MONGODB_DB", "nyayaai")
        self._memory_sessions: Dict[str, Dict[str, Any]] = {}
        self._collection = None

        if self.mongo_uri and MongoClient:
            client = MongoClient(self.mongo_uri)
            db = client[self.mongo_db]
            self._collection = db["chat_sessions"]
            self._collection.create_index("sessionId", unique=True)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_session(self, user_id: Optional[str], case_context: Dict[str, Any]) -> Dict[str, Any]:
        session = {
            "sessionId": str(uuid4()),
            "userId": user_id,
            "caseContext": case_context,
            "messages": [],
            "createdAt": self._now_iso(),
            "updatedAt": self._now_iso(),
        }
        if self._collection is not None:
            self._collection.insert_one(session)
        else:
            self._memory_sessions[session["sessionId"]] = session
        return session

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        if self._collection is not None:
            return self._collection.find_one({"sessionId": session_id}, {"_id": 0})
        return self._memory_sessions.get(session_id)

    def append_message(self, session_id: str, role: str, content: str) -> Dict[str, Any]:
        message = {
            "role": role,
            "content": content,
            "timestamp": self._now_iso(),
        }
        if self._collection is not None:
            self._collection.update_one(
                {"sessionId": session_id},
                {
                    "$push": {"messages": message},
                    "$set": {"updatedAt": self._now_iso()},
                },
            )
            return message

        session = self._memory_sessions[session_id]
        session["messages"].append(message)
        session["updatedAt"] = self._now_iso()
        return message

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        session = self.get_session(session_id)
        if not session:
            return []
        return session.get("messages", [])

    def update_case_context(self, session_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        session = self.get_session(session_id)
        if not session:
            return None

        merged = {**session.get("caseContext", {}), **(updates or {})}
        if self._collection is not None:
            self._collection.update_one(
                {"sessionId": session_id},
                {"$set": {"caseContext": merged, "updatedAt": self._now_iso()}},
            )
            fresh = self.get_session(session_id)
            return fresh.get("caseContext", {}) if fresh else merged

        session["caseContext"] = merged
        session["updatedAt"] = self._now_iso()
        return merged
