"""SQLite persistence for weekly well-being check-ins."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def default_db_path() -> Path:
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "checkins.db"


class CheckinStore:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkins (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    inputs_json TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    risk_index INTEGER NOT NULL,
                    probabilities_json TEXT NOT NULL,
                    drivers_json TEXT NOT NULL,
                    actions_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_checkins_user ON checkins(user_id, created_at)"
            )
            conn.commit()

    def save_checkin(
        self,
        *,
        user_id: str,
        inputs: dict[str, Any],
        risk_level: str,
        risk_index: int,
        probabilities: dict[str, float],
        drivers: list[dict[str, Any]],
        actions: list[str],
        created_at: datetime | None = None,
    ) -> int:
        user_id = user_id.strip()
        if not user_id:
            raise ValueError("user_id is required")
        ts = (created_at or datetime.now(timezone.utc)).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO checkins (
                    user_id, created_at, inputs_json, risk_level, risk_index,
                    probabilities_json, drivers_json, actions_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    ts,
                    json.dumps(inputs),
                    risk_level,
                    risk_index,
                    json.dumps(probabilities),
                    json.dumps(drivers),
                    json.dumps(actions),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_users(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT user_id FROM checkins ORDER BY user_id"
            ).fetchall()
        return [r["user_id"] for r in rows]

    def get_checkins(self, user_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM checkins
                WHERE user_id = ?
                ORDER BY created_at ASC
                """,
                (user_id.strip(),),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "user_id": r["user_id"],
                    "created_at": r["created_at"],
                    "inputs": json.loads(r["inputs_json"]),
                    "risk_level": r["risk_level"],
                    "risk_index": r["risk_index"],
                    "probabilities": json.loads(r["probabilities_json"]),
                    "drivers": json.loads(r["drivers_json"]),
                    "actions": json.loads(r["actions_json"]),
                }
            )
        return out
