import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "assistant_chat.db")
DB_PATH = os.path.abspath(DB_PATH)

def get_connection():
    return sqlite3.connect(DB_PATH)

def create_table():
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            question TEXT,
            answer TEXT,
            blocks_used TEXT,
            job_id TEXT,
            user_id TEXT
        )
        """)
        conn.commit()

# Appelle ça une fois au démarrage
create_table()

def save_interaction(session_id, question, answer, blocks_used, job_id, user_id=None):
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("""
        INSERT INTO chat_sessions (session_id, question, answer, blocks_used, job_id, user_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            question,
            answer,
            json.dumps(blocks_used) if blocks_used is not None else None,
            job_id,
            user_id
        ))
        conn.commit()

def get_session_history(session_id):
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("""
        SELECT question, answer, blocks_used, timestamp FROM chat_sessions
        WHERE session_id = ?
        ORDER BY timestamp ASC
        """, (session_id,))
        rows = c.fetchall()
    # Parse blocks_used
    return [{
        "question": q,
        "answer": a,
        "blocks_used": json.loads(b) if b else [],
        "timestamp": t
    } for (q, a, b, t) in rows]

# (Optionnel) Supprimer l’historique d’une session
def clear_session(session_id):
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        conn.commit()
