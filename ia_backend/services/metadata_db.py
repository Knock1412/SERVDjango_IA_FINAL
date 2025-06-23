import sqlite3
import numpy as np
import json

from pathlib import Path
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity



DB_PATH = Path(__file__).resolve().parent.parent / "metadonnee.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS document_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entreprise TEXT NOT NULL,
            job_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            nb_pages INTEGER,
            nb_blocs INTEGER,
            resume TEXT,
            mots_cles TEXT,
            themes TEXT,
            date_analyse TEXT,
            embedding TEXT  -- JSON string de la liste [float]
        );
        """)
        conn.commit()

def insert_metadata(metadata: dict):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO document_metadata 
        (entreprise, job_id, filename, nb_pages, nb_blocs, resume, mots_cles, themes, date_analyse, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata["entreprise"],
            metadata["job_id"],
            metadata["filename"],
            metadata.get("nb_pages"),
            metadata.get("nb_blocs"),
            metadata.get("resume"),
            ",".join(metadata.get("mots_cles", [])),
            ",".join(metadata.get("themes", [])),
            metadata.get("date_analyse") or datetime.now().isoformat(),
            json.dumps(metadata.get("embedding")) if metadata.get("embedding") else None
        ))
        conn.commit()

def find_nearest_pdf_by_embedding(question_embedding, entreprise, job_id):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
        SELECT filename, embedding FROM document_metadata
        WHERE entreprise = ? AND job_id = ? AND embedding IS NOT NULL
        """, (entreprise, job_id))
        results = cur.fetchall()

    best_sim = 0
    best_filename = None
    for filename, emb_str in results:
        try:
            pdf_embedding = np.array(json.loads(emb_str)).reshape(1, -1)
            sim = cosine_similarity([question_embedding], pdf_embedding)[0][0]
            if sim > best_sim:
                best_sim = sim
                best_filename = filename
        except:
            continue

    return best_filename

def find_documents_by_keyword(keyword, entreprise, job_id):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
        SELECT filename, resume FROM document_metadata
        WHERE entreprise = ? AND job_id = ?
        AND (themes LIKE ? OR mots_cles LIKE ? OR resume LIKE ?)
        """, (entreprise, job_id, f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"))
        return cur.fetchall()