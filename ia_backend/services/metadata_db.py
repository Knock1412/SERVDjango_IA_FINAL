import sqlite3
import numpy as np
import json
import faiss
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
from sentence_transformers.util import cos_sim
import logging

logger = logging.getLogger(__name__)
DB_PATH = Path(__file__).resolve().parent.parent / "metadonnee.db"

# --- Initialisation FAISS ---
EMBEDDING_DIM = 384  # Dimension des embeddings (adaptez selon votre modèle)
faiss_index = None
index_last_update = None

def get_connection():
    """Connection pool pour meilleures performances"""
    return sqlite3.connect(DB_PATH, timeout=10)

def init_db():
    """Initialisation avec index et optimisations"""
    with get_connection() as conn:
        # Activation des optimisations SQLite
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-10000")  # 10MB cache
        
        conn.execute("""
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
            embedding BLOB,  -- Stockage binaire pour FAISS
            embedding_norm REAL,
            UNIQUE(entreprise, job_id, filename)
        );
        """)
        
        # Index pour recherches textuelles
        conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS document_search 
        USING FTS5(entreprise, job_id, filename, resume, mots_cles, themes);
        """)
        
        conn.commit()
    init_faiss_index()

def init_faiss_index():
    """Charge ou crée l'index FAISS"""
    global faiss_index, index_last_update
    
    try:
        # Vérifie si l'index existe déjà
        faiss_index = faiss.read_index(str(DB_PATH.parent / "pdf_embeddings.faiss"))
        index_last_update = datetime.now()
        logger.info("Index FAISS chargé depuis le disque")
    except:
        # Crée un nouvel index
        faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        logger.info("Nouvel index FAISS créé")

def update_faiss_index():
    """Met à jour l'index FAISS depuis la base"""
    global faiss_index, index_last_update
    
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, embedding FROM document_metadata WHERE embedding IS NOT NULL")
        embeddings = []
        ids = []
        
        for id, emb_blob in cur.fetchall():
            try:
                emb = np.frombuffer(emb_blob, dtype='float32')
                embeddings.append(emb)
                ids.append(id)
            except Exception as e:
                logger.error(f"Erreur lecture embedding ID {id}: {e}")
        
        if embeddings:
            embeddings = np.vstack(embeddings).astype('float32')
            ids = np.array(ids).astype('int64')
            
            # Crée un nouvel index
            new_index = faiss.IndexFlatIP(EMBEDDING_DIM)
            new_index.add_with_ids(embeddings, ids)
            
            # Sauvegarde l'index
            faiss.write_index(new_index, str(DB_PATH.parent / "pdf_embeddings.faiss"))
            faiss_index = new_index
            index_last_update = datetime.now()
            logger.info(f"Index FAISS mis à jour avec {len(ids)} embeddings")

def insert_metadata(metadata: dict):
    """Insertion optimisée avec mise à jour de l'index"""
    with get_connection() as conn:
        # Conversion de l'embedding en binaire
        embedding = None
        embedding_norm = None
        if metadata.get("embedding"):
            emb_array = np.array(metadata["embedding"], dtype='float32')
            embedding = emb_array.tobytes()
            embedding_norm = float(np.linalg.norm(emb_array))
        
        # Insertion ou mise à jour
        conn.execute("""
        INSERT OR REPLACE INTO document_metadata
        (entreprise, job_id, filename, nb_pages, nb_blocs, resume, mots_cles, themes, date_analyse, embedding, embedding_norm)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            embedding,
            embedding_norm
        ))
        
        # Mise à jour de la table FTS
        conn.execute("""
        INSERT OR REPLACE INTO document_search
        (rowid, entreprise, job_id, filename, resume, mots_cles, themes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            conn.execute("SELECT last_insert_rowid()").fetchone()[0],
            metadata["entreprise"],
            metadata["job_id"],
            metadata["filename"],
            metadata.get("resume", ""),
            ",".join(metadata.get("mots_cles", [])),
            ",".join(metadata.get("themes", []))
        ))
        
        conn.commit()
    
    # Mise à jour asynchrone de l'index FAISS
    update_faiss_index()

def find_nearest_pdf_by_embedding(question_embedding: List[float], entreprise: str, job_id: str, top_k: int = 3) -> Optional[List[Tuple[str, float]]]:
    """
    Recherche les PDFs les plus proches en utilisant FAISS.
    Retourne une liste de (filename, score) triée par pertinence.
    """
    if faiss_index is None:
        init_faiss_index()
    
    try:
        # Convertit l'embedding en format FAISS
        query_emb = np.array([question_embedding], dtype='float32')
        
        # Recherche dans FAISS
        D, I = faiss_index.search(query_emb, k=top_k)
        
        # Récupère les métadonnées correspondantes
        with get_connection() as conn:
            cur = conn.cursor()
            placeholders = ",".join("?" * len(I[0]))
            cur.execute(f"""
            SELECT filename, embedding_norm 
            FROM document_metadata 
            WHERE id IN ({placeholders}) AND entreprise = ? AND job_id = ?
            """, (*I[0], entreprise, job_id))
            
            results = cur.fetchall()
            if not results:
                return None
                
            # Calcule les scores normalisés
            scores = []
            for (filename, norm), score in zip(results, D[0]):
                if norm > 0:
                    normalized_score = score / norm
                    scores.append((filename, float(normalized_score)))
            
            return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
            
    except Exception as e:
        logger.error(f"Erreur recherche FAISS: {e}")
        return None

def find_documents_by_keyword_semantic(
    question: str, 
    entreprise: str, 
    job_id: str, 
    encode_text_fn, 
    top_k: int = 5
) -> List[Tuple[str, str]]:
    """
    Recherche hybride : FTS (BM25) + Similarité sémantique sur mots_cles et themes.
    encode_text_fn doit être une fonction qui retourne l'embedding d'un texte.
    Retourne une liste de (filename, extrait/thèmes).
    """
    # 1. Recherche FTS classique
    results_fts = find_documents_by_keyword(question, entreprise, job_id, top_k=top_k)

    # 2. Recherche sémantique sur les mots_cles et themes encodés
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            SELECT filename, mots_cles, themes FROM document_metadata
            WHERE entreprise = ? AND job_id = ?
            """, (entreprise, job_id))
            
            question_emb = encode_text_fn(question)
            scored = []
            for filename, mots_cles, themes in cur.fetchall():
                text = ((mots_cles or "") + " " + (themes or "")).strip()
                if not text:
                    continue
                sim = cos_sim(encode_text_fn(text), question_emb)[0][0].item()
                scored.append((filename, text[:200], sim))
            
            top_semantic = sorted(scored, key=lambda x: x[2], reverse=True)[:top_k]
    except Exception as e:
        logger.error(f"Erreur recherche sémantique: {e}")
        top_semantic = []

    # 3. Fusion des deux sources (déduplique par filename)
    combined = {f: r for f, r in results_fts}
    for filename, snippet, _ in top_semantic:
        if filename not in combined:
            combined[filename] = snippet
    
    return list(combined.items())[:top_k]

def find_documents_by_keyword(keyword: str, entreprise: str, job_id: str, top_k: int = 5) -> List[Tuple[str, str]]:
    """
    Recherche full-text optimisée avec ranking BM25.
    Retourne une liste de (filename, resume_extract).
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            SELECT filename, snippet(document_search, 2, '', '', '...', 16) as extract
            FROM document_search
            WHERE entreprise = ? AND job_id = ? 
            AND document_search MATCH ?
            ORDER BY bm25(document_search, 0.0, 0.5, 1.0) DESC
            LIMIT ?
            """, (entreprise, job_id, f"{keyword}*", top_k))
            
            return cur.fetchall()
            
    except Exception as e:
        logger.error(f"Erreur recherche full-text: {e}")
        return []
