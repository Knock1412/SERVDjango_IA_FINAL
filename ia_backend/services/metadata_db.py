import sqlite3
import numpy as np
import json
import faiss
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
from sentence_transformers.util import cos_sim
import logging

# Configuration du logger avec un format clair et des niveaux appropriés
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

DB_PATH = Path(__file__).resolve().parent.parent / "metadonnee.db"

# --- Initialisation FAISS ---
EMBEDDING_DIM = 384  # Dimension des embeddings utilisés
faiss_index = None   # Index FAISS global
index_last_update = None  # Timestamp de la dernière mise à jour

def get_connection():
    """Obtient une connexion à la base de données SQLite avec timeout."""
    return sqlite3.connect(DB_PATH, timeout=10)

def init_db():
    """Initialise la base de données avec les tables nécessaires et optimise les performances."""
    logger.info("Initialisation de la base de données...")
    with get_connection() as conn:
        # Optimisations SQLite
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-10000")  # Cache de 10MB

        # Table principale pour les métadonnées
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
            embedding BLOB,
            embedding_norm REAL,
            UNIQUE(entreprise, job_id, filename)
        );
        """)

        # Table de recherche full-text
        conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS document_search 
        USING FTS5(entreprise, job_id, filename, resume, mots_cles, themes);
        """)

        conn.commit()
    logger.info("Base de données initialisée avec succès")
    init_faiss_index()

def init_faiss_index():
    """Charge l'index FAISS depuis le disque ou crée un nouvel index si nécessaire."""
    global faiss_index, index_last_update
    try:
        index_path = DB_PATH.parent / "pdf_embeddings.faiss"
        faiss_index = faiss.read_index(str(index_path))
        index_last_update = datetime.now()
        logger.info(f"Index FAISS chargé depuis {index_path} - {faiss_index.ntotal} embeddings")
    except Exception as e:
        logger.warning(f"Impossible de charger l'index FAISS, création d'un nouvel index: {str(e)}")
        faiss_index = faiss.IndexIDMap(faiss.IndexFlatIP(EMBEDDING_DIM))
        logger.info("Nouvel index FAISS créé (vide)")

def update_faiss_index():
    """Met à jour l'index FAISS avec les nouveaux embeddings de la base."""
    global faiss_index, index_last_update
    logger.info("Mise à jour de l'index FAISS...")

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
                logger.error(f"Erreur de décodage de l'embedding ID {id}: {str(e)}")

        if embeddings:
            embeddings = np.vstack(embeddings).astype('float32')
            ids = np.array(ids).astype('int64')
            logger.info(f"Chargement de {len(ids)} embeddings pour mise à jour FAISS")

            base_index = faiss.IndexFlatIP(EMBEDDING_DIM)
            new_index = faiss.IndexIDMap(base_index)
            new_index.add_with_ids(embeddings, ids)

            index_path = DB_PATH.parent / "pdf_embeddings.faiss"
            faiss.write_index(new_index, str(index_path))
            faiss_index = new_index
            index_last_update = datetime.now()
            logger.info(f"Index FAISS mis à jour avec succès - {len(ids)} embeddings - sauvegardé dans {index_path}")
        else:
            logger.warning("Aucun embedding trouvé pour mettre à jour l'index FAISS")

def insert_metadata(metadata: dict):
    """Insère ou met à jour les métadonnées d'un document dans la base."""
    logger.info(f"Insertion/mise à jour des métadonnées pour {metadata['entreprise']}/{metadata['job_id']}/{metadata['filename']}")
    
    with get_connection() as conn:
        # Préparation de l'embedding si présent
        embedding = None
        embedding_norm = None
        if metadata.get("embedding"):
            try:
                emb_array = np.array(metadata["embedding"], dtype='float32')
                embedding = emb_array.tobytes()
                embedding_norm = float(np.linalg.norm(emb_array))
                logger.debug(f"Embedding préparé - norme: {embedding_norm:.4f}")
            except Exception as e:
                logger.error(f"Erreur de préparation de l'embedding: {str(e)}")

        # Insertion dans la table principale
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

        # Insertion dans la table de recherche full-text
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
        logger.info("Métadonnées insérées/mises à jour avec succès")

    # Mise à jour de l'index FAISS après insertion
    update_faiss_index()
    logger.debug("Rechargement de l'index FAISS pour vérification")
    init_faiss_index()

def find_nearest_pdf_by_embedding(question_embedding: List[float], entreprise: str, job_id: str, top_k: int = 3) -> Optional[List[Tuple[str, float]]]:
    """Trouve les documents les plus proches d'un embedding donné en utilisant FAISS."""
    logger.info(f"Recherche des documents similaires pour {entreprise}/{job_id} - top_k={top_k}")
    
    if faiss_index is None:
        logger.warning("Index FAISS non initialisé, tentative d'initialisation...")
        init_faiss_index()

    try:
        query_emb = np.array([question_embedding], dtype='float32')
        logger.debug(f"Embedding de requête converti - shape: {query_emb.shape}")

        # Recherche FAISS
        D, I = faiss_index.search(query_emb, k=top_k)
        logger.debug(f"Résultats FAISS - distances: {D} - indices: {I}")

        # Filtrage des IDs valides (exclure -1)
        valid_ids = [int(i) for i in I[0] if i != -1]
        if not valid_ids:
            logger.warning("⚠️ Aucun index FAISS valide trouvé")
            return None

        with get_connection() as conn:
            cur = conn.cursor()
            placeholders = ",".join("?" * len(valid_ids))
            query = f"""
            SELECT filename, embedding_norm 
            FROM document_metadata 
            WHERE id IN ({placeholders}) AND entreprise = ? AND job_id = ?
            """
            logger.debug(f"Exécution de la requête SQL: {query} - params: {valid_ids}, {entreprise}, {job_id}")
            
            cur.execute(query, (*valid_ids, entreprise, job_id))
            results = cur.fetchall()
            logger.debug(f"{len(results)} documents trouvés dans la base")

            if not results:
                logger.warning("Aucun document correspondant trouvé dans la base")
                return None

            # Calcul des scores normalisés
            scores = []
            for (filename, norm), score in zip(results, D[0][:len(results)]):
                if norm > 0:
                    normalized_score = score / norm
                    scores.append((filename, float(normalized_score)))
                    logger.debug(f"Document {filename} - score normalisé: {normalized_score:.4f}")

            # Tri et retour des meilleurs résultats
            top_results = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
            logger.info(f"Recherche terminée - {len(top_results)} résultats retournés")
            return top_results

    except Exception as e:
        logger.error(f"Erreur lors de la recherche par embedding: {str(e)}", exc_info=True)
        return None


def find_documents_by_keyword_semantic(question: str, entreprise: str, job_id: str, encode_text_fn, top_k: int = 5) -> List[Tuple[str, str]]:
    """Recherche combinée par mots-clés et similarité sémantique."""
    logger.info(f"Recherche combinée pour '{question}' dans {entreprise}/{job_id}")
    
    # Recherche full-text d'abord
    results_fts = find_documents_by_keyword(question, entreprise, job_id, top_k=top_k)
    logger.debug(f"Résultats FTS initiaux: {results_fts}")

    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            SELECT filename, mots_cles, themes FROM document_metadata
            WHERE entreprise = ? AND job_id = ?
            """, (entreprise, job_id))

            # Encodage sémantique de la question
            question_emb = encode_text_fn(question)
            logger.debug(f"Embedding sémantique généré - dimension: {len(question_emb)}")

            scored = []
            for filename, mots_cles, themes in cur.fetchall():
                text = ((mots_cles or "") + " " + (themes or "")).strip()
                if not text:
                    continue
                
                # Calcul de similarité cosinus
                text_emb = encode_text_fn(text)
                score = cos_sim(text_emb, question_emb)[0][0].item()
                scored.append((filename, text[:200], score))
                logger.debug(f"Similarité avec {filename}: {score:.4f} - texte: {text[:60]}...")

            top_semantic = sorted(scored, key=lambda x: x[2], reverse=True)[:top_k]
            logger.debug(f"Top {len(top_semantic)} résultats sémantiques")

    except Exception as e:
        logger.error(f"Erreur lors de la recherche sémantique: {str(e)}", exc_info=True)
        top_semantic = []

    # Combinaison des résultats FTS et sémantiques
    combined = {f: r for f, r in results_fts}
    for filename, snippet, _ in top_semantic:
        if filename not in combined:
            combined[filename] = snippet

    final_results = list(combined.items())[:top_k]
    logger.info(f"Recherche combinée terminée - {len(final_results)} résultats retournés")
    return final_results

def find_documents_by_keyword(keyword: str, entreprise: str, job_id: str, top_k: int = 5) -> List[Tuple[str, str]]:
    """Recherche full-text simple par mots-clés."""
    logger.info(f"Recherche FTS pour '{keyword}' dans {entreprise}/{job_id}")
    
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            query = """
            SELECT filename, snippet(document_search, 2, '', '', '...', 16) as extract
            FROM document_search
            WHERE entreprise = ? AND job_id = ? 
            AND document_search MATCH ?
            ORDER BY bm25(document_search, 0.0, 0.5, 1.0) DESC
            LIMIT ?
            """
            logger.debug(f"Exécution de la requête FTS: {query} - params: {entreprise}, {job_id}, {keyword}*, {top_k}")
            
            cur.execute(query, (entreprise, job_id, f"{keyword}*", top_k))
            results = cur.fetchall()
            logger.debug(f"{len(results)} résultats FTS trouvés")
            return results

    except Exception as e:
        logger.error(f"Erreur lors de la recherche FTS: {str(e)}", exc_info=True)
        return []