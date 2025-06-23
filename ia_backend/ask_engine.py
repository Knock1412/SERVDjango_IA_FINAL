import os
import json
import logging
import uuid
import torch
import numpy as np
import time
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util, CrossEncoder

# --- Imports spécifiques backend IA ---
from ia_backend.services.ollama_gateway import generate_ollama
from ia_backend.services.metadata_db import (
    find_nearest_pdf_by_embedding,
    find_documents_by_keyword
)
from ia_backend.services.chat_memory import save_interaction

# --- Initialisation logging et modèles ---
logger = logging.getLogger(__name__)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ---------------------------------------------------------------------------
#    Utilitaire : classification LLM “générale/précise” + score de confiance
# ---------------------------------------------------------------------------
def classify_question_with_score(question: str) -> Tuple[str, float]:
    """
    Utilise le LLM (Ollama) pour classifier une question : 'générale' ou 'précise'
    Retourne aussi un score de confiance [0,1].
    """
    prompt = f"""[INST]
Tu es un assistant IA. Ton rôle est de classer une question utilisateur en deux catégories :
- "générale" : la question demande une vue d’ensemble, une liste de documents ou un thème transversal.
- "précise" : la question cherche une information précise à l’intérieur d’un seul document.

QUESTION :
{question}

Renvoie uniquement une réponse JSON valide comme ceci :
{{
  "type": "générale" ou "précise",
  "confiance": nombre entre 0 et 1
}}
[/INST]"""

    try:
        response = generate_ollama(prompt=prompt, num_predict=120).strip()
        data = json.loads(response)
        q_type = data.get("type", "").lower()
        confiance = float(data.get("confiance", 0.0))
        if q_type in ["générale", "precise", "précise"]:
            return (q_type, round(confiance, 3))
    except Exception as e:
        logger.error(f"Erreur classification LLM : {e}")
    return ("précise", 0.0)  # fallback : on considère précise


# ---------------------------------------------------------------------------
#     Utilitaire : charger tous les blocs d’un job_id (fichier par PDF)
# ---------------------------------------------------------------------------
def load_all_blocks(entreprise: str, job_id: str) -> List[Tuple[str, Dict]]:
    folder_path = os.path.join("cache_json", "save_summaryblocks", entreprise, job_id)
    logger.debug(f"Chargement des blocs depuis : {folder_path}")
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Dossier introuvable : {folder_path}")

    blocks: List[Tuple[str, Dict]] = []
    for filename in sorted(os.listdir(folder_path)):
        if not filename.startswith("bloc_") or not filename.endswith(".json"):
            continue
        full_path = os.path.join(folder_path, filename)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            summary = data.get("summary", "")
            embedding = data.get("embedding")
            if embedding is None:
                logger.warning(f"Pas d'embedding pour {filename}, bloc ignoré.")
                continue
            meta = {
                "source": filename,
                "score": data.get("score", 0),
                "translated": data.get("translated", False),
                "embedding": embedding
            }
            blocks.append((summary, meta))
        except Exception as e:
            logger.error(f"Erreur lecture {filename}: {e}")
            continue
    return blocks


# ---------------------------------------------------------------------------
#     Utilitaire : trouver les blocs les plus pertinents (pipeline IA PDF)
# ---------------------------------------------------------------------------
def find_relevant_blocks(
    question: str,
    blocks: List[Tuple[str, Dict]],
    top_k: int = 5,
    relevance_threshold: float = 0.4
) -> List[Dict]:
    candidate_blocks = []
    embeddings = []
    scores_debug = []

    for summary, meta in blocks:
        emb = meta.get("embedding")
        if emb is not None:
            embeddings.append(emb)
            candidate_blocks.append((summary, meta))

    if not embeddings:
        return []

    block_embs = torch.tensor(np.stack(embeddings), dtype=torch.float32).to("cpu")
    question_emb = model.encode(question, convert_to_tensor=True).to(torch.float32).to("cpu")
    similarities = util.pytorch_cos_sim(question_emb, block_embs)[0]

    scored = []
    for i, (_, meta) in enumerate(candidate_blocks):
        sim = similarities[i].item()
        quality = meta.get("score", 0)
        combined = 0.7 * sim + 0.3 * quality
        scores_debug.append({
            "filename": meta["source"],
                        "sim_score": round(sim, 4),
            "quality": round(quality, 4),
            "combined": round(combined, 4)
        })
        if combined >= relevance_threshold:
            scored.append((i, combined))

    if not scored:
        scored = sorted(
            [(i, 0.7 * similarities[i].item() + 0.3 * candidate_blocks[i][1].get("score", 0))
             for i in range(len(candidate_blocks))],
            key=lambda x: x[1], reverse=True
        )[:top_k]
    else:
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]

    pre_top = [i for i, _ in scored]
    cross_inputs = [(question, candidate_blocks[i][0]) for i in pre_top]
    rerank_scores = reranker.predict(cross_inputs)
    reranked = sorted(zip(pre_top, rerank_scores), key=lambda x: x[1], reverse=True)[:top_k]

    result = []
    for rank, (idx, rerank_score) in enumerate(reranked, 1):
        summary, meta = candidate_blocks[idx]
        block_score_debug = scores_debug[idx]
        logger.info(
            f"ASK Bloc sélectionné (RANK {rank}) : {meta['source']} | sim={block_score_debug['sim_score']}, "
            f"quality={block_score_debug['quality']}, combined={block_score_debug['combined']} | rerank_score={round(float(rerank_score),4)}"
        )
        result.append({
            "text": summary,
            "source": meta.get("source"),
            "debug_scores": block_score_debug
        })
    return result

# ---------------------------------------------------------------------------
#    Utilitaire : prompt pour synthèse multi-documents (mode générale)
# ---------------------------------------------------------------------------
def build_summary_prompt_from_metadata(question: str, docs: list) -> str:
    doc_lines = []
    for filename, resume in docs:
        doc_lines.append(f"- {filename}: {resume[:120]}...")
    docs_str = "\n".join(doc_lines)
    return f"""[INST]
Voici la question de l'utilisateur :
{question}

Voici la liste des documents pertinents avec leur résumé :
{docs_str}

Réponds par une synthèse claire :
- Si la question concerne plusieurs documents, cite-les brièvement.
- Si le sujet n’est traité dans aucun document, indique-le poliment.
- 2 à 3 phrases maximum.
[/INST]"""

# ---------------------------------------------------------------------------
#    Utilitaires prompts (PDF pipeline)
# ---------------------------------------------------------------------------
def build_prompt(question: str, selected_blocks: List[Dict]) -> str:
    parts = [f"--- Bloc {i+1} ({b['source']}) ---\n{b['text']}" for i, b in enumerate(selected_blocks)]
    context = "\n\n".join(parts)
    return f"""[INST]
Tu es un assistant IA expert et synthétique. Réponds en deux à trois phrases claires et précises à la question suivante, en t'appuyant exclusivement sur les extraits fournis.

QUESTION :
{question}

EXTRAITS :
{context}

INSTRUCTIONS :
- Réponds en deux à trois phrases maximum, sans introduction, ni structure académique.
- Si l’information n’est pas présente, dis-le poliment (“Non mentionné dans le document.”).
- Corrige les éventuelles fautes dans la question.
[/INST]"""

def build_reformulation_prompt(question: str, selected_blocks: List[Dict]) -> str:
    parts = [f"--- Bloc {i+1} ({b['source']}) ---\n{b['text']}" for i, b in enumerate(selected_blocks)]
    context = "\n\n".join(parts)
    return f"""[INST]
Tu es un assistant IA expert et synthétique. L’utilisateur demande une reformulation de ta réponse précédente à la même question.
Rédige une réponse différente, uniquement à partir des extraits fournis, sans te répéter.

QUESTION :
{question}

EXTRAITS :
{context}

RÈGLES :
- 1 à 3 phrases claires, sans préambule.
- Ne redis pas la même chose.
- Si l’information n’est pas présente, dis-le poliment (“Non mentionné dans le document.”).
[/INST]"""

# ---------------------------------------------------------------------------
#    FONCTION CENTRALE : Génération de la réponse IA
# ---------------------------------------------------------------------------
def generate_answer(
    question: str,
    blocks: List[Tuple[str, Dict]],
    job_id: str = None,
    session_id: str = None,
    user_id: str = None,
    entreprise: str = "Entreprise_S3_Test",
    reformule: bool = False,
    general: bool = False
) -> str:
    total_start = time.time()

    # --- 1. Classifier la question (générale ou précise) ---
    q_type, confiance = classify_question_with_score(question)
    logger.info(f"🧠 Type de question détecté : {q_type.upper()} (confiance={confiance})")

    if q_type == "générale":
        logger.info("🟡 Branche GÉNÉRALE — recherche via métadonnées uniquement")
        results = find_documents_by_keyword(question, entreprise, job_id)
        if not results:
            return "Aucun document ne correspond à cette thématique."

        prompt = build_summary_prompt_from_metadata(question, results)
        gen_start = time.time()
        answer = generate_ollama(
            prompt=prompt,
            num_predict=400,
            models=["llama3:instruct"],
            temperature=0.4
        ).strip()
        gen_time = time.time() - gen_start
        logger.info(f"⏱️ Temps génération (générale) : {gen_time:.2f}s")
        blocks_used = [doc[0] for doc in results]  # liste des filenames

    else:
        logger.info("🔵 Branche PRÉCISE — recherche vectorielle du PDF cible")
        question_emb = model.encode(question).tolist()
        pdf_filename = find_nearest_pdf_by_embedding(question_emb, entreprise, job_id)

        if not pdf_filename:
            logger.warning("Aucun document pertinent trouvé via embeddings.")
            return "Je n’ai pas trouvé de document correspondant à votre question."

        logger.info(f"📂 Document sélectionné via métadonnées : {pdf_filename}")

        # Charger uniquement les blocs liés à ce PDF
        all_blocks = load_all_blocks(entreprise, job_id)
        filtered_blocks = [
            b for b in all_blocks if pdf_filename in b[1].get("source", "")
        ]

        selected = find_relevant_blocks(question, filtered_blocks)
        retrieval_time = time.time() - total_start
        logger.info(f"ASK ⏱️ Temps sélection blocs (retrieval+rérank) : {retrieval_time:.2f}s")

        if not selected:
            logger.info("🧠 Aucun bloc pertinent — pas d'information.")
            return "Je n’ai pas trouvé cette information dans les documents analysés."

        logger.info(f"🧠 Prompt déclenché : 📄 PDF ({'reformulation' if reformule else 'standard'})")
        prompt = build_reformulation_prompt(question, selected) if reformule else build_prompt(question, selected)
        gen_start = time.time()
        answer = generate_ollama(
            prompt=prompt,
            num_predict=550,
            models=["llama3:instruct"],
            temperature=0.3
        ).strip()
        gen_time = time.time() - gen_start
        logger.info(f"⏱️ Temps génération (pdf) : {gen_time:.2f}s")
        blocks_used = [b["source"] for b in selected]

    total_time = time.time() - total_start
    logger.info(f"✅ Réponse générée en {total_time:.2f}s (total)")

    if session_id is None:
        session_id = str(uuid.uuid4())
    try:
        save_interaction(
            session_id=session_id,
            question=question,
            answer=answer,
            blocks_used=blocks_used,
            job_id=job_id,
            user_id=user_id
        )
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde de l'interaction chat : {e}")

    return answer

# ---------------------------------------------------------------------------
# FIN DU MODULE ask_engine.py
# ---------------------------------------------------------------------------


