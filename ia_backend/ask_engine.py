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
    find_nearest_pdf_by_embedding,      # FAISS
    find_documents_by_keyword,          # FTS5 (gardé si besoin)
    find_documents_by_keyword_semantic
)


from ia_backend.services.chat_memory import save_interaction

# --- Initialisation logging et modèles ---
logger = logging.getLogger(__name__)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ---------------------------------------------------------------------------
#    Utilitaire : classification LLM “générale/précise” + score de confiance
# ---------------------------------------------------------------------------
from functools import lru_cache
import concurrent.futures
from typing import Tuple, Dict
import numpy as np

# Cache pour les embeddings de questions (réduit les appels LLM)
QUESTION_EMBEDDINGS_CACHE = lru_cache(maxsize=1000)

# Modèle lightweight pour pré-classification
FAST_CLASSIFIER = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Exemples de questions pré-classifiées pour few-shot learning
# Exemples de questions pré-classifiées pour few-shot learning
PRECLASSIFIED_EXAMPLES = [
    # Questions GÉNÉRALES
    ("Liste des documents sur la fiscalité", "générale"),
    ("Résumez les rapports financiers 2023", "générale"),
    ("Quels documents traitent des politiques éducatives ?", "générale"),
    ("Quels sont les thèmes abordés dans les documents récents ?", "générale"),
    ("Montre-moi les documents liés à la transformation numérique", "générale"),
    ("Quels fichiers abordent l'enseignement à distance ?", "générale"),
    ("Y a-t-il des documents qui parlent d'éthique en IA ?", "générale"),
    ("Quels rapports concernent les innovations pédagogiques ?", "générale"),

    # Questions PRÉCISES
    ("Quel est l'article sur les impôts locaux ?", "précise"),
    ("Page 42 du document X", "précise"),
    ("Quelles sont les conclusions du rapport sur la visioconférence ?", "précise"),
    ("Combien de pages contient le document sur le Cartable Électronique ?", "précise"),
    ("Quelle est la date de publication du PDF sur la nanobureautique ?", "précise"),
    ("Quels logiciels sont mentionnés dans le bloc 3 du document X ?", "précise"),
    ("Quelles sont les critiques soulevées dans le document sur les TICE ?", "précise"),
    ("Quelle méthode pédagogique est décrite dans le document sur le MO5 ?", "précise")
]


def classify_question_with_score_v2(question: str) -> Tuple[str, float]:
    """
    Version optimisée avec fallback intelligent et pré-classification
    """
    # Étape 1: Pré-classification rapide avec embedding (évite 60% des appels LLM)
    pre_class, pre_conf = fast_preclassify(question)
    if pre_conf > 0.85:  # Seuil de confiance élevé
        return (pre_class, pre_conf)

    # Étape 2: Appel LLM seulement si nécessaire
    llm_class, llm_conf = call_llm_classifier(question)
    
    # Étape 3: Fusion intelligente des résultats
    final_class, final_conf = combine_results(
        pre_class, pre_conf,
        llm_class, llm_conf
    )
    
    return (final_class, final_conf)

@QUESTION_EMBEDDINGS_CACHE
def fast_preclassify(question: str) -> Tuple[str, float]:
    """
    Classification rapide avec similarité sémantique sur exemples connus
    """
    question_emb = FAST_CLASSIFIER.encode(question)
    examples_embs = [FAST_CLASSIFIER.encode(ex[0]) for ex in PRECLASSIFIED_EXAMPLES]
    
    similarities = util.pytorch_cos_sim(question_emb, examples_embs)[0]
    max_idx = similarities.argmax().item()
    
    if similarities[max_idx] > 0.75:  # Seuil de similarité
        return (PRECLASSIFIED_EXAMPLES[max_idx][1], float(similarities[max_idx]))
    
    return ("inconnu", 0.0)

def call_llm_classifier(question: str) -> Tuple[str, float]:
    """
    Appel LLM optimisé avec timeout et retry
    """
    prompt = build_few_shot_prompt(question)
    
    for attempt in range(2):  # 2 tentatives max
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    generate_ollama,
                    prompt=prompt,
                    num_predict=80,  # Réduit pour la classification
                    
                )
                response = future.result(timeout=3.0)
                
            data = json.loads(response.strip())
            return validate_response(data)
        
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {str(e)}")
            continue
            
    return ("précise", 0.5)  # Fallback conservateur

def build_few_shot_prompt(question: str) -> str:
    """
    Prompt avec exemples pour meilleure consistance
    """
    examples_str = "\n".join(
        f"- Exemple {i+1} ({type}): {q}"
        for i, (q, type) in enumerate(PRECLASSIFIED_EXAMPLES)
    )
    
    return f"""[INST]
Tu es un classifieur de questions. Voici des exemples :

{examples_str}

Classifie cette nouvelle question :

QUESTION: {question}

Réponds UNIQUEMENT en JSON valide :
{{
  "type": "générale"|"précise",
  "confiance": 0.0-1.0,
  "raison": "explication courte"
}}
[/INST]"""

def combine_results(
    pre_class: str, pre_conf: float,
    llm_class: str, llm_conf: float
) -> Tuple[str, float]:
    """
    Combine intelligemment la pré-classification et la classification LLM
    """
    if llm_conf >= 0.75:
        return (llm_class, llm_conf)

    if pre_class == llm_class:
        # Accord entre classifieur rapide et LLM
        avg_conf = round((pre_conf + llm_conf) / 2, 3)
        return (llm_class, avg_conf)

    # Divergence : on fait confiance au LLM s’il dépasse un certain seuil
    if llm_conf >= 0.6:
        return (llm_class, llm_conf)
    else:
        return (pre_class, pre_conf)

def validate_response(data: Dict) -> Tuple[str, float]:
    """
    Validation robuste de la réponse LLM
    """
    if not isinstance(data, dict):
        raise ValueError("Invalid JSON format")
    
    q_type = data.get("type", "").lower()
    if q_type not in {"générale", "précise", "precise"}:
        raise ValueError("Invalid question type")
    
    confiance = min(max(float(data.get("confiance", 0.5)), 1.0), 0.0)
    
    # Post-processing basé sur l'explication
    if "raison" in data:
        if "document spécifique" in data["raison"].lower():
            q_type = "précise"
        elif "plusieurs" in data["raison"].lower():
            q_type = "générale"
    
    return (q_type, round(confiance, 2))





# ---------------------------------------------------------------   ------------
#     Utilitaire : charger tous les blocs d’un job_id (fichier par PDF)
# ---------------------------------------------------------------------------
def load_all_blocks(entreprise: str, job_id: str) -> List[Tuple[str, Dict]]:
    """
    Charge tous les blocs de résumé depuis le cache JSON pour une entreprise et un job donnés.
    
    Args:
        entreprise: Nom de l'entreprise
        job_id: Identifiant du job
        
    Returns:
        Liste de tuples (résumé, métadonnées) pour chaque bloc valide
        
    Raises:
        FileNotFoundError: Si le dossier spécifié n'existe pas
    """
    folder_path = os.path.join("cache_json", "save_summaryblocks", entreprise, job_id)
    logger.info(f"Chargement des blocs depuis le dossier: {folder_path}")
    
    if not os.path.exists(folder_path):
        logger.error(f"Dossier introuvable: {folder_path}")
        raise FileNotFoundError(f"Dossier introuvable : {folder_path}")

    blocks: List[Tuple[str, Dict]] = []
    valid_files = 0
    skipped_files = 0
    
    for filename in sorted(os.listdir(folder_path)):
        # Filtrage des fichiers JSON de blocs
        if not filename.startswith("bloc_") or not filename.endswith(".json"):
            skipped_files += 1
            continue
            
        full_path = os.path.join(folder_path, filename)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            if "embedding" not in data:
                logger.warning(f"Fichier {filename} ignoré - embedding manquant")
                skipped_files += 1
                continue

            # Vérifie que l'embedding est bien une liste de 384 floats
            emb = data["embedding"]
            if isinstance(emb, str):
                try:
                    emb = json.loads(emb)
                except Exception as e:
                    logger.warning(f"{filename} - embedding JSON mal formé: {e}")
                    skipped_files += 1
                    continue

            if not isinstance(emb, list) or len(emb) != 384:
                logger.warning(f"{filename} - embedding invalide (type: {type(emb)}, taille: {len(emb) if isinstance(emb, list) else 'N/A'})")
                skipped_files += 1
                continue

            # Construction des métadonnées
            meta = {
                "source": filename,
                "score": data.get("score", 0),
                "translated": data.get("translated", False),
                "embedding": emb
            }
            blocks.append((data.get("summary", ""), meta))
            valid_files += 1
            
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de décodage JSON dans {filename}: {str(e)}")
            skipped_files += 1
        except Exception as e:
            logger.error(f"Erreur inattendue lors du traitement de {filename}: {str(e)}")
            skipped_files += 1

    logger.info(
        f"Chargement terminé - {valid_files} blocs valides, "
        f"{skipped_files} fichiers ignorés/erronés"
    )
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
    """
    Trouve les blocs les plus pertinents pour une question donnée en combinant:
    - Similarité sémantique (70%)
    - Score de qualité du bloc (30%)
    - Re-ranking croisé
    
    Args:
        question: Question à laquelle répondre
        blocks: Liste des blocs à analyser
        top_k: Nombre maximum de résultats à retourner
        relevance_threshold: Seuil minimal de pertinence
        
    Returns:
        Liste des blocs pertinents avec leurs métadonnées
    """
    # Phase 1: Préparation des embeddings
    candidate_blocks = []
    embeddings = []
    logger.info(f"Début de recherche sur {len(blocks)} blocs - top_k={top_k}")
    
    for summary, meta in blocks:
        if "embedding" in meta:
            embeddings.append(meta["embedding"])
            candidate_blocks.append((summary, meta))
    
    if not embeddings:
        logger.warning("Aucun embedding valide trouvé - recherche annulée")
        return []

    # Phase 2: Calcul des similarités
    logger.debug("Conversion des embeddings en tenseurs...")
    block_embs = torch.tensor(np.stack(embeddings), dtype=torch.float32).to("cpu")
    question_emb = model.encode(question, convert_to_tensor=True).to(torch.float32).to("cpu")
    
    logger.info("Calcul des scores de similarité...")
    similarities = util.pytorch_cos_sim(question_emb, block_embs)[0]
    
    # Phase 3: Combinaison des scores
    scored_blocks = []
    debug_scores = []
    
    for i, (_, meta) in enumerate(candidate_blocks):
        sim_score = similarities[i].item()
        quality_score = meta.get("score", 0)
        combined_score = 0.7 * sim_score + 0.3 * quality_score
        
        debug_scores.append({
            "source": meta["source"],
            "sim_score": round(sim_score, 4),
            "quality_score": round(quality_score, 4),
            "combined_score": round(combined_score, 4)
        })
        
        logger.debug(
            f"Bloc {meta['source']} - "
            f"Similarité: {sim_score:.4f}, "
            f"Qualité: {quality_score:.4f}, "
            f"Combined: {combined_score:.4f}"
        )
        
        if combined_score >= relevance_threshold:
            scored_blocks.append((i, combined_score))

    logger.info(
        f"{len(scored_blocks)} blocs dépassent le seuil de {relevance_threshold} "
        f"sur {len(candidate_blocks)} analysés"
    )

    # Phase 4: Sélection initiale
    if not scored_blocks:
        logger.warning("Aucun bloc ne dépasse le seuil - utilisation des meilleurs scores")
        scored_blocks = [
            (i, 0.7 * similarities[i].item() + 0.3 * candidate_blocks[i][1].get("score", 0))
            for i in range(len(candidate_blocks))
        ]
    
    # Tri et sélection des top_k
    scored_blocks.sort(key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in scored_blocks[:top_k]]
    
    # Phase 5: Re-ranking croisé
    logger.info("Application du re-ranking croisé...")
    cross_inputs = [(question, candidate_blocks[i][0]) for i in top_indices]
    rerank_scores = reranker.predict(cross_inputs)
    
    # Construction des résultats finaux
    results = []
    for rank, (idx, rerank_score) in enumerate(
        sorted(zip(top_indices, rerank_scores), key=lambda x: x[1], reverse=True),
        1
    ):
        summary, meta = candidate_blocks[idx]
        scores = debug_scores[idx]
        
        logger.info(
            f"Bloc sélectionné (#{rank}): {meta['source']}\n"
            f"  - Similarité: {scores['sim_score']:.4f}\n"
            f"  - Qualité: {scores['quality_score']:.4f}\n"
            f"  - Score combiné: {scores['combined_score']:.4f}\n"
            f"  - Re-rank score: {float(rerank_score):.4f}"
        )
        
        results.append({
            "text": summary,
            "source": meta["source"],
            "debug_scores": scores
        })

    return results[:top_k]

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

Réponds par une synthèse claire :
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
    q_type, confiance = classify_question_with_score_v2(question)
    logger.info(f"🧠 Type de question détecté : {q_type.upper()} (confiance={confiance})")

    if q_type == "générale":
        logger.info("🟡 Branche GÉNÉRALE — recherche hybride (FTS+embeddings) sur mots_cles/themes")
        results = find_documents_by_keyword_semantic(
            question, entreprise, job_id, encode_text_fn=model.encode
        )
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
        pdf_matches = find_nearest_pdf_by_embedding(question_emb, entreprise, job_id)

        if not pdf_matches:
            logger.warning("Aucun document pertinent trouvé via embeddings.")
            return "Je n’ai pas trouvé de document correspondant à votre question."

        # On prend uniquement le fichier avec le meilleur score
        pdf_filename, score = pdf_matches[0]
        logger.info(f"📂 Document sélectionné via FAISS : {pdf_filename} (score={score:.4f})")

        # Charger uniquement les blocs liés à ce PDF
        all_blocks = load_all_blocks(entreprise, job_id)
        filtered_blocks = [
            b for b in all_blocks if b[1].get("pdf_filename") == pdf_filename
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


