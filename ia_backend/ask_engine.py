import os
import json
import logging
import uuid
import torch
import numpy as np
import time
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from ia_backend.services.ollama_gateway import generate_ollama
from ia_backend.services.chat_memory import save_interaction  # <-- Import chat memory

# Initialisation du logger
logger = logging.getLogger(__name__)

# Chargement des modèles
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

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
            # sécurité : si pas d'embedding, on skippe ce bloc
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

# --- Sélection des blocs pertinents (hybride embedding + rerank)
def find_relevant_blocks(question: str, blocks: List[Tuple[str, Dict]], top_k: int = 3) -> List[Dict]:
    candidate_blocks = []
    embeddings = []
    scores_debug = []  # Pour logs détaillés

    for summary, meta in blocks:
        emb = meta.get("embedding")
        if emb is not None:
            embeddings.append(emb)
            candidate_blocks.append((summary, meta))

    if not embeddings:
        return []

    # Embeddings des blocs et question sur le CPU, et tous en float32
    block_embs = torch.tensor(np.stack(embeddings), dtype=torch.float32).to("cpu")
    question_emb = model.encode(question, convert_to_tensor=True).to(torch.float32).to("cpu")

    similarities = util.pytorch_cos_sim(question_emb, block_embs)[0]

    scored = []
    for i, (_, meta) in enumerate(candidate_blocks):
        sim = similarities[i].item()
        quality = meta.get("score", 0)
        combined = 0.7 * sim + 0.3 * quality
        scored.append((i, combined))
        # Ajout pour debug
        scores_debug.append({
            "filename": meta["source"],
            "sim_score": round(sim, 4),
            "quality": round(quality, 4),
            "combined": round(combined, 4)
        })

    scored.sort(key=lambda x: x[1], reverse=True)
    pre_top = [i for i, _ in scored[:10]]  # Top 10 pour rerank

    # Re-ranking avec CrossEncoder
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

# --- Construction du prompt pour Ollama
def build_prompt(question: str, selected_blocks: List[Dict]) -> str:
    parts = []
    for i, b in enumerate(selected_blocks):
        parts.append(f"--- Bloc {i+1} ({b['source']}) ---\n{b['text']}")
    context = "\n\n".join(parts)
    return f"""[INST]
Tu es un assistant IA expert et synthétique. Réponds en une seule phrase claire et précise à la question suivante, en t'appuyant exclusivement sur les extraits fournis.

QUESTION :
{question}

EXTRAITS :
{context}

INSTRUCTIONS :
- Réponds en une à deux phrases maximum, sans introduction, ni structure académique.
- Si aucun élément pertinent, réponds : "Non mentionné dans le document."
[/INST]"""

# --- Génération de la réponse et enregistrement dans le chat memory
def generate_answer(
    question: str,
    blocks: List[Tuple[str, Dict]],
    job_id: str = None,
    session_id: str = None,
    user_id: str = None
) -> str:
    start_time = time.time()
    selected = find_relevant_blocks(question, blocks)
    elapsed = time.time() - start_time
    logger.info(f"ASK ⏱️ Temps total de sélection (retrieval+rérank+prompt) : {elapsed:.2f}s")

    if not selected:
        answer = "Aucune information pertinente trouvée dans le document."
    else:
        prompt = build_prompt(question, selected)
        gen_start = time.time()
        answer = generate_ollama(
            prompt=prompt,
            num_predict=400,
            models=["llama3:instruct"],
            temperature=0.3,
        ).strip()
        gen_elapsed = time.time() - gen_start
        logger.info(f"ASK ⏱️ Temps génération LLM : {gen_elapsed:.2f}s")

    # Enregistrement dans SQLite chat memory
    if session_id is None:
        session_id = str(uuid.uuid4())
    blocks_used = [b["source"] for b in selected] if selected else []
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
