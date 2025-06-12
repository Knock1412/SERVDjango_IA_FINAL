import os
import json
import logging
import uuid
import torch
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from ia_backend.services.ollama_gateway import generate_ollama
from ia_backend.services.chat_memory import save_interaction  # <-- Import chat memory

# Initialisation du logger
logger = logging.getLogger(__name__)

# Chargement des modèles
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Chargement des blocs de résumé avec embeddings
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
    for summary, meta in blocks:
        emb = meta.get("embedding")
        if emb is not None:
            embeddings.append(emb)
            candidate_blocks.append((summary, meta))

    if not embeddings:
        return []

    # --- Convert to float32 tensor and send to CPU
    block_embs = torch.tensor(np.stack(embeddings), dtype=torch.float32).to("cpu")
    question_emb = model.encode(question, convert_to_tensor=True).to(torch.float32).to("cpu")

    logger.debug(f"block_embs dtype: {block_embs.dtype}, device: {block_embs.device}")
    logger.debug(f"question_emb dtype: {question_emb.dtype}, device: {question_emb.device}")

    similarities = util.pytorch_cos_sim(question_emb, block_embs)[0]

    scored = []
    for i, (_, meta) in enumerate(candidate_blocks):
        sim = similarities[i].item()
        quality = meta.get("score", 0)
        combined = 0.7 * sim + 0.3 * quality
        scored.append((i, combined))
    scored.sort(key=lambda x: x[1], reverse=True)
    pre_top = [i for i, _ in scored[:10]]  # Top 10 pour rerank

    # 2) Re-ranking avec CrossEncoder
    cross_inputs = [(question, candidate_blocks[i][0]) for i in pre_top]
    rerank_scores = reranker.predict(cross_inputs)
    reranked = sorted(zip(pre_top, rerank_scores), key=lambda x: x[1], reverse=True)[:top_k]

    result = []
    for idx, _ in reranked:
        summary, meta = candidate_blocks[idx]
        result.append({
            "text": summary,
            "source": meta.get("source")
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
    selected = find_relevant_blocks(question, blocks)
    if not selected:
        answer = "Aucune information pertinente trouvée dans le document."
    else:
        prompt = build_prompt(question, selected)
        answer = generate_ollama(
            prompt=prompt,
            num_predict=400,
            models=["llama3:instruct"],
            temperature=0.3,
            
        ).strip()

    # --- Enregistrement dans SQLite chat memory
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
