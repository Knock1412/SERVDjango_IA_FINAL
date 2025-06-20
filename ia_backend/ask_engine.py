import os
import json
import logging
import uuid
import torch
import numpy as np
import time
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from duckduckgo_search import DDGS
from ia_backend.services.ollama_gateway import generate_ollama
from ia_backend.services.chat_memory import save_interaction

logger = logging.getLogger(__name__)

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def is_general_question(question: str) -> bool:
    import re
    patterns = [
        r"\b(quelle est|quand|combien|où|qui|pourquoi|comment|définir|définition de|loi|date)\b",
        r"\b(c’est quoi|peux[- ]tu m’expliquer|explique moi|histoire de)\b",
        r"\b(météo|heure|jour|capital[e]?|président|actualités?)\b",
    ]
    question = question.lower()
    return any(re.search(p, question) for p in patterns)

def search_web_duckduckgo(query: str, num_results: int = 5) -> List[str]:
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=num_results):
            if "body" in r:
                results.append(f"{r['title']}: {r['body']}")
    return results

def build_web_summary_prompt(question: str, results: List[str]) -> str:
    sources = "\n\n".join(f"- {res}" for res in results)
    return f"""[INST]
Tu es un assistant IA généraliste. Réponds en deux à trois phrases claire et précise à la question suivante, en t'appuyant exclusivement sur le résultat web.

Question : {question}

Résultats trouvés :
{sources}

Consignes :
- Réponds en deux à trois phrases maximum, sans introduction, ni structure académique.
- Synthétise uniquement les infos pertinentes
- Ne spécule pas. Si pas d'info claire, dis-le.
- Corrige les éventuelles fautes dans la question.
[/INST]"""

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
            [(i, 0.7 * similarities[i].item() + 0.3 * meta.get("score", 0))
             for i, (_, meta) in enumerate(candidate_blocks)],
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

def build_prompt(question: str, selected_blocks: List[Dict]) -> str:
    parts = [f"--- Bloc {i+1} ({b['source']}) ---\n{b['text']}" for i, b in enumerate(selected_blocks)]
    context = "\n\n".join(parts)
    return f"""[INST]
Tu es un assistant IA expert et synthétique. Réponds en deux à trois phrases claire et précise à la question suivante, en t'appuyant exclusivement sur les extraits fournis.

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

def generate_answer(
    question: str,
    blocks: List[Tuple[str, Dict]],
    job_id: str = None,
    session_id: str = None,
    user_id: str = None,
    reformule: bool = False,
    general: bool = False
) -> str:
    total_start = time.time()

    # 1. D'abord, détecte si c'est une question générale
    if general or is_general_question(question):
        logger.info("🧠 Prompt déclenché : 🌐 Web (DuckDuckGo)")
        web_results = search_web_duckduckgo(question)
        if not web_results:
            return "Désolé, je n'ai rien trouvé sur le web."
        prompt = build_web_summary_prompt(question, web_results)
        gen_start = time.time()
        answer = generate_ollama(
            prompt=prompt,
            num_predict=400,
            models=["llama3:instruct"],
            temperature=0.4
        ).strip()
        gen_time = time.time() - gen_start
        logger.info(f"⏱️ Temps génération (web) : {gen_time:.2f}s")
        blocks_used = []

    else:
        # 2. Sinon, logique classique PDF
        selected = find_relevant_blocks(question, blocks)
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
