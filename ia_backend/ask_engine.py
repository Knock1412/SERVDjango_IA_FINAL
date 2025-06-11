import os
import json
import time  # ⏱️ Pour mesurer le temps
from sentence_transformers import SentenceTransformer, util
from ia_backend.services.ollama_gateway import generate_ollama

# 🧠 Chargement du modèle multilingue rapide
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def load_all_blocks(entreprise: str, job_id: str) -> list[str]:
    """
    Charge tous les blocs de texte d'un document à partir du cache JSON structuré.
    """
    folder_path = os.path.join("cache_json", "save_summaryblocks", entreprise, job_id)
    print(f"📂 Tentative de chargement des blocs depuis : {folder_path}")  # DEBUG

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Dossier introuvable : {folder_path}")
    
    blocks = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.startswith("bloc_") and filename.endswith(".json"):
            full_path = os.path.join(folder_path, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                try:
                    content = json.load(f)
                    if isinstance(content, str):
                        blocks.append(content)
                    elif isinstance(content, dict) and "summary" in content:
                        blocks.append(content["summary"])
                    else:
                        print(f"⚠️ Clé 'summary' manquante dans {filename}")
                except Exception as e:
                    print(f"❌ Erreur lecture {filename}: {e}")
    return blocks


def find_relevant_blocks(question: str, blocks: list[str], top_k: int = 5) -> list[str]:
    """
    Sélectionne les blocs les plus pertinents par similarité sémantique avec la question.
    """
    if not blocks:
        return []

    question_emb = model.encode(question, convert_to_tensor=True)
    block_embs = model.encode(blocks, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(question_emb, block_embs)[0]
    top_indices = similarities.topk(k=min(top_k, len(blocks))).indices

    return [blocks[i] for i in top_indices]


def generate_answer(question: str, context_blocks: list[str]) -> str:
    """
    Construit un prompt structuré avec les blocs et interroge Ollama.
    """
    context = "\n---\n".join(context_blocks)
    prompt = f"""[INST] Tu es un assistant IA spécialisé en réponse précise à partir de documents techniques.

Voici une question :
{question}

Voici les extraits pertinents du document :
{context}

Donne une réponse claire, structurée, et pertinente.[/INST]"""
    start_time = time.time()
    result = generate_ollama(prompt=prompt, num_predict=500)
    duration = round(time.time() - start_time, 2)
    print(f"🕒 Temps de génération de la réponse : {duration}s")

    return generate_ollama(prompt=prompt, num_predict=500)
