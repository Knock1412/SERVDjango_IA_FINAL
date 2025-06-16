import os
import time
import logging
from sentence_transformers import SentenceTransformer  # ✅ Pour embeddings

from ia_backend.services.pdf_utils import (
    extract_blocks_from_pdf,
    extract_text_from_block,
    extract_full_text
)
from ia_backend.services.summarizer import (
    summarize_block,
    summarize_global,
    evaluate_summary_score,
    is_summary_valid  # <-- AJOUT IMPORT
)
from ia_backend.services.cache_manager import save_json
from ia_backend.services.backup_service import save_global_summary
from ia_backend.services.job_logger import log_job_history
from ia_backend.services.language_detection_and_translation import process_text_block

# ---------- Logging centralisé optimisé ----------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ---------- Paramètres industriels centralisés ----------
BLOCK_THRESHOLD_INITIAL = 0.68
INTERMEDIATE_GROUP_SIZE = 5 
MAX_ATTEMPTS = 4

# ✅ Chargement du modèle d'embedding une seule fois
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ---------- Definition du Job ----------
class Job:
    def __init__(self, priority, job_id, entreprise, pdf_path, pdf_url, folder_name):
        self.priority = priority
        self.job_id = job_id
        self.entreprise = entreprise
        self.pdf_path = pdf_path
        self.pdf_url = pdf_url
        self.folder_name = folder_name

# ---------- Fonction utilitaire ----------
def save_txt(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# ---------- Pipeline complet ----------
def process_job(job: Job):
    start_total = time.time()

    logger.info(f"\n🚀 Démarrage traitement job {job.job_id} (priorité : {job.priority})")

    full_pdf_text = extract_full_text(job.pdf_path)
    logger.info(f"📄 Extraction full text : {len(full_pdf_text)} caractères extraits")

    full_text_path = f"temp_cache/{job.folder_name}_full_text.txt"
    save_txt(full_text_path, full_pdf_text)

    blocks = extract_blocks_from_pdf(job.pdf_path)
    total_pages = sum(len(page) for page in blocks)
    logger.info(f"🧩 Découpage dynamique | Total pages approx : {total_pages}")

    summaries = []

    for idx, block_indexes in enumerate(blocks):
        logger.info(f"\n⏳ Bloc {idx + 1} en cours...")

        text = extract_text_from_block(block_indexes, job.pdf_path)
        if not text.strip():
            text = "(Bloc vide ou inexploitable.)"

        logger.info(f"📊 Bloc {idx+1} — Longueur brute : {len(text.strip())} caractères")

        if len(text) > 8000:
            logger.info(f"✂️ Bloc {idx+1} pré-tronqué à 8000 avant IA")
            text = text[:8000]

        best_score = 0
        best_summary = ""
        translated = False

        for attempt in range(1, MAX_ATTEMPTS + 1):
            summary = summarize_block(text)
            processed_summary, translated = process_text_block(summary)
            score = evaluate_summary_score(text, processed_summary)

            # 👇 On n'accepte que structuré ET score >= threshold
            if is_summary_valid(processed_summary) and score >= BLOCK_THRESHOLD_INITIAL:
                logger.info(f"✅ Résumé structuré et au-dessus du seuil trouvé à l'essai {attempt} (score={score:.3f})")
                best_score = score
                best_summary = processed_summary
                break  # satisfait, on arrête la boucle
            else:
                if score > best_score:
                    best_score = score
                    best_summary = processed_summary
                logger.warning(f"Résumé rejeté (non structuré ou score trop bas) essai {attempt} (score={score:.3f})")

        if not best_summary.strip():
            logger.error(f"Bloc {idx+1} ignoré : aucun résumé généré.")
            continue

        if not (is_summary_valid(best_summary) and best_score >= BLOCK_THRESHOLD_INITIAL):
            logger.warning(f"Bloc {idx+1}: Pas de résumé structuré et/ou au-dessus du seuil, mais on garde le meilleur essai (score={best_score:.3f})")

        logger.info(f"\n✅ Bloc {idx+1} retenu : score={best_score:.3f}")

        embedding = embedding_model.encode(best_summary).tolist()

        json_dir = f"cache_json/save_summaryblocks/{job.entreprise}/{job.job_id}"
        save_json(json_dir, idx, {
            "bloc": idx + 1,
            "summary": best_summary,
            "source_pdf": job.pdf_url,
            "score": best_score,
            "translated": translated,
            "embedding": embedding
        })

        summaries.append((idx + 1, best_summary))

    summaries.sort()
    joined = [s for _, s in summaries]

    intermediates = []
    for i in range(0, len(joined), INTERMEDIATE_GROUP_SIZE):
        group = joined[i:i+INTERMEDIATE_GROUP_SIZE]
        logger.info(f"\n🔄 Fusion intermédiaire lot {i//INTERMEDIATE_GROUP_SIZE + 1} ({len(group)} résumés)")

        intermediate_summary = summarize_global(group, is_final=False)
        processed_summary, translated = process_text_block(intermediate_summary)

        inter_json_dir = f"cache_json/save_summaryintermediates/Entreprise_{job.entreprise}/{job.folder_name}"
        os.makedirs(inter_json_dir, exist_ok=True)
        save_json(inter_json_dir, i//INTERMEDIATE_GROUP_SIZE, {
            "intermediate_block": i//INTERMEDIATE_GROUP_SIZE + 1,
            "summary": processed_summary
        })

        intermediates.append(processed_summary)

    logger.info(f"\n🔍 Fusion finale sur {len(intermediates)} intermédiaires...")

    final_summary = summarize_global(intermediates, is_final=True)
    global_score = evaluate_summary_score(full_pdf_text, final_summary, partial_summaries=intermediates)
    logger.info(f"📊 Score global (info only) = {global_score:.3f}")

    save_global_summary(job.entreprise, job.folder_name, final_summary, job_id=job.job_id)
    log_job_history(job.job_id, job.entreprise, job.pdf_url, "terminé", "mistral", start_total)

    logger.info(f"\n✅ Traitement finalisé pour job {job.job_id}")

    return {
        "summary": final_summary,
        "mode": "hierarchical_v5"
    }
