import threading
import queue
import time
import os
import logging

from ia_backend.services.pdf_utils import (
    extract_blocks_from_pdf,
    extract_text_from_block,
    extract_full_text,
    detect_annex_start_page,
    is_likely_annex
)
from ia_backend.services.summarizer import (
    summarize_block,
    summarize_global,
    evaluate_summary_score,
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
    formatter = logging.Formatter('%(message)s')  # ✅ Suppression timestamp
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ---------- Paramètres industriels centralisés ----------
BLOCK_THRESHOLD_INITIAL = 0.7
INTERMEDIATE_GROUP_SIZE = 5
MAX_ATTEMPTS = 4

# ---------- Job Queue ----------


class Job:
    def __init__(self, priority, job_id, entreprise, pdf_path, pdf_url, folder_name):
        self.priority = priority
        self.job_id = job_id
        self.entreprise = entreprise
        self.pdf_path = pdf_path
        self.pdf_url = pdf_url
        self.folder_name = folder_name


def save_txt(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# ---------- Pipeline complet ----------
def process_job(job: Job):
    try:
        
        start_total = time.time()

        logger.info(f"\n🚀 Démarrage traitement job {job.job_id} (priorité : {job.priority})")

        full_pdf_text = extract_full_text(job.pdf_path)
        logger.info(f"📄 Extraction full text : {len(full_pdf_text)} caractères extraits")

        full_text_path = f"temp_cache/{job.folder_name}_full_text.txt"
        save_txt(full_text_path, full_pdf_text)

        total_pages = extract_blocks_from_pdf(job.pdf_path, return_pages_only=True)
        logger.info(f"🧩 Découpage dynamique | Total pages : {total_pages}")

        annex_start_page = detect_annex_start_page(job.pdf_path)
        blocks = extract_blocks_from_pdf(job.pdf_path)
        summaries = []

        for idx, block_indexes in enumerate(blocks):
            logger.info(f"\n⏳ Bloc {idx + 1} en cours...")

            text = extract_text_from_block(block_indexes, job.pdf_path)
            if not text.strip():
                text = "(Bloc vide ou inexploitable.)"

            logger.info(f"📊 Bloc {idx+1} — Longueur brute : {len(text.strip())} caractères")

            if is_likely_annex(block_indexes, total_pages, annex_start_page):
                logger.info(f"🚩 Bloc {idx+1} détecté comme ANNEXE. Ignoré.")
                continue

            if len(text) > 7000:
                logger.info(f"✂️ Bloc {idx+1} pré-tronqué à 7000 avant IA")
                text = text[:7000]

            best_score = 0
            best_summary = ""

            for attempt in range(1, MAX_ATTEMPTS + 1):
                summary = summarize_block(text)
                processed_summary, translated = process_text_block(summary)

                lang_info = "Traduit EN->FR" if translated else "Déjà en FR"
                score = evaluate_summary_score(text, processed_summary)
                logger.info(f"⚙️ Bloc #{idx+1} Essai {attempt} - Score={score:.3f} - Best={best_score:.3f} - {lang_info}")

                if score > best_score:
                    best_score = score
                    best_summary = processed_summary

                if score >= BLOCK_THRESHOLD_INITIAL:
                    logger.info(f"✅ Bloc {idx+1} validé dès essai {attempt} (score={score:.3f})")
                    break

            logger.info(f"\n✅ Bloc {idx+1} terminé : meilleur score={best_score:.3f}")

            json_dir = f"cache_json/save_summaryblocks/{job.entreprise}/{job.folder_name}"
            save_json(json_dir, idx, {
                "bloc": idx + 1,
                "summary": best_summary,
                "source_pdf": job.pdf_url,
                "score": best_score,
                "translated": translated
            })

            summaries.append((idx + 1, best_summary))

        summaries.sort()
        joined = [s for _, s in summaries]

        intermediates = []
        for i in range(0, len(joined), INTERMEDIATE_GROUP_SIZE):
            group = joined[i:i+INTERMEDIATE_GROUP_SIZE]
            logger.info(f"\n🔄 Fusion intermédiaire lot {i//INTERMEDIATE_GROUP_SIZE + 1} ({len(group)} résumés)")

            intermediate_summary = summarize_global(group, num_predict=1000)
            processed_summary, translated = process_text_block(intermediate_summary)

            inter_json_dir = f"cache_json/save_summaryintermediates/{job.entreprise}/{job.folder_name}"
            os.makedirs(inter_json_dir, exist_ok=True)
            save_json(inter_json_dir, i//INTERMEDIATE_GROUP_SIZE, {
                "intermediate_block": i//INTERMEDIATE_GROUP_SIZE + 1,
                "summary": processed_summary
            })

            intermediates.append(processed_summary)

        logger.info(f"\n🔍 Fusion finale sur {len(intermediates)} intermédiaires...")

        final_predict_len = 1300 if len(intermediates) <= 10 else 1500
        final_summary = summarize_global(intermediates, num_predict=final_predict_len, is_final=True)
        global_score = evaluate_summary_score(full_pdf_text, final_summary, partial_summaries=intermediates)
        logger.info(f"📊 Score global (info only) = {global_score:.3f}")

        save_global_summary(job.entreprise, job.folder_name, final_summary)
        log_job_history(job.job_id, job.entreprise, job.pdf_url, "terminé", "mistral", start_total)

        logger.info(f"\n✅ Traitement finalisé pour job {job.job_id}")

    except Exception as e:
        logger.error(f"❌ Erreur job {job.job_id}: {e}", exc_info=True)