import threading
import queue
import time
import os

from ia_backend.services.pdf_utils import (
    extract_blocks_from_pdf,
    extract_text_from_block,
    extract_full_text,
)
from ia_backend.services.summarizer import (
    summarize_block,
    summarize_global,
    determine_predict_length,
    evaluate_summary_score,
)
from ia_backend.services.cache_manager import save_json
from ia_backend.services.backup_service import save_global_summary
from ia_backend.services.job_logger import log_job_history

# 🚩 NOUVEL IMPORT POUR LA DETECTION / TRADUCTION
from ia_backend.services.language_detection_and_translation import process_text_block

# Seuils dynamiques
BLOCK_THRESHOLD_INITIAL = 0.7
BLOCK_THRESHOLD_SECONDARY = 0.65

job_queue = queue.PriorityQueue()
JOB_STATUS = {}
JOB_RESULTS = {}

class Job:
    def __init__(self, priority, job_id, entreprise, pdf_path, pdf_url, folder_name):
        self.priority = priority
        self.job_id = job_id
        self.entreprise = entreprise
        self.pdf_path = pdf_path
        self.pdf_url = pdf_url
        self.folder_name = folder_name

    def __lt__(self, other):
        return self.priority < other.priority

def save_txt(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def process_job(job: Job):
    try:
        JOB_STATUS[job.job_id] = "en cours"
        start_total = time.time()
        print(f"\n🚀 Traitement démarré pour job {job.job_id} (priorité : {job.priority})")

        full_pdf_text = extract_full_text(job.pdf_path)
        print(f"📄 Extraction complète : {len(full_pdf_text)} caractères extraits")

        full_text_path = f"temp_cache/{job.folder_name}_full_text.txt"
        save_txt(full_text_path, full_pdf_text)

        total_pages = extract_blocks_from_pdf(job.pdf_path, return_pages_only=True)
        chunk_size = 1  # Toujours 1 page fixe désormais
        print(f"🧩 Mode : page_par_page | Total pages : {total_pages}")

        blocks = extract_blocks_from_pdf(job.pdf_path, chunk_size=chunk_size)
        summaries = []

        for idx, block_indexes in enumerate(blocks):
            print(f"⏳ Bloc {idx + 1} en cours...")
            text = extract_text_from_block(block_indexes, job.pdf_path)
            if not text.strip():
                text = "(Bloc vide ou inexploitable.)"

            if len(text) > 7000:
                print(f"✂️ Bloc {idx+1} pré-tronqué à 7000 avant IA (brut={len(text)})")
                text = text[:7000]

            best_score = 0
            best_summary = ""
            attempt = 0

            # === Première boucle avec seuil 0.7 (max 5 tentatives)
            while attempt < 5:
                attempt += 1

                # Génération du résumé via IA
                summary, model = summarize_block(text)

                # 🚩 NOUVEAU : détection + traduction dès la génération
                processed_summary, translated = process_text_block(summary)

                if translated:
                    print(f"🌍 Bloc {idx+1} traduit de l'anglais vers le français (avant scoring).")
                else:
                    print(f"🌍 Bloc {idx+1} déjà en français avant scoring.")

                # Le scoring est toujours réalisé sur la version traduite
                score = evaluate_summary_score(text, processed_summary)
                print(f"⚙️ Bloc #{idx+1} Essai {attempt} - Score = {round(score,3)} - Meilleur = {round(best_score,3)}")

                if score > best_score:
                    best_score = score
                    best_summary = processed_summary  # On stocke directement la version traduite

                if best_score >= BLOCK_THRESHOLD_INITIAL:
                    print(f"✅ Bloc {idx+1} validé avec score {round(best_score,3)} après {attempt} tentatives")
                    break
                else:
                    print("🔁 Reprocessing...")

            # Si toujours pas validé, on passe au seuil relâché 0.65
            if best_score < BLOCK_THRESHOLD_INITIAL:
                print(f"⚠️ Passage seuil secondaire pour bloc {idx+1}")

                if best_score >= BLOCK_THRESHOLD_SECONDARY:
                    print(f"✅ Bloc {idx+1} validé immédiatement au score secondaire {round(best_score,3)}")
                else:
                    secondary_attempt = 0
                    while secondary_attempt < 7:
                        secondary_attempt += 1

                        summary, model = summarize_block(text)
                        processed_summary, translated = process_text_block(summary)

                        if translated:
                            print(f"🌍 Bloc {idx+1} traduit de l'anglais vers le français (avant scoring secondaire).")
                        else:
                            print(f"🌍 Bloc {idx+1} déjà en français avant scoring secondaire.")

                        score = evaluate_summary_score(text, processed_summary)
                        print(f"⚙️ Bloc #{idx+1} Second Essai {secondary_attempt} - Score = {round(score,3)} - Meilleur = {round(best_score,3)}")

                        if score > best_score:
                            best_score = score
                            best_summary = processed_summary

                        if best_score >= BLOCK_THRESHOLD_SECONDARY:
                            print(f"✅ Bloc {idx+1} validé après seuil secondaire avec score {round(best_score,3)}")
                            break

            # Enregistrement final du bloc (toujours version traduite)
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
        num_predict = determine_predict_length(len(joined))
        print(f"🔍 Génération résumé global avec num_predict = {num_predict}")

        final_summary, model_used = summarize_global(joined, num_predict=num_predict)
        global_score = evaluate_summary_score(full_pdf_text, final_summary, partial_summaries=joined)
        print(f"📊 Score résumé global (info only) = {round(global_score, 3)}")

        save_global_summary(job.entreprise, job.folder_name, final_summary)
        log_job_history(job.job_id, job.entreprise, job.pdf_url, "terminé", model_used, start_total)

        JOB_STATUS[job.job_id] = "terminé"
        JOB_RESULTS[job.job_id] = {
            "summary": final_summary,
            "mode": "page_par_page"
        }

    except Exception as e:
        print(f"❌ Erreur job {job.job_id} : {e}")
        JOB_STATUS[job.job_id] = "échec"
        JOB_RESULTS[job.job_id] = {"error": str(e)}

def lot_worker():
    lot_id = 0
    batch = []
    last_flush = time.time()

    while True:
        try:
            job = job_queue.get(timeout=1)
            batch.append(job)
        except queue.Empty:
            pass

        if batch and (len(batch) >= 5 or time.time() - last_flush > 10):
            lot_id += 1
            print(f"\n📦 Lancement du lot #{lot_id} ({len(batch)} jobs)")
            print("—" * 40)
            for job in batch:
                process_job(job)
            print(f"✅ Lot #{lot_id} terminé")
            print("═" * 40)
            batch.clear()
            last_flush = time.time()

threading.Thread(target=lot_worker, daemon=True).start()
