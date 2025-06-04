import threading
import queue
import time
import os
import math

from ia_backend.services.pdf_utils import (
    extract_blocks_from_pdf,
    extract_text_from_block,
    extract_full_text,
    detect_annex_start_page,  # 👈 NOUVELLE IMPORTATION
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

# Paramètres IA et industriels
BLOCK_THRESHOLD_INITIAL = 0.7
BLOCK_THRESHOLD_SECONDARY = 0.65
INTERMEDIATE_GROUP_SIZE = 5
MAX_ATTEMPTS_INITIAL = 4
MAX_ATTEMPTS_SECONDARY = 4

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
        print(f"🧩 Mode : découpage dynamique hiérarchique | Total pages : {total_pages}")

        # 🔎 Nouvelle détection d'annexe
        annex_start_page = detect_annex_start_page(job.pdf_path)

        blocks = extract_blocks_from_pdf(job.pdf_path)
        summaries = []

        for idx, block_indexes in enumerate(blocks):
            print(f"⏳ Bloc {idx + 1} en cours...")
            text = extract_text_from_block(block_indexes, job.pdf_path)
            if not text.strip():
                text = "(Bloc vide ou inexploitable.)"

            print(f"📊 Bloc {idx+1} — Longueur brute : {len(text.strip())} caractères")

            if is_likely_annex(text, block_indexes, total_pages, annex_start_page):
                print(f"🚩 Bloc {idx+1} détecté comme ANNEXE via is_likely_annex(). Ignoré.")
                continue

            if len(text) > 7000:
                print(f"✂️ Bloc {idx+1} pré-tronqué à 7000 avant IA (brut={len(text)})")
                text = text[:7000]

            best_score = 0
            best_summary = ""
            attempt = 0

            while attempt < MAX_ATTEMPTS_INITIAL:
                attempt += 1

                summary, model = summarize_block(text)
                processed_summary, translated = process_text_block(summary)

                if translated:
                    print(f"🌍 Bloc {idx+1} traduit de l'anglais vers le français (avant scoring).")
                else:
                    print(f"🌍 Bloc {idx+1} déjà en français avant scoring.")

                score = evaluate_summary_score(text, processed_summary)
                print(f"⚙️ Bloc #{idx+1} Essai {attempt} - Score = {round(score,3)} - Meilleur = {round(best_score,3)}")

                if score > best_score:
                    best_score = score
                    best_summary = processed_summary

                if best_score >= BLOCK_THRESHOLD_INITIAL:
                    print(f"✅ Bloc {idx+1} validé avec score {round(best_score,3)} après {attempt} tentatives")
                    break
                else:
                    print("🔁 Reprocessing...")

            if best_score < BLOCK_THRESHOLD_INITIAL:
                print(f"⚠️ Passage seuil secondaire pour bloc {idx+1}")

                if best_score >= BLOCK_THRESHOLD_SECONDARY:
                    print(f"✅ Bloc {idx+1} validé immédiatement au score secondaire {round(best_score,3)}")
                else:
                    secondary_attempt = 0
                    while secondary_attempt < MAX_ATTEMPTS_SECONDARY:
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
            print(f"🔄 Fusion intermédiaire du lot {i//INTERMEDIATE_GROUP_SIZE + 1} contenant {len(group)} résumés...")

            intermediate_summary, model_used = summarize_global(group, num_predict=1000)
            processed_summary, translated = process_text_block(intermediate_summary)

            inter_json_dir = f"cache_json/save_summaryintermediates/{job.entreprise}/{job.folder_name}"
            os.makedirs(inter_json_dir, exist_ok=True)
            save_json(inter_json_dir, i//INTERMEDIATE_GROUP_SIZE, {
                "intermediate_block": i//INTERMEDIATE_GROUP_SIZE + 1,
                "summary": processed_summary
            })

            intermediates.append(processed_summary)

        print(f"🔍 Fusion finale sur {len(intermediates)} résumés intermédiaires...")

        final_predict_len = 1300 if len(intermediates) <= 10 else 1500

        final_summary, model_used = summarize_global(intermediates, num_predict=final_predict_len, is_final=True)
        global_score = evaluate_summary_score(full_pdf_text, final_summary, partial_summaries=intermediates)
        print(f"📊 Score résumé global (info only) = {round(global_score, 3)}")

        save_global_summary(job.entreprise, job.folder_name, final_summary)
        log_job_history(job.job_id, job.entreprise, job.pdf_url, "terminé", model_used, start_total)

        JOB_STATUS[job.job_id] = "terminé"
        JOB_RESULTS[job.job_id] = {
            "summary": final_summary,
            "mode": "hierarchical_v4.2_annex_smart"
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
