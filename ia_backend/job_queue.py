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

# Seuil unique pour les blocs (le global n'est plus bloquant)
BLOCK_THRESHOLD = 0.75

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
        chunk_size = 1 if total_pages < 10 else 3
        mode = "page_par_page" if chunk_size == 1 else "bloc_par_3_pages"
        print(f"🧩 Mode : {mode} | Total pages : {total_pages}")

        blocks = extract_blocks_from_pdf(job.pdf_path, chunk_size=chunk_size)
        summaries = []

        for idx, block_indexes in enumerate(blocks):
            print(f"⏳ Bloc {idx + 1} en cours...")
            text = extract_text_from_block(block_indexes, job.pdf_path)
            char_count = len(text.strip()) if text else 0

            if char_count > 8000:
                print(f"✂️ Bloc {idx + 1} trop long ({char_count}) → tronqué à 8000")
                text = text[:8000]

            if not text.strip():
                text = "(Bloc vide ou inexploitable.)"

            attempts = 0
            best_score = 0
            best_summary = ""

            while True:
                summary, model = summarize_block(text)
                score = evaluate_summary_score(text, summary)
                attempts += 1

                print(f"⚙️ Tentative bloc #{idx+1} - Essai {attempts} - Score = {round(score,3)} / Meilleur = {round(best_score,3)}")

                if score > best_score:
                    best_score = score
                    best_summary = summary

                if best_score >= BLOCK_THRESHOLD:
                    print(f"✅ Bloc {idx + 1} validé avec score {round(best_score,3)} après {attempts} tentatives")
                    break
                else:
                    print(f"🔁 Reprocessing bloc {idx+1}...")

            json_dir = f"cache_json/save_summaryglobal/{job.entreprise}/save_summaryblock"
            save_json(json_dir, idx, {
                "bloc": idx + 1,
                "summary": best_summary,
                "source_pdf": job.pdf_url,
                "score": best_score
            })

            txt_dir = f"temp_cache/{job.folder_name}_blocks"
            save_txt(f"{txt_dir}/block_{idx+1}.txt", text)

            summaries.append((idx + 1, best_summary))

        summaries.sort()
        joined = [s for _, s in summaries]
        num_predict = determine_predict_length(len(joined))
        print(f"🔍 num_predict global = {num_predict}")

        final_summary, model_used = summarize_global(joined, num_predict=num_predict)
        global_score = evaluate_summary_score(full_pdf_text, final_summary, partial_summaries=joined)
        print(f"📊 Résumé global généré (score info = {round(global_score, 3)})")

        save_global_summary(job.entreprise, job.folder_name, final_summary)
        log_job_history(job.job_id, job.entreprise, job.pdf_url, "terminé", model_used, start_total)

        JOB_STATUS[job.job_id] = "terminé"
        JOB_RESULTS[job.job_id] = {
            "summary": final_summary,
            "mode": mode
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
