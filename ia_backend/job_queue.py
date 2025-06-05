from typing import Tuple, List, Dict, Any, Optional
import threading
import queue
import time
import os
from dataclasses import dataclass
from datetime import datetime

from ia_backend.services.pdf_utils import (
    chunk_pdf as smart_chunking,
    extract_text_hybrid,
    AnnexDetector,
    PDFMetrics,
    process_pdf
)
from ia_backend.services.summarizer import (
    summarize_block,
    summarize_global,
    evaluate_summary_score
)
from ia_backend.services.cache_manager import save_json
from ia_backend.services.backup_service import save_global_summary
from ia_backend.services.job_logger import log_job_history
from ia_backend.services.language_detection_and_translation import process_text_block

# --------- Configuration ---------
@dataclass
class JobConfig:
    BLOCK_THRESHOLD_INITIAL: float = 0.7
    BLOCK_THRESHOLD_SECONDARY: float = 0.65
    INTERMEDIATE_GROUP_SIZE: int = 5
    MAX_ATTEMPTS_INITIAL: int = 4
    MAX_ATTEMPTS_SECONDARY: int = 4
    MAX_TEXT_LENGTH: int = 7000
    MIN_TEXT_LENGTH: int = 50
    INTERMEDIATE_NUM_PREDICT: int = 1000  # Pour les résumés intermédiaires
    FINAL_NUM_PREDICT: int = 1500         # Pour le résumé final

# --------- File d'attente ---------
job_queue = queue.PriorityQueue()
JOB_STATUS: Dict[str, str] = {}
JOB_RESULTS: Dict[str, Dict[str, Any]] = {}
BLOCK_SCORES: Dict[str, List[Dict[str, Any]]] = {}  # Stocke les scores par bloc

class Job:
    def __init__(self, priority: int, job_id: str, entreprise: str, 
                 pdf_path: str, pdf_url: str, folder_name: str):
        self.priority = priority
        self.job_id = job_id
        self.entreprise = entreprise
        self.pdf_path = pdf_path
        self.pdf_url = pdf_url
        self.folder_name = folder_name

    def __lt__(self, other):
        return self.priority < other.priority

# --------- Logging Utilities ---------
def log_step(message: str, level: str = "info"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    colors = {
        "info": "\033[94m",    # Bleu
        "success": "\033[92m", # Vert
        "warning": "\033[93m", # Jaune
        "error": "\033[91m",   # Rouge
        "debug": "\033[90m"    # Gris
    }
    print(f"{colors.get(level, '')}[{timestamp}] {message}\033[0m")

# --------- Traitement principal ---------
def process_job(job: Job) -> None:
    try:
        JOB_STATUS[job.job_id] = "en cours"
        BLOCK_SCORES[job.job_id] = []  # Initialise le stockage des scores
        start_time = time.time()
        log_step(f"Début traitement job {job.job_id} - Priorité: {job.priority}", "info")
        
        # Traitement PDF
        log_step(f"Analyse du PDF: {os.path.basename(job.pdf_path)}", "info")
        result = process_pdf(job.pdf_path)
        log_step(f"PDF analysé - {result['metrics']['total_pages']} pages trouvées", "success")
        
        summaries = []
        for page_num, text in result['pages']:
            page_log = f"Page {page_num + 1}/{result['metrics']['total_pages']}"
            
            if len(text) < JobConfig.MIN_TEXT_LENGTH:
                log_step(f"{page_log} ignorée (trop courte: {len(text)} caractères)", "warning")
                continue

            if len(text) > JobConfig.MAX_TEXT_LENGTH:
                text = text[:JobConfig.MAX_TEXT_LENGTH]
                log_step(f"{page_log} tronquée à {JobConfig.MAX_TEXT_LENGTH} caractères", "debug")

            log_step(f"Traitement {page_log}...", "info")
            best_summary, best_score = generate_best_summary(
                text, 
                page_num,
                job.entreprise,
                job.folder_name,
                job.job_id  # Passer job_id pour le suivi des scores
            )
            
            if best_summary:
                summaries.append((page_num, best_summary))
                log_step(f"{page_log} traitée - Score: {best_score:.2f}", "success")
                # Enregistrement du score pour ce bloc
                BLOCK_SCORES[job.job_id].append({
                    'page': page_num + 1,
                    'score': best_score,
                    'summary_length': len(best_summary)
                })

        # Fusion finale
        if summaries:
            log_step(f"Fusion des {len(summaries)} résumés partiels...", "info")
            final_summary = summarize_final(
                summaries,
                result['metrics'],
                job
            )
            
            duration = time.time() - start_time
            log_step(f"Job {job.job_id} terminé en {duration:.2f}s", "success")
            
            # Ajout des scores moyens dans les résultats
            scores = [s['score'] for s in BLOCK_SCORES[job.job_id]]
            avg_score = sum(scores) / len(scores) if scores else 0
            
            JOB_STATUS[job.job_id] = "terminé"
            JOB_RESULTS[job.job_id] = {
                "summary": final_summary,
                "metrics": result['metrics'],
                "block_scores": BLOCK_SCORES[job.job_id],  # Scores détaillés
                "average_score": avg_score,  # Score moyen
                "mode": "page_by_page_v1"
            }

    except Exception as e:
        log_step(f"ERREUR job {job.job_id}: {str(e)}", "error")
        JOB_STATUS[job.job_id] = "échec"
        JOB_RESULTS[job.job_id] = {"error": str(e)}

def generate_best_summary(text: str, page_num: int, entreprise: str, folder_name: str, job_id: str) -> Tuple[str, float]:
    """Génère le meilleur résumé avec plusieurs tentatives"""
    best_score = 0.0
    best_summary = ""
    
    for attempt in range(1, JobConfig.MAX_ATTEMPTS_INITIAL + 1):
        log_step(f"Tentative {attempt}/{JobConfig.MAX_ATTEMPTS_INITIAL} pour page {page_num + 1}", "debug")
        
        try:
            # 1. Génération du résumé
            summary_content = summarize_block(text)
            if not summary_content:
                log_step(f"Échec summarize_block (tentative {attempt}) - résumé vide", "warning")
                continue
                
            # 2. Traitement du résumé
            processed_summary, _ = process_text_block(summary_content)
            if not processed_summary:
                log_step(f"Échec process_text_block (tentative {attempt}) - résultat vide", "warning")
                continue
                
            # 3. Évaluation
            score = evaluate_summary_score(text, processed_summary)
            log_step(f"Score obtenu: {score:.2f} (tentative {attempt})", "debug")
            
            if score > best_score:
                best_score = score
                best_summary = processed_summary
                log_step(f"Nouveau meilleur score: {best_score:.2f}", "debug")
                
            if best_score >= JobConfig.BLOCK_THRESHOLD_INITIAL:
                break
                
        except Exception as e:
            log_step(f"Erreur lors de la tentative {attempt}: {str(e)}", "error")
            continue
        
        if score > best_score:
            best_score = score
            best_summary = processed_summary
        
        if best_score >= JobConfig.BLOCK_THRESHOLD_INITIAL:
            log_step(f"Score suffisant atteint ({best_score:.2f})", "debug")
            break

    if best_summary:
        save_json(
            f"cache_json/save_summaryblocks/{entreprise}/{folder_name}",
            page_num,
            {
                "page": page_num + 1,
                "summary": best_summary,
                "score": best_score,
                "job_id": job_id
            }
        )
    
    return best_summary, best_score

def summarize_final(summaries: List[Tuple[int, str]], metrics: Dict, job: Job) -> str:
    """Fusionne les résumés et sauvegarde les résultats intermédiaires"""
    summaries.sort()
    text_list = [s for _, s in summaries]
    
    # Dossier pour les intermédiaires
    intermediate_dir = f"cache_json/save_summaryintermediates/{job.entreprise}/{job.folder_name}"
    os.makedirs(intermediate_dir, exist_ok=True)
    
    # Fusion intermédiaire
    intermediates = []
    for i in range(0, len(text_list), JobConfig.INTERMEDIATE_GROUP_SIZE):
        group = text_list[i:i + JobConfig.INTERMEDIATE_GROUP_SIZE]
        intermediate = summarize_global(group, num_predict=JobConfig.INTERMEDIATE_NUM_PREDICT)
        
        # Sauvegarde de chaque groupe intermédiaire
        save_json(
            intermediate_dir,
            f"intermediate_{i//JobConfig.INTERMEDIATE_GROUP_SIZE}",
            {
                "pages": f"{i}-{min(i+JobConfig.INTERMEDIATE_GROUP_SIZE, len(text_list))}",
                "content": intermediate,
                "timestamp": datetime.now().isoformat()
            }
        )
        intermediates.append(intermediate)
    
    # Fusion finale
    final_summary = summarize_global(
        intermediates,
        num_predict=JobConfig.FINAL_NUM_PREDICT,
        is_final=True
    )
    
    # Sauvegarde finale
    save_global_summary(job.entreprise, job.folder_name, final_summary)
    
    return final_summary

# --------- Worker ---------
def batch_worker():
    """Traite les jobs par lots pour optimiser les ressources"""
    batch = []
    last_process = time.time()
    
    while True:
        try:
            job = job_queue.get(timeout=1)
            batch.append(job)
            log_step(f"Job {job.job_id} ajouté au lot actuel", "debug")
        except queue.Empty:
            pass
        
        if batch and (len(batch) >= 5 or time.time() - last_process > 10):
            log_step(f"Traitement du lot de {len(batch)} jobs", "info")
            for job in batch:
                process_job(job)
            log_step("Lot terminé", "success")
            batch.clear()
            last_process = time.time()

# Démarrage
log_step("Service job_queue démarré", "success")
threading.Thread(target=batch_worker, daemon=True).start()