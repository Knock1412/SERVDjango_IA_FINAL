import uuid
import requests
import os
import json
import time
import hashlib
import logging
from urllib.parse import urlparse

from rest_framework.decorators import api_view
from rest_framework.response import Response

from ia_backend.services.pdf_utils import extract_blocks_from_pdf, detect_annex_start_page
from ia_backend.services.backup_service import load_global_summary_if_exists, save_global_summary
from ia_backend.job_queue import Job
from ia_backend.tasks import process_job_task

from ia_backend.ask_engine import load_all_blocks, find_relevant_blocks, generate_answer
from celery.result import AsyncResult

# ---------- Logging centralisé ----------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

CACHE_DIR = "temp_cache"

def extract_filename_from_url(pdf_url):
    return f"doc_{hashlib.md5(pdf_url.encode()).hexdigest()[:12]}"

@api_view(["POST"])
def summarize_from_url(request):
    start_total = time.time()

    pdf_url = request.data.get("url", "")
    entreprise = request.data.get("entreprise", "anonyme")

    if not pdf_url:
        return Response({"error": "URL manquante."}, status=400)

    folder_name = extract_filename_from_url(pdf_url)
    local_pdf_filename = f"{folder_name}.pdf"

    # ✅ Vérif de cache AVANT génération de job_id
    summary_data = load_global_summary_if_exists(entreprise, folder_name)
    if summary_data:
        summary = summary_data.get("summary")
        cached_job_id = summary_data.get("job_id", folder_name)  # fallback = dossier
        duration_total = round(time.time() - start_total, 2)
        logger.info(f"✅ Résumé chargé depuis le cache en {duration_total}s")
        return Response({
            "summary": summary,
            "job_id": cached_job_id,
            "mode": "cache",
            "duration": duration_total
        })

    # ❌ Pas trouvé en cache => générer job_id
    job_id = str(uuid.uuid4())
    job_folder = f"{folder_name}_{job_id[:8]}"
    folder_cache_dir = os.path.join(CACHE_DIR, job_folder)
    os.makedirs(folder_cache_dir, exist_ok=True)
    local_pdf_path = os.path.join(folder_cache_dir, local_pdf_filename)

    try:
        logger.info(f"📄 Téléchargement du PDF depuis {pdf_url}...")
        pdf_response = requests.get(pdf_url, timeout=30)

        if pdf_response.status_code != 200:
            raise ValueError("Erreur HTTP")
        if b'%PDF' not in pdf_response.content[:1024]:
            raise ValueError("Signature PDF manquante")

        with open(local_pdf_path, "wb") as f:
            f.write(pdf_response.content)
        logger.info(f"📥 PDF sauvegardé dans {local_pdf_path}")

    except Exception as e:
        logger.error(f"Erreur téléchargement PDF: {e}")
        return Response({"error": "PDF invalide.", "detail": str(e)}, status=400)

    try:
        total_pages = extract_blocks_from_pdf(local_pdf_path, return_pages_only=True)
        annex_page = detect_annex_start_page(local_pdf_path)
        effective_pages = annex_page if annex_page else total_pages
        estimated_blocks = effective_pages if effective_pages < 10 else effective_pages // 3
    except Exception as e:
        logger.warning(f"Erreur lecture préliminaire PDF : {e}")
        estimated_blocks = 10

    job_data = {
        "priority": estimated_blocks,
        "job_id": job_id,
        "entreprise": entreprise,
        "pdf_path": local_pdf_path,
        "pdf_url": pdf_url,
        "folder_name": folder_name
    }

    logger.info(f"📨 Job {job_id} envoyé à Celery (priorité={estimated_blocks})")

    task = process_job_task.apply_async(args=[job_data])

    return Response({
        "job_id": job_id,
        "task_id": task.id,
        "status": "processing"
    })

@api_view(["GET"])
def get_summarize_status(request, task_id):
    res = AsyncResult(task_id)

    if res.state == 'PENDING':
        return Response({"status": "pending"})
    elif res.state == 'STARTED':
        return Response({"status": "processing"})
    elif res.state == 'SUCCESS':
        result_data = res.result or {}
        return Response({
            "status": "completed",
            "summary": result_data.get("summary", ""),
            "mode": result_data.get("mode", ""),
        })
    elif res.state == 'FAILURE':
        logger.error(f"Erreur Celery: {res.result}")
        return Response({
            "status": "failed",
            "error": str(res.result)
        }, status=500)
    else:
        return Response({"status": res.state})

@api_view(["POST"])
def ask_from_url(request):
    question = request.data.get("question")
    job_id = request.data.get("job_id")
    entreprise = request.data.get("entreprise")

    if not question or not job_id or not entreprise:
        return Response({"error": "question, job_id et entreprise sont requis."}, status=400)

    try:
        blocks = load_all_blocks(entreprise, job_id)
    except FileNotFoundError:
        return Response({"error": "Blocs non trouvés pour ce job"}, status=404)
    except Exception as e:
        return Response({"error": f"Erreur lors du chargement des blocs : {str(e)}"}, status=500)

    if not blocks:
        return Response({"error": "Aucun bloc disponible pour ce document."}, status=204)

    selected_blocks = find_relevant_blocks(question, blocks)
    start_time = time.time()
    answer = generate_answer(question, selected_blocks)
    duration = round(time.time() - start_time, 2)
    logger.info(f"🕒 Temps de génération de la réponse : {duration}s")

    return Response({
        "question": question,
        "answer": answer,
        "job_id": job_id,
        "entreprise": entreprise
    })
