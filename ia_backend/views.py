import uuid
import requests
import os
import shutil
import json
import time
import hashlib
from urllib.parse import urlparse

from rest_framework.decorators import api_view
from rest_framework.response import Response

from ia_backend.services.pdf_utils import extract_blocks_from_pdf
from ia_backend.services.backup_service import load_global_summary_if_exists
from ia_backend.job_queue import job_queue, Job, JOB_RESULTS

CACHE_DIR = "temp_cache"

def extract_filename_from_url(pdf_url):
    return f"doc_{hashlib.md5(pdf_url.encode()).hexdigest()[:12]}"

@api_view(["POST"])
def summarize_from_url(request):
    start_total = time.time()  # DÉBUT DU TIMER
    job_id = str(uuid.uuid4())

    pdf_url = request.data.get("url", "")
    entreprise = request.data.get("entreprise", "anonyme")

    if not pdf_url:
        return Response({"error": "URL manquante."}, status=400)

    folder_name = extract_filename_from_url(pdf_url)
    job_folder = f"{folder_name}_{job_id[:8]}"
    folder_cache_dir = os.path.join(CACHE_DIR, job_folder)
    os.makedirs(folder_cache_dir, exist_ok=True)
    local_pdf_path = os.path.join(folder_cache_dir, f"{folder_name}.pdf")

    # Vérifie s'il existe déjà un résumé
    existing_summary = load_global_summary_if_exists(entreprise, folder_name)
    if existing_summary:
        duration_total = round(time.time() - start_total, 2)
        print(f"✅ Résumé depuis cache en {duration_total} sec")
        return Response({"summary": existing_summary, "job_id": job_id, "mode": "cache"})

    try:
        print("📄 Téléchargement du PDF...")
        pdf_response = requests.get(pdf_url)

        if pdf_response.status_code != 200:
            raise ValueError("Erreur HTTP")
        if b'%PDF' not in pdf_response.content[:1024]:
            raise ValueError("Signature PDF manquante")

        with open(local_pdf_path, "wb") as f:
            f.write(pdf_response.content)
        print(f"📥 PDF sauvegardé dans : {local_pdf_path}")

    except Exception as e:
        return Response({"error": "PDF invalide.", "detail": str(e)}, status=400)

    # 📏 Estimation du nombre de blocs pour priorisation
    total_pages = extract_blocks_from_pdf(local_pdf_path, return_pages_only=True)
    estimated_blocks = total_pages if total_pages < 10 else total_pages // 3

    # 📨 Ajout à la file
    job = Job(
        priority=estimated_blocks,
        job_id=job_id,
        entreprise=entreprise,
        pdf_path=local_pdf_path,
        pdf_url=pdf_url,
        folder_name=folder_name
    )
    job_queue.put(job)
    print(f"📨 Job {job_id} ajouté à la file (priorité = {estimated_blocks})")
    print(f"⏳ En attente de la fin du job {job_id}...")

    # 🕓 Attente bloquante
    while True:
        result = JOB_RESULTS.get(job_id)
        if result:
            duration_total = round(time.time() - start_total, 2)
            print(f"✅ Traitement total terminé en {duration_total} sec")
            return Response({
                "summary": result.get("summary", ""),
                "job_id": job_id,
                "mode": result.get("mode", ""),
                "duration": duration_total
            })
        time.sleep(2)


@api_view(["POST"])
def ask_from_url(request):
    from ia_backend.services.qa_engine import answer_question_from_pdf

    pdf_url = request.data.get("url", "")
    question = request.data.get("question", "")
    if not pdf_url or not question:
        return Response({"error": "URL ou question manquante ou invalide."}, status=400)
    try:
        answer = answer_question_from_pdf(requests.get(pdf_url).content, question)
        return Response({"answer": answer})
    except Exception as e:
        return Response({"error": str(e)}, status=500)
