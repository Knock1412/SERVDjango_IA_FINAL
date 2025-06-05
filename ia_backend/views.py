import uuid
import requests
import os
import time
import hashlib
import logging
from typing import Dict

from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse

from ia_backend.services.pdf_utils import (
    chunk_pdf as extract_blocks_from_pdf,
    extract_text_hybrid
)
from ia_backend.services.backup_service import load_global_summary_if_exists
from ia_backend.job_queue import job_queue, Job, JOB_RESULTS
from ia_backend.services.qa_engine import answer_question_from_pdf

# Configuration du logger
logger = logging.getLogger(__name__)

# --------- Configuration ---------
CACHE_DIR = "temp_cache"
MAX_TIMEOUT = 1800  # 30 minutes
MAX_PRIORITY = 20   # Priorité maximale

# --------- Utilitaires ---------
def get_document_hash(pdf_url: str) -> str:
    """Génère un hash unique pour l'URL"""
    return hashlib.md5(pdf_url.encode()).hexdigest()[:12]

def validate_pdf(content: bytes) -> bool:
    """Vérifie que le contenu est bien un PDF"""
    return content.startswith(b'%PDF') and len(content) > 1024

# --------- Views ---------
@api_view(["POST"])
def summarize_from_url(request) -> JsonResponse:
    """
    Endpoint pour résumer un PDF depuis une URL
    """
    logger.info("Requête reçue sur /summarize_from_url/")
    logger.debug(f"Headers: {request.headers}")
    logger.debug(f"Données reçues: {request.data}")

    try:
        # Validation des entrées
        pdf_url = request.data["url"]
        entreprise = request.data.get("entreprise", "default")
        logger.info(f"Traitement PDF: {pdf_url} pour {entreprise}")

        # Vérification du cache
        doc_hash = get_document_hash(pdf_url)
        logger.debug(f"Hash document: {doc_hash}")
        
        if cached_summary := load_global_summary_if_exists(entreprise, doc_hash):
            logger.info("Résumé trouvé en cache")
            return JsonResponse({
                "summary": cached_summary,
                "cache": True,
                "job_id": str(uuid.uuid4())
            })

        # Téléchargement sécurisé
        logger.info("Téléchargement du PDF...")
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()
        
        if not validate_pdf(response.content):
            logger.error("Contenu PDF invalide")
            raise ValueError("Contenu PDF invalide")

        # Sauvegarde temporaire
        os.makedirs(CACHE_DIR, exist_ok=True)
        pdf_path = os.path.join(CACHE_DIR, f"{doc_hash}.pdf")
        logger.debug(f"Sauvegarde temporaire: {pdf_path}")
        
        with open(pdf_path, "wb") as f:
            f.write(response.content)

        # Estimation de la priorité
        chunks = extract_blocks_from_pdf(pdf_path)
        priority = min(len(chunks), MAX_PRIORITY)
        logger.info(f"Nombre de chunks: {len(chunks)}, Priorité: {priority}")

        # Création et soumission du job
        job_id = str(uuid.uuid4())
        job = Job(
            priority=priority,
            job_id=job_id,
            entreprise=entreprise,
            pdf_path=pdf_path,
            pdf_url=pdf_url,
            folder_name=doc_hash
        )
        job_queue.put(job)
        logger.info(f"Job {job_id} soumis à la file d'attente")

        # Attente avec timeout
        start_time = time.time()
        logger.debug("Attente du résultat...")
        
        while (time.time() - start_time) < MAX_TIMEOUT:
            if job_id in JOB_RESULTS:
                result = JOB_RESULTS[job_id]
                logger.info(f"Job {job_id} terminé en {time.time() - start_time:.2f}s")
                return JsonResponse({
                    "summary": result["summary"],
                    "job_id": job_id,
                    "duration": round(time.time() - start_time, 2),
                    "mode": result.get("mode", "default")
                })
            time.sleep(2)

        logger.error("Timeout du job")
        raise TimeoutError("Temps de traitement dépassé")

    except KeyError as e:
        logger.error(f"Paramètre manquant: {str(e)}")
        return JsonResponse({"error": "Le paramètre 'url' est requis"}, status=400)
    except Exception as e:
        logger.error(f"Erreur inattendue: {str(e)}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)

@api_view(["POST"])
def ask_from_url(request) -> JsonResponse:
    """
    Endpoint pour poser une question sur un PDF
    """
    logger.info("Requête reçue sur /ask_from_url/")
    logger.debug(f"Headers: {request.headers}")
    logger.debug(f"Données reçues: {request.data}")

    try:
        pdf_url = request.data["url"]
        question = request.data["question"]
        logger.info(f"Question sur PDF: {pdf_url} - Question: {question[:50]}...")
        
        # Téléchargement direct
        pdf_content = requests.get(pdf_url, timeout=10).content
        
        if not validate_pdf(pdf_content):
            logger.error("Contenu PDF invalide")
            raise ValueError("Contenu PDF invalide")
            
        answer = answer_question_from_pdf(pdf_content, question)
        logger.info("Réponse générée avec succès")
        return JsonResponse({"answer": answer})
    
    except KeyError as e:
        logger.error(f"Paramètre manquant: {str(e)}")
        return JsonResponse({"error": f"Paramètre manquant: {str(e)}"}, status=400)
    except Exception as e:
        logger.error(f"Erreur inattendue: {str(e)}", exc_info=True)
        return JsonResponse({"error": str(e)}, status=500)