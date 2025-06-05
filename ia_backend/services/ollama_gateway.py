import threading
import queue
import requests
from typing import List, Tuple, Optional
import logging
from typing import Optional
import time

# Initialisation du logger (à mettre au début du fichier)
logger = logging.getLogger(__name__)
# --------- Configuration ---------
OLLAMA_MAX_WORKERS = 2  # Parallelisme
OLLAMA_TIMEOUT = 30     # Timeout API (sec)
DEFAULT_MODELS = ["mistral", "phi3"]

# --------- File d'attente ---------
ollama_queue = queue.Queue()

# --------- Worker ---------
def ollama_worker():
    """Worker qui traite les requêtes Ollama en boucle"""
    while True:
        try:
            task, result_queue = ollama_queue.get()
            result = task()
            result_queue.put(result)
        except requests.exceptions.RequestException as e:
            print(f"🌐 Erreur réseau Ollama: {str(e)}")
            result_queue.put(("Erreur réseau", None))
        except Exception as e:
            print(f"⚠️ Erreur worker: {str(e)}")
            result_queue.put(("Erreur interne", None))
        finally:
            ollama_queue.task_done()

# --------- Fonction Principale ---------
def generate_ollama(
    prompt: str,
    num_predict: int = 800,
    models: Optional[List[str]] = None,
    temperature: float = 0.7,
    top_k: int = 40
) -> str:  # Retourne directement le texte
    """Version simplifiée retournant uniquement le texte généré"""
    models = models or DEFAULT_MODELS
    
    for model in models:
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": num_predict,
                        "temperature": temperature,
                        "top_k": top_k
                    }
                },
                timeout=OLLAMA_TIMEOUT
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
            
        except Exception as e:
            logger.warning(f"Échec modèle {model}: {str(e)}")
            continue
            
    logger.error("Tous les modèles ont échoué")
    return ""

# --------- Initialisation ---------
for _ in range(OLLAMA_MAX_WORKERS):
    threading.Thread(target=ollama_worker, daemon=True).start()