import threading
import queue
import requests
from typing import List, Optional
import logging
import time

logger = logging.getLogger(__name__)

OLLAMA_MAX_WORKERS = 2
DEFAULT_MODELS = ["mistral"]

ollama_queue = queue.Queue()

def ollama_worker():
    while True:
        try:
            task, result_queue = ollama_queue.get()
            result = task()
            result_queue.put(result)
        except requests.exceptions.RequestException as e:
            print(f"🌐 Erreur réseau Ollama: {str(e)}")
            result_queue.put("Erreur réseau")
        except Exception as e:
            print(f"⚠️ Erreur worker: {str(e)}")
            result_queue.put("Erreur interne")
        finally:
            ollama_queue.task_done()

def generate_ollama(
    prompt: str,
    num_predict: int = 800,
    models: Optional[List[str]] = None,
    temperature: float = 0.7,
    top_k: int = 40
) -> str:
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
            )
            response.raise_for_status()
            json_response = response.json()
            response_text = json_response.get("response")

            if not response_text or not isinstance(response_text, str):
                logger.warning(f"Ollama a renvoyé une réponse vide ou invalide : {json_response}")
                continue

            return response_text.strip()

        except Exception as e:
            logger.warning(f"Échec modèle {model}: {str(e)}")
            continue

    logger.error("Tous les modèles ont échoué")
    return ""

def call_llm(prompt: str, model: Optional[str] = "mistral") -> str:
    """
    Wrapper simple pour générer une réponse avec un seul modèle.
    """
    return generate_ollama(prompt=prompt, models=[model])

# Démarrage des workers
for _ in range(OLLAMA_MAX_WORKERS):
    threading.Thread(target=ollama_worker, daemon=True).start()
