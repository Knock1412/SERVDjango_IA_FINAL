import threading
import queue
import requests

# File d’attente pour les requêtes IA
ollama_queue = queue.Queue()
OLLAMA_MAX_WORKERS = 2  # Nbre max de requêtes concurrentes

def ollama_worker():
    while True:
        fn, result_queue = ollama_queue.get()
        try:
            result = fn()
            result_queue.put(result)
        except Exception as e:
            print(f"⚠️ Worker exception : {e}")
            result_queue.put(("Échec de génération IA", None))
        finally:
            ollama_queue.task_done()

# Démarrage des workers
for _ in range(OLLAMA_MAX_WORKERS):
    threading.Thread(target=ollama_worker, daemon=True).start()

def generate_ollama(prompt, num_predict, models=None):
    models = models or ["mistral"]

    for model in models:
        result_queue = queue.Queue()

        def task():
            res = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": num_predict}
                },
               
            )
            summary = res.json().get("response", "").strip()
            return (summary, model) if summary else ("", None)

        # Enfile la tâche
        ollama_queue.put((task, result_queue))
        summary, model_used = result_queue.get()

        if summary:
            return summary, model_used

        print(f"⚠️ Erreur modèle {model} : Timeout ou réponse vide")

    return "Échec de génération IA", None
