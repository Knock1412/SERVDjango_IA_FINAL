import os
import json
import time
from datetime import datetime

def log_job_history(job_id, entreprise, pdf_url, status, model_used, start_time):
    """
    Crée un historique de traitement IA journalier, par entreprise, dans le dossier :
    cache_json/YYYY-MM-DD/NOM_ENTREPRISE/job_history.json
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")
    base_path = os.path.join("cache_json", today, entreprise)
    os.makedirs(base_path, exist_ok=True)

    path = os.path.join(base_path, "job_history.json")

    try:
        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    job_data = {
        "job_id": job_id,
        "pdf_url": pdf_url,
        "status": status,
        "model_global": model_used,
        "created_at": datetime.utcnow().isoformat(),
        "duration": round(time.time() - start_time, 2)
    }

    history.append(job_data)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"📝 Historique mis à jour pour {entreprise} ({today}) : {job_id}")
