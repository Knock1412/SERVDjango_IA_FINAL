from celery import shared_task
from .job_queue import process_job, Job

@shared_task(bind=True)
def process_job_task(self, job_data):
    """
    Tâche Celery IA principale (version prod scalable).
    - bind=True => accès aux retries, logs, etc.
    - Retourne directement le résumé final à Celery
    """
    try:
        # Reconstruction de l'objet Job
        job = Job(**job_data)

        # On exécute directement le traitement
        result = process_job(job)  # 👉 attention, process_job doit renvoyer le résultat final

        return {
            "job_id": job.job_id,
            "summary": result.get("summary", "(résumé indisponible)"),
            "mode": result.get("mode", "unknown")
        }

    except Exception as e:
        # log d'erreur + retry automatique
        self.retry(exc=e, countdown=10, max_retries=3)
