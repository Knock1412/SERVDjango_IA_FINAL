from celery import shared_task
from .job_queue import process_job, Job

@shared_task(bind=True)
def process_job_task(self, job_data):
    """
    Tâche Celery IA principale.
    - bind=True => pour accès aux retries / logs Celery natifs
    """
    try:
        # reconstruction de l'objet Job depuis le dictionnaire reçu
        job = Job(**job_data)
        process_job(job)

    except Exception as e:
        # log d'erreur possible (bonus pour debug production)
        print(f"❌ Erreur dans process_job_task : {e}")
        raise self.retry(exc=e, countdown=10, max_retries=3)
