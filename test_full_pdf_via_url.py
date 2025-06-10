import requests
import threading
import time
import sys

# 🗂️ Liste des 5 PDF à tester
pdfs = [
    ("https://edutice.hal.science/edutice-00000852v1/document", "Entreprise_1"),
    ("https://edutice.hal.science/edutice-00000413v1/document", "Entreprise_2"),
    ("https://edutice.hal.science/edutice-00000862v1/document", "Entreprise_3"),
    ("https://edutice.hal.science/edutice-00001410v1/document", "Entreprise_4"),
    ("https://edutice.hal.science/edutice-00001245v1/document", "Entreprise_5"),
]

API_URL = "http://127.0.0.1:8000"

class Job:
    def __init__(self, url, entreprise, label):
        self.url = url
        self.entreprise = entreprise
        self.label = label
        self.task_id = None
        self.finished = False
        self.summary = ""
        self.mode = ""

    def start(self):
        try:
            res = requests.post(f"{API_URL}/summarize_from_url/", json={
                "url": self.url,
                "entreprise": self.entreprise
            })
            if res.status_code == 200:
                data = res.json()
                self.task_id = data.get("task_id")
                print(f"🚀 [{self.label}] Tâche Celery lancée : job_id={data.get('job_id')} | task_id={self.task_id}")
            else:
                print(f"❌ [{self.label}] Erreur HTTP {res.status_code}")
                self.finished = True
        except Exception as e:
            print(f"❌ [{self.label}] Exception au démarrage : {e}")
            self.finished = True

    def poll_status(self):
        if self.task_id is None:
            return
        try:
            res = requests.get(f"{API_URL}/get_summarize_status/{self.task_id}/")
            if res.status_code == 200:
                data = res.json()
                if data["status"] == "completed":
                    self.summary = data.get("summary", "")
                    self.mode = data.get("mode", "")
                    self.finished = True
                elif data["status"] == "failed":
                    self.summary = "❌ Erreur de traitement"
                    self.finished = True
            else:
                self.summary = f"Erreur HTTP polling {res.status_code}"
                self.finished = True
        except Exception as e:
            self.summary = f"Erreur polling {e}"
            self.finished = True

# Préparation des jobs
jobs = [Job(url, entreprise, f"PDF {i+1}") for i, (url, entreprise) in enumerate(pdfs)]

# Lancement initial en mode multithread
threads = []
for job in jobs:
    print(f"\n📤 [{job.label}] {job.entreprise} envoie un document...")
    t = threading.Thread(target=job.start)
    t.start()
    threads.append(t)

# On attend que toutes les requêtes soient envoyées
for t in threads:
    t.join()

# Boucle de monitoring centralisée propre
while not all(j.finished for j in jobs):
    for job in jobs:
        if not job.finished:
            job.poll_status()
    sys.stdout.write("\r")
    sys.stdout.write("🔄 " + " | ".join([f"{j.label}:{'✅' if j.finished else '⏳'}" for j in jobs]))
    sys.stdout.flush()
    time.sleep(2)

print("\n\n🎯 Tous les jobs sont terminés.\n")

# Résultats finaux
for job in jobs:
    print(f"📄 [{job.label}] Résumé final (mode={job.mode}):")
    print(job.summary[:1000] + ("..." if job.summary and len(job.summary) > 1000 else ""))
    print("-" * 80)
