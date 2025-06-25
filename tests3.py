import requests
import time
import sys

API_URL = "http://192.168.10.121:8000"  # ⚠️ Assure-toi que cette IP est correcte
ENTREPRISE = "Entreprise_S3_Test"
PDF_URL = "https://edutice.hal.science/edutice-00000852v1/document"

def wait_for_summary(task_id: str):
    spinner = ["|", "/", "-", "\\"]
    i = 0
    while True:
        try:
            status_response = requests.get(f"{API_URL}/get_summarize_status/{task_id}/")
            status_data = status_response.json()

            if status_data["status"] == "completed":
                print(f"\n✅ Résumé généré pour task_id {task_id}.")
                print("📄 Résumé :\n")
                print(status_data.get("summary", "(Résumé vide)"))
                return True

            elif status_data["status"] == "failed":
                print(f"\n❌ Échec du traitement pour task_id {task_id} :", status_data.get("error"))
                return False

            else:
                sys.stdout.write(f"\r⏳ [task {task_id}] statut = {status_data['status']} {spinner[i % len(spinner)]}")
                sys.stdout.flush()
                i += 1
                time.sleep(1.5)

        except Exception as e:
            print(f"\n❌ Erreur réseau pour task_id {task_id} : {e}")
            return False

def process_pdf(url: str):
    print(f"\n📤 Envoi du PDF : {url}")
    try:
        response = requests.post(
            f"{API_URL}/summarize_from_url/",
            json={"url": url, "entreprise": ENTREPRISE}
        )

        print("↩️ Réponse brute :", response.text)

        if response.status_code != 200:
            print(f"❌ Erreur HTTP {response.status_code} : {response.text}")
            return

        data = response.json()
        job_id = data.get("job_id", "-")
        task_id = data.get("task_id")

        print(f"🔗 Job ID : {job_id}")

        if data.get("mode") == "cache":
            print("♻️ Résumé chargé depuis le cache.")
            print("📄 Résumé :\n", data.get("summary", "(Résumé manquant)"))
        else:
            print(f"🚀 Tâche Celery lancée pour job_id {job_id} / task_id {task_id}")
            success = wait_for_summary(task_id)
            if not success:
                print(f"⛔ Résumé non obtenu pour {url}")
                return

        print(f"\n✅ Document traité. Tu peux poser des questions sur ce job_id : {job_id}")

    except Exception as e:
        print(f"❌ Erreur inattendue pour {url} : {e}")

# ▶️ Lancement
if __name__ == "__main__":
    process_pdf(PDF_URL)
