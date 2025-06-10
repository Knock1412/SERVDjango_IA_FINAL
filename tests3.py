import requests
import time
import sys

pdf_url = "https://edutice.hal.science/edutice-00001245v1/document"
entreprise = "Entreprise_S3_Test"
API_URL = "http://127.0.0.1:8000"

print("📥 Envoi du PDF S3 pour résumé...\n")

response = requests.post(
    f"{API_URL}/summarize_from_url/",
    json={"url": pdf_url, "entreprise": entreprise}
)

if response.status_code == 200:
    data = response.json()

    if data.get("mode") == "cache":
        print("✅ Résumé chargé directement depuis le cache :\n")
        print(data["summary"])
    else:
        job_id = data.get("job_id", "-")
        task_id = data.get("task_id")

        print(f"🚀 Tâche Celery lancée : job_id={job_id} | task_id={task_id}")
        print("⏳ Attente du résultat...\n")

        spinner = ["|", "/", "-", "\\"]
        i = 0

        while True:
            status_response = requests.get(f"{API_URL}/get_summarize_status/{task_id}/")
            status_data = status_response.json()

            if status_data["status"] == "completed":
                print("\n✅ Résumé généré :\n")
                print(status_data.get("summary", "(aucun résumé)"))
                break
            elif status_data["status"] == "failed":
                print("\n❌ Erreur durant le traitement :", status_data.get("error"))
                break
            else:
                sys.stdout.write(f"\r🔄 Statut actuel : {status_data['status']} {spinner[i % len(spinner)]}")
                sys.stdout.flush()
                i += 1
                time.sleep(1.5)
else:
    print(f"❌ Erreur HTTP {response.status_code} :")
    print(response.text)
