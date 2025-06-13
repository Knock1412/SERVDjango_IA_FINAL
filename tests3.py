import requests
import time
import sys
import uuid  # ✅ Pour session_id

pdf_url = "https://edutice.hal.science/edutice-00000862v1/document"
entreprise = "Entreprise_S3_Test"
API_URL = "http://127.0.0.1:8000"
session_id = str(uuid.uuid4())  # ✅ Session unique pour la conversation

print("📥 Envoi du PDF S3 pour résumé...\n")

response = requests.post(
    f"{API_URL}/summarize_from_url/",
    json={"url": pdf_url, "entreprise": entreprise}
)

if response.status_code == 200:
    data = response.json()
    job_id = data.get("job_id", "-")
    task_id = data.get("task_id")

    if data.get("mode") == "cache":
        print("✅ Résumé chargé depuis le cache :\n")
        print(data.get("summary", "(pas de résumé trouvé)"))
    else:
        print(f"🚀 Tâche Celery lancée : job_id={job_id} | task_id={task_id}")
        print("⏳ Attente du résumé global...\n")

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
                sys.exit(1)
            else:
                sys.stdout.write(f"\r🔄 Statut actuel : {status_data['status']} {spinner[i % len(spinner)]}")
                sys.stdout.flush()
                i += 1
                time.sleep(1.5)

    print("\n💬 ASK est maintenant disponible. Tape une question à poser sur le PDF :\n(tape 'exit' pour quitter)\n")

    while True:
        question = input("❓ Ta question : ").strip()
        if question.lower() in ("exit", "quit", ""):
            print("👋 Fin du test.")
            break

        ask_response = requests.post(
            f"{API_URL}/ask_from_url/",
            json={
                "job_id": job_id,
                "question": question,
                "entreprise": entreprise,
                "session_id": session_id  # ✅ Ajout ici
            }
        )

        if ask_response.status_code == 200:
            ask_data = ask_response.json()
            print(f"\n✅ Réponse :\n{ask_data.get('answer', '(pas de réponse générée)')}\n")
        else:
            print(f"\n❌ Erreur : {ask_response.status_code}")
            print(ask_response.text)

else:
    print(f"❌ Erreur HTTP {response.status_code} :")
    print(response.text)
