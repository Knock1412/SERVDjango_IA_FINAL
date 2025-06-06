import requests


pdf_url = "https://edutice.hal.science/edutice-00001245v1/document"

entreprise = "Entreprise_S3_Test"

print("📥 Envoi du PDF S3 pour résumé...\n")

response = requests.post(
    "http://127.0.0.1:8000/summarize_from_url/",
    json={"url": pdf_url, "entreprise": entreprise}
)

if response.status_code == 200:
    data = response.json()
    job_id = data.get("job_id", "-")
    summary = data.get("summary", "(pas de résumé)")

    print(f"✅ Résumé généré pour job {job_id} :\n")
    print(summary)

else:
    print(f"❌ Erreur HTTP {response.status_code} :")
    print(response.text)
