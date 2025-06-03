import requests
import threading

# 🗂️ Liste des 12 PDF à tester
pdfs = [
    ("https://edutice.hal.science/edutice-00000852v1/document", "Entreprise_1"),
    ("https://edutice.hal.science/edutice-00000413v1/document", "Entreprise_2"),
    ("https://edutice.hal.science/edutice-00000862v1/document", "Entreprise_3"),
    ("https://edutice.hal.science/edutice-00001410v1/document", "Entreprise_4"),
    ("https://edutice.hal.science/edutice-00001245v1/document", "Entreprise_5"),
    
]

def envoyer_et_attendre(pdf_url, entreprise, label):
    print(f"\n📤 [{label}] {entreprise} envoie un document...")
    try:
        res = requests.post("http://127.0.0.1:8000/summarize_from_url/", json={
            "url": pdf_url,
            "entreprise": entreprise
        })

        if res.status_code == 200:
            data = res.json()
            print(f"\n✅ [{label}] Résumé reçu pour {entreprise} (mode : {data.get('mode')})")
            print(f"📄 Job ID : {data.get('job_id')}")
            print("-" * 60)
            print(data.get("summary", "(pas de résumé)"))
            print("-" * 60)
        else:
            print(f"❌ [{label}] Erreur HTTP {res.status_code}")
            print(res.text)
    except Exception as e:
        print(f"❌ [{label}] Exception : {e}")

# 🔄 Lancer chaque job dans un thread
threads = []
for i, (url, entreprise) in enumerate(pdfs):
    label = f"PDF {i + 1}"
    t = threading.Thread(target=envoyer_et_attendre, args=(url, entreprise, label))
    t.start()
    threads.append(t)

# ⏳ Attendre la fin de tous les jobs
for t in threads:
    t.join()

print("\n🎯 Tous les tests de la file d’attente sont terminés.")
