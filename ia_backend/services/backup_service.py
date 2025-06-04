import os
import json
import shutil

SAVE_DIR = os.path.join("cache_json", "save_summaryglobal")
BLOCKS_DIR = os.path.join("cache_json", "save_summaryblocks")

def get_summary_path(entreprise, folder_name):
    """
    Retourne le chemin vers le fichier résumé global :
    cache_json/save_summaryglobal/<entreprise>/summary_<pdf>.json
    """
    entreprise_dir = os.path.join(SAVE_DIR, entreprise)
    os.makedirs(entreprise_dir, exist_ok=True)
    filename = f"summary_{folder_name}.json"
    return os.path.join(entreprise_dir, filename)

def save_global_summary(entreprise, folder_name, summary):
    """
    Sauvegarde le résumé global dans le bon dossier.
    """
    path = get_summary_path(entreprise, folder_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary}, f, indent=2, ensure_ascii=False)
    print(f"💾 Résumé global sauvegardé : {path}")

def load_global_summary_if_exists(entreprise, folder_name):
    """
    Vérifie s'il existe déjà un résumé global.
    """
    path = get_summary_path(entreprise, folder_name)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("summary")
        except json.JSONDecodeError:
            return None
    return None

def save_block_jsons(entreprise, folder_name, cache_dir):
    """
    Sauvegarde tous les fichiers bloc_XX.json dans un dossier permanent :
    cache_json/save_summaryblocks/<entreprise>/<folder_name>/
    """
    dest_dir = os.path.join(BLOCKS_DIR, entreprise, folder_name)
    os.makedirs(dest_dir, exist_ok=True)

    for file in sorted(os.listdir(cache_dir)):
        if file.startswith("bloc_") and file.endswith(".json"):
            shutil.copy(os.path.join(cache_dir, file), os.path.join(dest_dir, file))

    print(f"📦 Résumés partiels sauvegardés dans : {dest_dir}")
