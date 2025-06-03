import os
import json

def save_json(folder_path, bloc_index, data):
    os.makedirs(folder_path, exist_ok=True)
    filename = f"bloc_{bloc_index+1:02d}.json"
    with open(os.path.join(folder_path, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_all_json(folder_path):
    summaries = []
    for f in sorted(os.listdir(folder_path)):
        if f.endswith(".json"):
            with open(os.path.join(folder_path, f), "r", encoding="utf-8") as j:
                data = json.load(j)
                summaries.append((data["bloc"], data["summary"]))
    summaries.sort(key=lambda x: x[0])
    return summaries
