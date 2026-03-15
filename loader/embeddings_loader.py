import os
from colorama import Fore
from huggingface_hub import hf_hub_download

REPO_ID = "jklzz02/anime-embeddings"
FILES = [
    "embeddings/anime_embeddings.npy",
    "embeddings/anime_nlp_embeddings.npy",
    "embeddings/anime_compatibility_embeddings.npy",
    "embeddings/anime_cf_embeddings.npy",
    "embeddings/user_embeddings.npy",
    "json/id_to_index.json",
    "json/index_to_id.json",
    "json/user_mappings.json",
    "json/rating_stats.json",
    "anime-dataset.csv"
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../data"))

def ensure_data():
    for file in FILES:
        local_path = os.path.join(DATA_DIR, file)
        if not os.path.exists(local_path):
            print(f"Downloading {file}...")
            hf_hub_download(
                repo_id=REPO_ID, filename=file, repo_type="model", local_dir=DATA_DIR
            )
        else:
            print(f"{Fore.GREEN}{local_path}{Fore.RESET} already loaded")

if __name__ == "__main__":
    ensure_data()