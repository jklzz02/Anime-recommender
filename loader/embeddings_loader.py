import os

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
]


def ensure_data():
    for file in FILES:
        local_path = os.path.join("data", file)
        if not os.path.exists(local_path):
            print(f"Downloading {file}...")
            hf_hub_download(
                repo_id=REPO_ID, filename=file, repo_type="model", local_dir="data"
            )


if __name__ == "__main__":
    ensure_data()
