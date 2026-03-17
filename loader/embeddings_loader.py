import os
import logging
import numpy as np
import json
from colorama import Fore
from huggingface_hub import hf_hub_download
from functools import lru_cache

_logger = logging.getLogger(__name__)

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
            _logger.info(f"Downloading {file}...")
            _download(file)
        else:
            _logger.info(f"{Fore.GREEN}{local_path}{Fore.RESET} already loaded")

@lru_cache(maxsize=None)
def load_anime_embeddings():
    return _load_embedding("anime_embeddings")

@lru_cache(maxsize=None)
def load_anime_nlp_embeddings():
    return _load_embedding("anime_nlp_embeddings")

@lru_cache(maxsize=None)
def load_anime_compatibility_embeddings():
    return _load_embedding("anime_compatibility_embeddings")

@lru_cache(maxsize=None)
def load_anime_cf_embeddings():
    return _load_embedding("anime_cf_embeddings")

@lru_cache(maxsize=None)
def load_user_embeddings():
    return _load_embedding("user_embeddings")

@lru_cache(maxsize=None)
def load_id_to_index():
    return _load_index("id_to_index")

@lru_cache(maxsize=None)
def load_index_to_id():
    return _load_index("index_to_id")

@lru_cache(maxsize=None)
def load_user_mappings():
    return _load_index("user_mappings")

@lru_cache(maxsize=None)
def load_rating_stats():
    return _load_index("rating_stats")

def _load_embedding(embedding: str):

    if not embedding or not embedding.strip():
        raise ValueError(f"embedding argument cannot be null or empty")
    
    resource_name = f"{embedding}.npy" if not embedding.endswith(".npy") else embedding

    if not f"embeddings/{resource_name}" in FILES:
        raise ValueError(f"Cannot load '{resource_name}': not a valid embeddings file.")
    
    return _load_resource(resource_name)

def _load_index(index: str):

    if not index or not index.strip():
        raise ValueError(f"'index' argument cannot be null or empty")

    resource_name = f"{index}.json" if not index.endswith(".json") else index
    
    if not f"json/{resource_name}" in FILES:
        raise ValueError(f"Cannot load '{resource_name}': not a valid json index file.")
    
    file_name = f"json/{resource_name}"
    file_path = os.path.abspath(os.path.join(DATA_DIR, file_name))
    
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    
    except FileNotFoundError as ex:
        _logger.warning(f"Executing download attempt due to: {ex}")
        _download(file_name)
        with open(file_path, 'r') as f:
            return json.load(f)

def _load_resource(file_name: str):

    embeddings_name = f"embeddings/{file_name}"
    embeddings_path = os.path.abspath(os.path.join(DATA_DIR, embeddings_name))

    try:
        return np.load(embeddings_path)
    except FileNotFoundError as ex:
        _logger.warning(f"Executing download attempt due to: {ex}")
        _download(embeddings_name)
        return np.load(embeddings_path)
    

def _download(filename: str):
    hf_hub_download(
        repo_id=REPO_ID, filename=filename, repo_type="model", local_dir=DATA_DIR
    )

    _logger.info(f"Successfully downloaded '{filename}' at {DATA_DIR}")


if __name__ == "__main__":
    ensure_data()