import numpy as np
import json

from typing import List
from sklearn.metrics.pairwise import cosine_similarity

embeddings = np.load("data/embeddings/anime_embeddings.npy")

with open("data/json/id_to_index.json", "r") as f:
    id_to_index = json.load(f)

with open("data/json/index_to_id.json", "r") as f:
    index_to_id = json.load(f)
    index_to_id = {int(k): v for k, v in index_to_id.items()}

def get_recommendations(anime_id: int, limit: int = 10) -> List[int]:
    idx = id_to_index.get(str(anime_id))

    if idx is None:
        return []

    query_vector = embeddings[int(idx)].reshape(1, -1)
    similarities = cosine_similarity(query_vector, embeddings).flatten()

    similar_indices = similarities.argsort()[::-1][1:limit + 1]
    return [int(index_to_id[i]) for i in similar_indices]