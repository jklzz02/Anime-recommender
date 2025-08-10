import numpy as np
import json

from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from data.build_embeddings import embeddings_path

embeddings = np.load(embeddings_path)

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

def get_recommendations_by_list(anime_ids: List[int], limit: int = 10) -> List[int]:
    valid_indices = [id_to_index[str(i)] for i in anime_ids if str(i) in id_to_index]

    if not valid_indices:
        return []

    vectors = [embeddings[int(idx)] for idx in valid_indices]
    query_vector = np.mean(vectors, axis=0).reshape(1, -1)

    similarities = cosine_similarity(query_vector, embeddings).flatten()
    similar_indices = similarities.argsort()[::-1]

    recommended_ids = []
    for i in similar_indices:
        candidate_id = int(index_to_id[i])
        if candidate_id not in anime_ids:
            recommended_ids.append(candidate_id)
        if len(recommended_ids) >= limit:
            break

    return recommended_ids

