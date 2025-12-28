import numpy as np
import json
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

content_embeddings = np.load("data/embeddings/anime_embeddings.npy")
nlp_embeddings = np.load("data/embeddings/anime_nlp_embeddings.npy")
compatibility_embeddings = np.load("data/embeddings/anime_compatibility_embeddings.npy")

with open("data/json/id_to_index.json", "r") as f:
    id_to_index = json.load(f)

with open("data/json/index_to_id.json", "r") as f:
    index_to_id = json.load(f)
    index_to_id = {int(k): v for k, v in index_to_id.items()}

model = SentenceTransformer("intfloat/e5-base")

def get_recommendations(anime_id: int, limit: int = 10) -> List[int]:
    """Get similar anime based on content embeddings"""
    idx = id_to_index.get(str(anime_id))

    if idx is None:
        return []

    query_vector = content_embeddings[int(idx)].reshape(1, -1)
    similarities = cosine_similarity(query_vector, content_embeddings).flatten()

    similar_indices = similarities.argsort()[::-1][1:limit + 1]
    return [int(index_to_id[i]) for i in similar_indices]

def get_recommendations_by_list(anime_ids: List[int], limit: int = 10) -> List[Tuple[int, float]]:
    """Get recommendations based on a list of anime"""
    valid_indices = [id_to_index[str(i)] for i in anime_ids if str(i) in id_to_index]

    if not valid_indices:
        return []

    vectors = [content_embeddings[int(idx)] for idx in valid_indices]
    query_vector = np.mean(vectors, axis=0).reshape(1, -1)

    similarities = cosine_similarity(query_vector, content_embeddings).flatten()
    similar_indices = similarities.argsort()[::-1]

    recommended_ids = []
    for i in similar_indices:
        candidate_id = int(index_to_id[i])
        if candidate_id not in anime_ids:
            recommended_ids.append(candidate_id)
        if len(recommended_ids) >= limit:
            break

    similarities = cosine_similarity(query_vector, content_embeddings).flatten()
    similar_indices = similarities.argsort()[::-1][1:limit + 1]
    return [(int(index_to_id[i]), float(similarities[i])) for i in similar_indices]

def get_recommendations_from_text(query: str, limit: int = 10) -> List[int]:
    """
    Get anime recommendations based on natural language text input.
    
    Examples:
        - "action anime with magic and adventure"
        - "romantic comedy set in high school"
        - "dark psychological thriller"
        - "anime about cooking and friendship"
    
    Args:
        query: Natural language description of desired anime
        limit: Maximum number of recommendations to return
    
    Returns:
        List of anime IDs ranked by relevance
    """
    query_embedding = model.encode(["query: " + query])[0].reshape(1, -1)
    
    similarities = cosine_similarity(query_embedding, nlp_embeddings).flatten()
    
    similar_indices = similarities.argsort()[::-1][:limit]
    
    return [int(index_to_id[i]) for i in similar_indices]


def get_recommendations_from_text_with_scores(query: str, limit: int = 10) -> List[Tuple[int, float]]:
    """
    Get anime recommendations with similarity scores.
    
    Returns:
        List of tuples (anime_id, similarity_score) where score is between 0 and 1
    """
    query_embedding = model.encode(["query: " + query])[0].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, nlp_embeddings).flatten()
    
    similar_indices = similarities.argsort()[::-1][:limit]
    
    return [(int(index_to_id[i]), float(similarities[i])) for i in similar_indices]
