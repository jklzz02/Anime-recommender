import numpy as np
import json
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from loader import get_transformer

content_embeddings = np.load("data/embeddings/anime_embeddings.npy")
nlp_embeddings = np.load("data/embeddings/anime_nlp_embeddings.npy")
compatibility_embeddings = np.load("data/embeddings/anime_compatibility_embeddings.npy")

with open("data/json/id_to_index.json", "r") as f:
    id_to_index = json.load(f)

with open("data/json/index_to_id.json", "r") as f:
    index_to_id = json.load(f)
    
index_to_id = {int(k): v for k, v in index_to_id.items()}

def normalize_matrix(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms

nlp_embeddings = normalize_matrix(nlp_embeddings)
content_embeddings = normalize_matrix(content_embeddings)
compatibility_embeddings = normalize_matrix(compatibility_embeddings)

model = get_transformer()

def get_recommendations(anime_id: int, limit: int = 10) -> List[Tuple[int, float]]:
    """Get similar anime based on content compatibility"""
    idx = id_to_index.get(str(anime_id))
    if idx is None:
        return []
    
    query_vector = compatibility_embeddings[int(idx)].reshape(1, -1)
    similarities = cosine_similarity(query_vector, compatibility_embeddings).flatten()
    
    similar_indices = similarities.argsort()[::-1][1:limit + 1]
    return [(int(index_to_id[i]), float(similarities[i])) for i in similar_indices]


def get_recommendations_by_list(anime_ids: List[int], limit: int = 10) -> List[Tuple[int, float]]:
    """Get recommendations based on multiple anime (averaged profile)"""
    valid_indices = [id_to_index[str(i)] for i in anime_ids if str(i) in id_to_index]
    if not valid_indices:
        return []
    
    vectors = [compatibility_embeddings[int(idx)] for idx in valid_indices]
    query_vector = np.mean(vectors, axis=0).reshape(1, -1)
    
    similarities = cosine_similarity(query_vector, compatibility_embeddings).flatten()
    similar_indices = similarities.argsort()[::-1]
    
    recommended = []
    for i in similar_indices:
        candidate_id = int(index_to_id[i])
        if candidate_id not in anime_ids:
            recommended.append((candidate_id, float(similarities[i])))
            if len(recommended) >= limit:
                break
    
    return recommended


def get_recommendations_from_text(query: str, limit: int = 10) -> List[int]:
    """NLP search using text embeddings"""
    query_embedding = model.encode([query], normalize_embeddings=True)[0].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, nlp_embeddings).flatten()
    similar_indices = similarities.argsort()[::-1][:limit]
    return [int(index_to_id[i]) for i in similar_indices]


def get_recommendations_from_text_with_scores(query: str, limit: int = 10) -> List[Tuple[int, float]]:
    """NLP search with similarity scores"""
    query_embedding = model.encode([query], normalize_embeddings=True)[0].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, nlp_embeddings).flatten()
    similar_indices = similarities.argsort()[::-1][:limit]
    return [(int(index_to_id[i]), float(similarities[i])) for i in similar_indices]


def get_recommendations_semantic_search(query: str, limit: int = 10) -> List[Tuple[int, float]]:
    """Content-based search (structured attributes)"""
    query_embedding = model.encode([query], normalize_embeddings=True)[0].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, content_embeddings).flatten()
    similar_indices = similarities.argsort()[::-1][:limit]
    return [(int(index_to_id[i]), float(similarities[i])) for i in similar_indices]
