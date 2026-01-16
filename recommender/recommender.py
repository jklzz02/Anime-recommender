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

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


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
    """
    Natural language search using NLP embeddings
    Example: 'dark psychological thriller' or 'cute slice of life'
    """
    query_with_prefix = QUERY_PREFIX + query
    query_embedding = model.encode([query_with_prefix], normalize_embeddings=True)[0].reshape(1, -1)
    
    similarities = cosine_similarity(query_embedding, nlp_embeddings).flatten()
    similar_indices = similarities.argsort()[::-1][:limit]
    
    return [int(index_to_id[i]) for i in similar_indices]


def get_recommendations_from_text_with_scores(query: str, limit: int = 10) -> List[Tuple[int, float]]:
    """
    Natural language search with similarity scores
    Returns: List of (anime_id, score) tuples
    """
    query_with_prefix = QUERY_PREFIX + query
    query_embedding = model.encode([query_with_prefix], normalize_embeddings=True)[0].reshape(1, -1)
    
    similarities = cosine_similarity(query_embedding, nlp_embeddings).flatten()
    similar_indices = similarities.argsort()[::-1][:limit]
    
    return [(int(index_to_id[i]), float(similarities[i])) for i in similar_indices]


def get_recommendations_semantic_search(query: str, limit: int = 10) -> List[Tuple[int, float]]:
    """
    Structured semantic search using content embeddings
    Better for specific attribute queries like 'action anime with high rating'
    """
    query_with_prefix = QUERY_PREFIX + query
    query_embedding = model.encode([query_with_prefix], normalize_embeddings=True)[0].reshape(1, -1)
    
    similarities = cosine_similarity(query_embedding, content_embeddings).flatten()
    similar_indices = similarities.argsort()[::-1][:limit]
    
    return [(int(index_to_id[i]), float(similarities[i])) for i in similar_indices]


def hybrid_text_search(query: str, 
                       limit: int = 10,
                       nlp_weight: float = 0.6,
                       content_weight: float = 0.4) -> List[Tuple[int, float]]:
    """
    Combines NLP and content-based search for better results
    
    Args:
        query: User's search query
        limit: Number of results to return
        nlp_weight: Weight for conversational search (0-1)
        content_weight: Weight for structured search (0-1)
    
    Returns:
        List of (anime_id, combined_score) sorted by relevance
    """
    total = nlp_weight + content_weight
    nlp_weight /= total
    content_weight /= total
    query_with_prefix = QUERY_PREFIX + query
    query_embedding = model.encode([query_with_prefix], normalize_embeddings=True)[0].reshape(1, -1)
    
    nlp_similarities = cosine_similarity(query_embedding, nlp_embeddings).flatten()
    content_similarities = cosine_similarity(query_embedding, content_embeddings).flatten()
    
    combined_scores = nlp_weight * nlp_similarities + content_weight * content_similarities
    
    similar_indices = combined_scores.argsort()[::-1][:limit]
    
    return [(int(index_to_id[i]), float(combined_scores[i])) for i in similar_indices]