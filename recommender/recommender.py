import numpy as np
import json
from typing import List, Dict, Tuple
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

def get_recommendations_by_list(anime_ids: List[int], limit: int = 10) -> List[int]:
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

    return recommended_ids

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

def calculate_compatibility_score(target_anime_id: int, user_anime_ids: List[int]) -> float:
    """
    Calculate compatibility score (1-100) between a target anime and a user's anime list.
    
    This measures how well the target anime matches the user's taste profile
    based on their watched/liked anime.
    
    Args:
        target_anime_id: The anime to score
        user_anime_ids: List of anime IDs the user has watched/liked
    
    Returns:
        Compatibility score from 1 to 100
    """
    target_idx = id_to_index.get(str(target_anime_id))
    
    if target_idx is None or not user_anime_ids:
        return 0.0
    
    valid_indices = [id_to_index[str(i)] for i in user_anime_ids if str(i) in id_to_index]
    
    if not valid_indices:
        return 0.0
    
    target_vector = compatibility_embeddings[int(target_idx)].reshape(1, -1)
    
    user_vectors = compatibility_embeddings[[int(idx) for idx in valid_indices]]
    
    similarities = cosine_similarity(target_vector, user_vectors).flatten()
    avg_similarity = np.mean(similarities)
    
    score = max(1, min(100, (avg_similarity - 0.2) * (100 / 0.7) + 1))
    
    return float(round(score, 2))


def calculate_compatibility_scores_batch(target_anime_ids: List[int], 
                                         user_anime_ids: List[int]) -> Dict[int, float]:
    """
    Calculate compatibility scores for multiple target anime at once.
    
    Args:
        target_anime_ids: List of anime IDs to score
        user_anime_ids: List of anime IDs the user has watched/liked
    
    Returns:
        Dictionary mapping anime_id -> compatibility_score (1-100)
    """
    if not user_anime_ids:
        return {anime_id: 0.0 for anime_id in target_anime_ids}
    
    valid_user_indices = [id_to_index[str(i)] for i in user_anime_ids if str(i) in id_to_index]
    
    if not valid_user_indices:
        return {anime_id: 0.0 for anime_id in target_anime_ids}
    
    user_vectors = compatibility_embeddings[[int(idx) for idx in valid_user_indices]]
    user_profile = np.mean(user_vectors, axis=0).reshape(1, -1)
    
    results = {}
    for anime_id in target_anime_ids:
        target_idx = id_to_index.get(str(anime_id))
        
        if target_idx is None:
            results[anime_id] = 0.0
            continue
        
        target_vector = compatibility_embeddings[int(target_idx)].reshape(1, -1)
        similarity = cosine_similarity(target_vector, user_profile)[0][0]
        
        score = max(1, min(100, (similarity - 0.2) * (100 / 0.7) + 1))
        results[anime_id] = float(round(score, 2))
    
    return results


def get_high_compatibility_recommendations(user_anime_ids: List[int], 
                                           limit: int = 10,
                                           min_score: float = 50.0) -> List[Tuple[int, float]]:
    """
    Get recommendations for anime with high compatibility scores.
    
    Args:
        user_anime_ids: List of anime IDs the user has watched/liked
        limit: Maximum number of recommendations
        min_score: Minimum compatibility score to include (1-100)
    
    Returns:
        List of tuples (anime_id, compatibility_score) sorted by score descending
    """
    if not user_anime_ids:
        return []
    
    valid_indices = [id_to_index[str(i)] for i in user_anime_ids if str(i) in id_to_index]
    
    if not valid_indices:
        return []
    
    user_vectors = compatibility_embeddings[[int(idx) for idx in valid_indices]]
    user_profile = np.mean(user_vectors, axis=0).reshape(1, -1)
    
    similarities = cosine_similarity(user_profile, compatibility_embeddings).flatten()
    
    scores = np.maximum(1, np.minimum(100, (similarities - 0.2) * (100 / 0.7) + 1))
    
    sorted_indices = scores.argsort()[::-1]
    
    recommendations = []
    for idx in sorted_indices:
        anime_id = int(index_to_id[idx])
        score = float(scores[idx])
        
        if anime_id in user_anime_ids:
            continue
        
        if score < min_score:
            break
        
        recommendations.append((anime_id, float(round(score, 2))))
        
        if len(recommendations) >= limit:
            break
    
    return recommendations


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    # Example 1: Original content-based recommendation
    print("Example 1: Content-based recommendations")
    recs = get_recommendations(anime_id=1, limit=5)
    print(f"Similar anime to ID 1: {recs}\n")
    
    # Example 2: NLP text-based recommendation
    print("Example 2: Text-based recommendations")
    text_recs = get_recommendations_from_text("dark psychological anime with complex characters", limit=5)
    print(f"Recommendations for 'dark psychological anime': {text_recs}\n")
    
    # Example 3: Compatibility scoring
    print("Example 3: Compatibility scoring")
    user_list = [1, 5, 10, 20]  # User's watched anime
    target = 50  # Anime to score
    score = calculate_compatibility_score(target, user_list)
    print(f"Compatibility score for anime {target}: {score}/100\n")
    
    # Example 4: High compatibility recommendations
    print("Example 4: High compatibility recommendations")
    compat_recs = get_high_compatibility_recommendations(user_list, limit=5, min_score=60)
    print(f"High compatibility recommendations: {compat_recs}")