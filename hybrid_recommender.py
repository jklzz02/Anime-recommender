import numpy as np
import json
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

content_embeddings = np.load("data/embeddings/anime_embeddings.npy")
nlp_embeddings = np.load("data/embeddings/anime_nlp_embeddings.npy")
compatibility_embeddings = np.load("data/embeddings/anime_compatibility_embeddings.npy")

user_embeddings = np.load("data/embeddings/user_embeddings.npy")
anime_cf_embeddings = np.load("data/embeddings/anime_cf_embeddings.npy")

with open("data/json/id_to_index.json", "r") as f:
    id_to_index = json.load(f)

with open("data/json/index_to_id.json", "r") as f:
    index_to_id = json.load(f)
    index_to_id = {int(k): v for k, v in index_to_id.items()}

with open("data/json/user_mappings.json", "r") as f:
    user_mappings = json.load(f)
    user_to_idx = {int(k): v for k, v in user_mappings["user_to_idx"].items()}
    idx_to_user = {int(k): v for k, v in user_mappings["idx_to_user"].items()}
    anime_to_cf_idx = {int(k): v for k, v in user_mappings["anime_to_idx"].items()}
    cf_idx_to_anime = {int(k): v for k, v in user_mappings["idx_to_anime"].items()}

with open("data/json/rating_stats.json", "r") as f:
    rating_stats = json.load(f)
    global_mean = rating_stats["global_mean"]
    anime_stats = {int(k): v for k, v in rating_stats["anime_stats"].items()}
    user_stats = {int(k): v for k, v in rating_stats["user_stats"].items()}

model = SentenceTransformer("intfloat/e5-base")

def get_cf_recommendations_for_user(user_id: int, limit: int = 10, 
                                    exclude_rated: bool = True) -> List[Tuple[int, float]]:
    """
    Get collaborative filtering recommendations for a specific user.
    
    Args:
        user_id: The user ID
        limit: Maximum number of recommendations
        exclude_rated: Whether to exclude anime the user has already rated
    
    Returns:
        List of tuples (anime_id, predicted_rating)
    """
    user_idx = user_to_idx.get(user_id)
    
    if user_idx is None:
        return []
    
    user_vector = user_embeddings[user_idx].reshape(1, -1)
    
    similarities = cosine_similarity(user_vector, anime_cf_embeddings).flatten()
    
    user_mean = user_stats.get(user_id, {}).get('mean', global_mean)
    predicted_ratings = user_mean + similarities * 5  # Scale factor
    predicted_ratings = np.clip(predicted_ratings, 1, 10)
    
    sorted_indices = predicted_ratings.argsort()[::-1]
    
    recommendations = []
    for cf_idx in sorted_indices:
        anime_id = cf_idx_to_anime[cf_idx]
        predicted_rating = float(predicted_ratings[cf_idx])
        
        recommendations.append((anime_id, predicted_rating))
        
        if len(recommendations) >= limit:
            break
    
    return recommendations


def get_cf_similar_anime(anime_id: int, limit: int = 10) -> List[Tuple[int, float]]:
    """
    Find similar anime using collaborative filtering embeddings.
    
    Args:
        anime_id: The anime ID to find similar anime for
        limit: Maximum number of similar anime to return
    
    Returns:
        List of tuples (anime_id, similarity_score)
    """
    cf_idx = anime_to_cf_idx.get(anime_id)
    
    if cf_idx is None:
        return []
    
    anime_vector = anime_cf_embeddings[cf_idx].reshape(1, -1)
    
    similarities = cosine_similarity(anime_vector, anime_cf_embeddings).flatten()
    
    sorted_indices = similarities.argsort()[::-1][1:limit + 1]  # Exclude self
    
    return [(cf_idx_to_anime[idx], float(similarities[idx])) for idx in sorted_indices]


def predict_user_rating(user_id: int, anime_id: int) -> Optional[float]:
    """
    Predict what rating a user would give to a specific anime.
    
    Args:
        user_id: The user ID
        anime_id: The anime ID
    
    Returns:
        Predicted rating (1-10) or None if prediction not possible
    """
    user_idx = user_to_idx.get(user_id)
    cf_idx = anime_to_cf_idx.get(anime_id)
    
    if user_idx is None or cf_idx is None:
        return None
    
    user_vector = user_embeddings[user_idx].reshape(1, -1)
    anime_vector = anime_cf_embeddings[cf_idx].reshape(1, -1)
    similarity = cosine_similarity(user_vector, anime_vector)[0][0]
    
    user_mean = user_stats.get(user_id, {}).get('mean', global_mean)
    predicted_rating = user_mean + similarity * 5
    predicted_rating = np.clip(predicted_rating, 1, 10)
    
    return float(predicted_rating)

def get_hybrid_recommendations(user_id: int, 
                               limit: int = 10,
                               cf_weight: float = 0.5,
                               content_weight: float = 0.5,
                               user_anime_list: Optional[List[int]] = None) -> List[Tuple[int, float, Dict]]:
    """
    Get hybrid recommendations combining collaborative filtering and content-based filtering.
    
    Args:
        user_id: The user ID
        limit: Maximum number of recommendations
        cf_weight: Weight for collaborative filtering score (0-1)
        content_weight: Weight for content-based score (0-1)
        user_anime_list: Optional list of anime IDs the user has watched (for content-based)
    
    Returns:
        List of tuples (anime_id, hybrid_score, score_breakdown)
        where score_breakdown contains {'cf_score', 'content_score', 'hybrid_score'}
    """
    total_weight = cf_weight + content_weight
    cf_weight = cf_weight / total_weight
    content_weight = content_weight / total_weight
    
    cf_recs = get_cf_recommendations_for_user(user_id, limit=limit * 3)
    cf_scores = {anime_id: score / 10.0 for anime_id, score in cf_recs}  # Normalize to 0-1
    
    content_scores = {}
    if user_anime_list:
        valid_indices = [id_to_index[str(i)] for i in user_anime_list if str(i) in id_to_index]
        
        if valid_indices:
            user_vectors = compatibility_embeddings[[int(idx) for idx in valid_indices]]
            user_profile = np.mean(user_vectors, axis=0).reshape(1, -1)
            
            similarities = cosine_similarity(user_profile, compatibility_embeddings).flatten()
            
            for idx, sim in enumerate(similarities):
                anime_id = int(index_to_id[idx])
                if anime_id not in user_anime_list:
                    content_scores[anime_id] = float(sim)
    
    all_anime_ids = set(cf_scores.keys()) | set(content_scores.keys())
    hybrid_results = []
    
    for anime_id in all_anime_ids:
        cf_score = cf_scores.get(anime_id, 0.0)
        content_score = content_scores.get(anime_id, 0.0)
        
        hybrid_score = (cf_weight * cf_score) + (content_weight * content_score)
        
        score_breakdown = {
            'cf_score': round(cf_score, 4),
            'content_score': round(content_score, 4),
            'hybrid_score': round(hybrid_score, 4)
        }
        
        hybrid_results.append((anime_id, hybrid_score, score_breakdown))
    
    hybrid_results.sort(key=lambda x: x[1], reverse=True)
    
    return hybrid_results[:limit]


def get_hybrid_recommendations_with_text(user_id: int,
                                        text_query: str,
                                        limit: int = 10,
                                        cf_weight: float = 0.33,
                                        content_weight: float = 0.33,
                                        nlp_weight: float = 0.34) -> List[Tuple[int, float, Dict]]:
    """
    Get hybrid recommendations combining CF, content-based, and text-based search.
    
    Args:
        user_id: The user ID
        text_query: Natural language query (e.g., "action anime with magic")
        limit: Maximum number of recommendations
        cf_weight: Weight for collaborative filtering
        content_weight: Weight for content-based filtering
        nlp_weight: Weight for NLP text matching
    
    Returns:
        List of tuples (anime_id, hybrid_score, score_breakdown)
    """

    total_weight = cf_weight + content_weight + nlp_weight
    cf_weight = cf_weight / total_weight
    content_weight = content_weight / total_weight
    nlp_weight = nlp_weight / total_weight
    
    cf_recs = get_cf_recommendations_for_user(user_id, limit=limit * 3)
    cf_scores = {anime_id: score / 10.0 for anime_id, score in cf_recs}
    
    query_embedding = model.encode(["query: " + text_query])[0].reshape(1, -1)
    nlp_similarities = cosine_similarity(query_embedding, nlp_embeddings).flatten()
    nlp_scores = {int(index_to_id[i]): float(nlp_similarities[i]) 
                  for i in range(len(nlp_similarities))}
    
    user_idx = user_to_idx.get(user_id)
    content_scores = {}
    
    if user_idx is not None:
        user_vector = user_embeddings[user_idx].reshape(1, -1)
        cf_sims = cosine_similarity(user_vector, anime_cf_embeddings).flatten()
        
        top_cf_indices = cf_sims.argsort()[::-1][:20]
        watched_anime_ids = [cf_idx_to_anime[idx] for idx in top_cf_indices]
        
        valid_indices = [id_to_index[str(aid)] for aid in watched_anime_ids if str(aid) in id_to_index]
        
        if valid_indices:
            user_vectors = compatibility_embeddings[[int(idx) for idx in valid_indices]]
            user_profile = np.mean(user_vectors, axis=0).reshape(1, -1)
            
            content_similarities = cosine_similarity(user_profile, compatibility_embeddings).flatten()
            content_scores = {int(index_to_id[i]): float(content_similarities[i]) 
                            for i in range(len(content_similarities))}
    
    all_anime_ids = set(cf_scores.keys()) | set(nlp_scores.keys()) | set(content_scores.keys())
    hybrid_results = []
    
    for anime_id in all_anime_ids:
        cf_score = cf_scores.get(anime_id, 0.0)
        nlp_score = nlp_scores.get(anime_id, 0.0)
        content_score = content_scores.get(anime_id, 0.0)
        
        hybrid_score = (cf_weight * cf_score) + (nlp_weight * nlp_score) + (content_weight * content_score)
        
        score_breakdown = {
            'cf_score': round(cf_score, 4),
            'nlp_score': round(nlp_score, 4),
            'content_score': round(content_score, 4),
            'hybrid_score': round(hybrid_score, 4)
        }
        
        hybrid_results.append((anime_id, hybrid_score, score_breakdown))
    
    hybrid_results.sort(key=lambda x: x[1], reverse=True)
    
    return hybrid_results[:limit]


def get_cold_start_recommendations(text_query: str,
                                   favorite_genres: Optional[List[str]] = None,
                                   limit: int = 10) -> List[Tuple[int, float]]:
    """
    Get recommendations for new users (cold start problem) using only content-based methods.
    
    Args:
        text_query: Natural language description of preferences
        favorite_genres: Optional list of favorite genres
        limit: Maximum number of recommendations
    
    Returns:
        List of tuples (anime_id, relevance_score)
    """
    if favorite_genres:
        text_query = text_query + " " + " ".join(favorite_genres)
    
    query_embedding = model.encode(["query: " + text_query])[0].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, nlp_embeddings).flatten()
    
    sorted_indices = similarities.argsort()[::-1][:limit]
    
    return [(int(index_to_id[i]), float(similarities[i])) for i in sorted_indices]


def get_similar_users(user_id: int, limit: int = 10) -> List[Tuple[int, float]]:
    """
    Find users with similar taste using collaborative filtering embeddings.
    
    Args:
        user_id: The user ID
        limit: Maximum number of similar users to return
    
    Returns:
        List of tuples (user_id, similarity_score)
    """
    user_idx = user_to_idx.get(user_id)
    
    if user_idx is None:
        return []
    
    user_vector = user_embeddings[user_idx].reshape(1, -1)
    similarities = cosine_similarity(user_vector, user_embeddings).flatten()
    
    sorted_indices = similarities.argsort()[::-1][1:limit + 1]  # Exclude self
    
    return [(idx_to_user[idx], float(similarities[idx])) for idx in sorted_indices]


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("HYBRID RECOMMENDATION SYSTEM EXAMPLES")
    print("=" * 60)
    
    # Example 1: Pure CF recommendations
    print("\n1. Collaborative Filtering Recommendations for User 1:")
    cf_recs = get_cf_recommendations_for_user(user_id=1, limit=5)
    for anime_id, score in cf_recs:
        print(f"   Anime ID {anime_id}: Predicted Rating {score:.2f}/10")
    
    # Example 2: CF-based similar anime
    print("\n2. Similar Anime using CF (to Anime ID 1):")
    similar = get_cf_similar_anime(anime_id=1, limit=5)
    for anime_id, score in similar:
        print(f"   Anime ID {anime_id}: Similarity {score:.4f}")
    
    # Example 3: Predict rating
    print("\n3. Predict User 1's Rating for Anime ID 100:")
    prediction = predict_user_rating(user_id=1, anime_id=100)
    if prediction:
        print(f"   Predicted Rating: {prediction:.2f}/10")
    
    # Example 4: Hybrid recommendations (CF + Content)
    print("\n4. Hybrid Recommendations for User 1:")
    user_watched = [21, 48, 1, 5, 120]  # Sample watched anime
    hybrid_recs = get_hybrid_recommendations(
        user_id=1,
        limit=5,
        cf_weight=0.6,
        content_weight=0.4,
        user_anime_list=user_watched
    )
    for anime_id, score, breakdown in hybrid_recs:
        print(f"   Anime ID {anime_id}: Score {score:.4f}")
        print(f"      CF: {breakdown['cf_score']:.4f}, Content: {breakdown['content_score']:.4f}")
    
    # Example 5: Hybrid with text query
    print("\n5. Hybrid Recommendations with Text Query:")
    hybrid_text_recs = get_hybrid_recommendations_with_text(
        user_id=1,
        text_query="dark action anime with supernatural elements",
        limit=5,
        cf_weight=0.4,
        content_weight=0.3,
        nlp_weight=0.3
    )
    for anime_id, score, breakdown in hybrid_text_recs:
        print(f"   Anime ID {anime_id}: Score {score:.4f}")
        print(f"      CF: {breakdown['cf_score']:.4f}, NLP: {breakdown['nlp_score']:.4f}, Content: {breakdown['content_score']:.4f}")
    
    # Example 6: Cold start recommendations
    print("\n6. Cold Start Recommendations (new user):")
    cold_start = get_cold_start_recommendations(
        text_query="action adventure with strong characters",
        favorite_genres=["Action", "Adventure", "Fantasy"],
        limit=5
    )
    for anime_id, score in cold_start:
        print(f"   Anime ID {anime_id}: Relevance {score:.4f}")
    
    # Example 7: Similar users
    print("\n7. Users Similar to User 1:")
    similar_users = get_similar_users(user_id=1, limit=5)
    for user_id, score in similar_users:
        print(f"   User ID {user_id}: Similarity {score:.4f}")
    
    print("\n" + "=" * 60)