import numpy as np
import json
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

content_embeddings = np.load("data/embeddings/anime_embeddings.npy")
nlp_embeddings = np.load("data/embeddings/anime_nlp_embeddings.npy")
compatibility_embeddings = np.load("data/embeddings/anime_compatibility_embeddings.npy")

anime_cf_embeddings = np.load("data/embeddings/anime_cf_embeddings.npy")
user_embeddings = np.load("data/embeddings/user_embeddings.npy")

with open("data/json/id_to_index.json", "r") as f:
    id_to_index = json.load(f)

with open("data/json/index_to_id.json", "r") as f:
    index_to_id = json.load(f)
    index_to_id = {int(k): v for k, v in index_to_id.items()}

with open("data/json/user_mappings.json", "r") as f:
    user_mappings = json.load(f)
    anime_to_cf_idx = {int(k): v for k, v in user_mappings["anime_to_idx"].items()}
    cf_idx_to_anime = {int(k): v for k, v in user_mappings["idx_to_anime"].items()}

with open("data/json/rating_stats.json", "r") as f:
    rating_stats = json.load(f)
    global_mean = rating_stats["global_mean"]

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def get_cf_recommendations_from_favorites(user_anime_ids: List[int], limit: int = 10) -> List[Tuple[int, float]]:
    """CF recommendations based on a list of favorite anime IDs."""
    valid_indices = [anime_to_cf_idx[aid] for aid in user_anime_ids if aid in anime_to_cf_idx]
    if not valid_indices:
        return []

    pseudo_user_vector = np.mean(anime_cf_embeddings[valid_indices], axis=0).reshape(1, -1)
    similarities = cosine_similarity(pseudo_user_vector, anime_cf_embeddings).flatten()

    sorted_indices = similarities.argsort()[::-1]
    recommendations = []
    for cf_idx in sorted_indices:
        anime_id = cf_idx_to_anime[cf_idx]
        if anime_id not in user_anime_ids:
            recommendations.append((anime_id, float(similarities[cf_idx])))
            if len(recommendations) >= limit:
                break

    return recommendations

def get_cf_similar_anime(anime_id: int, limit: int = 10) -> List[Tuple[int, float]]:
    """Find similar anime using collaborative filtering embeddings."""
    cf_idx = anime_to_cf_idx.get(anime_id)
    if cf_idx is None:
        return []

    anime_vector = anime_cf_embeddings[cf_idx].reshape(1, -1)
    similarities = cosine_similarity(anime_vector, anime_cf_embeddings).flatten()

    sorted_indices = similarities.argsort()[::-1][1:limit + 1]
    return [(cf_idx_to_anime[idx], float(similarities[idx])) for idx in sorted_indices]

def get_hybrid_recommendations_from_favorites(user_anime_ids: List[int],
                                              limit: int = 10,
                                              cf_weight: float = 0.5,
                                              content_weight: float = 0.5) -> List[Tuple[int, float, Dict]]:
    """Hybrid recommendations (CF + Content) using user's favorite anime IDs."""
    total_weight = cf_weight + content_weight
    cf_weight /= total_weight
    content_weight /= total_weight

    cf_recs = get_cf_recommendations_from_favorites(user_anime_ids, limit=limit * 3)
    cf_scores = {aid: score for aid, score in cf_recs}

    valid_indices = [id_to_index[str(aid)] for aid in user_anime_ids if str(aid) in id_to_index]
    content_scores = {}
    if valid_indices:
        user_profile = np.mean(compatibility_embeddings[valid_indices], axis=0).reshape(1, -1)
        similarities = cosine_similarity(user_profile, compatibility_embeddings).flatten()
        for idx, sim in enumerate(similarities):
            anime_id = int(index_to_id[idx])
            if anime_id not in user_anime_ids:
                content_scores[anime_id] = float(sim)

    all_ids = set(cf_scores.keys()) | set(content_scores.keys())
    hybrid_results = []
    for anime_id in all_ids:
        cf_score = cf_scores.get(anime_id, 0.0)
        content_score = content_scores.get(anime_id, 0.0)
        hybrid_score = cf_weight * cf_score + content_weight * content_score
        score_breakdown = {
            'cf_score': round(cf_score, 4),
            'content_score': round(content_score, 4),
            'hybrid_score': round(hybrid_score, 4)
        }
        hybrid_results.append((anime_id, hybrid_score, score_breakdown))

    hybrid_results.sort(key=lambda x: x[1], reverse=True)
    return hybrid_results[:limit]


def get_hybrid_recommendations_with_text_from_favorites(user_anime_ids: List[int],
                                                        text_query: str,
                                                        limit: int = 10,
                                                        cf_weight: float = 0.33,
                                                        content_weight: float = 0.33,
                                                        nlp_weight: float = 0.34) -> List[Tuple[int, float, Dict]]:
    """Hybrid recommendations (CF + Content + NLP) using favorites + text query."""
    total_weight = cf_weight + content_weight + nlp_weight
    cf_weight /= total_weight
    content_weight /= total_weight
    nlp_weight /= total_weight

    cf_recs = get_cf_recommendations_from_favorites(user_anime_ids, limit=limit * 3)
    cf_scores = {aid: score for aid, score in cf_recs}

    valid_indices = [id_to_index[str(aid)] for aid in user_anime_ids if str(aid) in id_to_index]
    content_scores = {}
    if valid_indices:
        user_profile = np.mean(compatibility_embeddings[valid_indices], axis=0).reshape(1, -1)
        similarities = cosine_similarity(user_profile, compatibility_embeddings).flatten()
        for idx, sim in enumerate(similarities):
            anime_id = int(index_to_id[idx])
            if anime_id not in user_anime_ids:
                content_scores[anime_id] = float(sim)

    query_embedding = model.encode(["query: " + text_query])[0].reshape(1, -1)
    nlp_similarities = cosine_similarity(query_embedding, nlp_embeddings).flatten()
    nlp_scores = {int(index_to_id[i]): float(nlp_similarities[i]) for i in range(len(nlp_similarities))}

    all_ids = set(cf_scores.keys()) | set(content_scores.keys()) | set(nlp_scores.keys())
    hybrid_results = []
    for anime_id in all_ids:
        cf_score = cf_scores.get(anime_id, 0.0)
        content_score = content_scores.get(anime_id, 0.0)
        nlp_score = nlp_scores.get(anime_id, 0.0)
        hybrid_score = cf_weight * cf_score + content_weight * content_score + nlp_weight * nlp_score
        score_breakdown = {
            'cf_score': round(cf_score, 4),
            'content_score': round(content_score, 4),
            'nlp_score': round(nlp_score, 4),
            'hybrid_score': round(hybrid_score, 4)
        }
        hybrid_results.append((anime_id, hybrid_score, score_breakdown))

    hybrid_results.sort(key=lambda x: x[1], reverse=True)
    return hybrid_results[:limit]

def predict_rating_from_favorites(user_anime_ids: List[int], anime_id: int) -> Optional[float]:
    """Predict rating using user's favorite anime IDs as a pseudo-profile."""
    valid_indices = [anime_to_cf_idx[aid] for aid in user_anime_ids if aid in anime_to_cf_idx]
    if not valid_indices or anime_id not in anime_to_cf_idx:
        return None

    pseudo_user_vector = np.mean(anime_cf_embeddings[valid_indices], axis=0).reshape(1, -1)
    anime_vector = anime_cf_embeddings[anime_to_cf_idx[anime_id]].reshape(1, -1)
    similarity = cosine_similarity(pseudo_user_vector, anime_vector)[0][0]

    predicted_rating = 1 + similarity * 9
    predicted_rating = np.clip(predicted_rating, 1, 10)
    return float(predicted_rating)

def get_similar_users_from_favorites(user_anime_ids: List[int], limit: int = 10) -> List[Tuple[int, float]]:
    """
    Find "users" with similar taste to the pseudo-profile from favorites.
    Uses existing CF user embeddings as reference knowledge.
    """
    valid_indices = [anime_to_cf_idx[aid] for aid in user_anime_ids if aid in anime_to_cf_idx]
    if not valid_indices:
        return []

    pseudo_user_vector = np.mean(anime_cf_embeddings[valid_indices], axis=0).reshape(1, -1)
    similarities = cosine_similarity(pseudo_user_vector, user_embeddings).flatten()

    sorted_indices = similarities.argsort()[::-1][:limit]
    return [(idx, float(similarities[idx])) for idx in sorted_indices]

def calculate_compatibility_score(
    target_anime_id: int, 
    user_anime_ids: List[int],
    user_ratings: Optional[Dict[int, float]] = None,
    score_floor: float = 50.0,
    cf_boost: float = 1.2,
) -> float:
    """
    Enhanced compatibility using hybrid CF + content approach with better calibration.
    
    Args:
        target_anime_id: The anime to score
        user_anime_ids: List of anime IDs the user has watched/liked
        user_ratings: Optional dict of {anime_id: rating} for weighting (1-10 scale)
        score_floor: Minimum compatibility score (default 50)
        cf_boost: Multiplier for CF importance when available (default 1.2)
    
    Returns:
        Compatibility score from 1 to 100
    """
    target_content_idx = id_to_index.get(str(target_anime_id))
    target_cf_idx = anime_to_cf_idx.get(target_anime_id)
    
    if target_content_idx is None or not user_anime_ids:
        return 0.0
    
    cf_user_ids = [aid for aid in user_anime_ids if aid in anime_to_cf_idx]
    content_user_indices = [
        id_to_index[str(aid)] for aid in user_anime_ids 
        if str(aid) in id_to_index
    ]
    
    if not cf_user_ids and not content_user_indices:
        return 0.0
    
    cf_weights = None
    content_weights = None
    
    if user_ratings:
        if cf_user_ids:
            cf_weights = np.array([
                max(0, user_ratings.get(aid, 7.0) - 5.0) / 5.0
                for aid in cf_user_ids
            ])
            cf_weights = np.power(cf_weights + 0.5, 2)
            if cf_weights.sum() > 0:
                cf_weights = cf_weights / cf_weights.sum()
            else:
                cf_weights = np.ones(len(cf_user_ids)) / len(cf_user_ids)
        
        if content_user_indices:
            content_weights = np.array([
                max(0, user_ratings.get(int(index_to_id[idx]), 7.0) - 5.0) / 5.0
                for idx in content_user_indices
            ])
            content_weights = np.power(content_weights + 0.5, 2)
            if content_weights.sum() > 0:
                content_weights = content_weights / content_weights.sum()
            else:
                content_weights = np.ones(len(content_user_indices)) / len(content_user_indices)
    
    cf_similarity = None
    content_similarity = None
    
    if target_cf_idx is not None and cf_user_ids:
        target_cf_vec = anime_cf_embeddings[target_cf_idx].reshape(1, -1)
        user_cf_vecs = anime_cf_embeddings[[anime_to_cf_idx[aid] for aid in cf_user_ids]]
        
        cf_sims = cosine_similarity(target_cf_vec, user_cf_vecs).flatten()
        
        if cf_weights is not None:
            cf_similarity = np.average(cf_sims, weights=cf_weights)
        else:
            cf_similarity = np.max(cf_sims) * 0.7 + np.mean(cf_sims) * 0.3
        
        cf_similarity = (cf_similarity + 1.0) / 2.0
    
    if content_user_indices:
        target_content_vec = compatibility_embeddings[int(target_content_idx)].reshape(1, -1)
        user_content_vecs = compatibility_embeddings[content_user_indices]
        
        content_sims = cosine_similarity(target_content_vec, user_content_vecs).flatten()
        
        if content_weights is not None:
            content_similarity = np.average(content_sims, weights=content_weights)
        else:
            content_similarity = np.max(content_sims) * 0.7 + np.mean(content_sims) * 0.3
    
    if cf_similarity is not None and content_similarity is not None:
        cf_weight = 0.65 * cf_boost
        content_weight = 0.35
        total_weight = cf_weight + content_weight
        cf_weight /= total_weight
        content_weight /= total_weight
        
        final_similarity = cf_weight * cf_similarity + content_weight * content_similarity
        
    elif cf_similarity is not None:
        final_similarity = cf_similarity
        
    elif content_similarity is not None:
        final_similarity = content_similarity
    else:
        return 0.0
    
    if final_similarity >= 0.7:
        score = 80 + (final_similarity - 0.7) / 0.3 * 20
    elif final_similarity >= 0.5:
        score = score_floor + (final_similarity - 0.5) / 0.2 * (80 - score_floor)
    else:
        score = 1 + (final_similarity / 0.5) * (score_floor - 1)
    
    score = np.clip(score, 1, 100)
    return float(score)