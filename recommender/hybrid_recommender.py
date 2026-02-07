import numpy as np
import json
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from loader import get_transformer

content_embeddings = np.load("data/embeddings/anime_embeddings.npy")
nlp_embeddings = np.load("data/embeddings/anime_nlp_embeddings.npy")
compatibility_embeddings = np.load("data/embeddings/anime_compatibility_embeddings.npy")
anime_cf_embeddings = np.load("data/embeddings/anime_cf_embeddings.npy")
user_embeddings = np.load("data/embeddings/user_embeddings.npy")

model = get_transformer()

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

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _validate_and_get_indices(user_anime_ids: List[int]) -> Tuple[List[int], List[int]]:
    """Validate anime IDs and get their indices in both CF and content spaces"""
    cf_user_ids = [aid for aid in user_anime_ids if aid in anime_to_cf_idx]
    content_user_indices = [
        id_to_index[str(aid)] for aid in user_anime_ids 
        if str(aid) in id_to_index
    ]
    return cf_user_ids, content_user_indices


def _compute_weighted_average(vectors: np.ndarray, 
                               anime_ids: List[int], 
                               user_ratings: Optional[Dict[int, float]],
                               id_getter=None) -> np.ndarray:
    """
    Compute weighted average of vectors based on user ratings
    Higher ratings get exponentially more weight
    """
    if user_ratings:
        weights = np.array([
            max(0, user_ratings.get(id_getter(aid) if id_getter else aid, 7.0) - 5.0) / 5.0
            for aid in anime_ids
        ])
        weights = np.power(weights + 0.5, 2)
        
        if weights.sum() > 0:
            weights = weights / weights.sum()
            return np.average(vectors, axis=0, weights=weights).reshape(1, -1)
    
    return np.mean(vectors, axis=0).reshape(1, -1)


def _compute_similarity_with_weights(target_vec: np.ndarray,
                                     user_vecs: np.ndarray,
                                     anime_ids: List[int],
                                     user_ratings: Optional[Dict[int, float]],
                                     id_getter=None) -> float:
    """
    Compute similarity between target and user vectors with rating-based weighting
    """
    sims = cosine_similarity(target_vec, user_vecs).flatten()
    
    if user_ratings:
        weights = np.array([
            max(0, user_ratings.get(id_getter(aid) if id_getter else aid, 7.0) - 5.0) / 5.0
            for aid in anime_ids
        ])
        weights = np.power(weights + 0.5, 2)
        if weights.sum() > 0:
            weights = weights / weights.sum()
            return np.average(sims, weights=weights)
    
    return np.max(sims) * 0.7 + np.mean(sims) * 0.3


def _compute_final_similarity(cf_similarity: Optional[float],
                               content_similarity: Optional[float],
                               cf_boost: float = 1.2) -> Optional[float]:
    """
    Combine CF and content similarities with optional CF boost
    """
    if cf_similarity is not None and content_similarity is not None:
        cf_weight = 0.65 * cf_boost
        content_weight = 0.35
        total_weight = cf_weight + content_weight
        cf_weight /= total_weight
        content_weight /= total_weight
        return cf_weight * cf_similarity + content_weight * content_similarity
    elif cf_similarity is not None:
        return cf_similarity
    elif content_similarity is not None:
        return content_similarity
    return None


def _similarity_to_score(similarity: float, score_floor: float = 50.0) -> float:
    """
    Convert cosine similarity to 1-100 compatibility score
    """
    if similarity >= 0.7:
        score = 80 + (similarity - 0.7) / 0.3 * 20
    elif similarity >= 0.5:
        score = score_floor + (similarity - 0.5) / 0.2 * (80 - score_floor)
    else:
        score = 1 + (similarity / 0.5) * (score_floor - 1)
    return np.clip(score, 1, 100)


def calculate_compatibility_score(target_anime_id: int, 
                                   user_anime_ids: List[int],
                                   user_ratings: Optional[Dict[int, float]] = None,
                                   score_floor: float = 50.0,
                                   cf_boost: float = 1.2) -> float:
    """
    Calculate compatibility score between target anime and user's profile
    Combines collaborative filtering and content-based similarity
    
    Returns: Score from 0-100
    """
    target_content_idx = id_to_index.get(str(target_anime_id))
    target_cf_idx = anime_to_cf_idx.get(target_anime_id)
    
    if target_content_idx is None or not user_anime_ids:
        return 0.0
    
    cf_user_ids, content_user_indices = _validate_and_get_indices(user_anime_ids)
    
    if not cf_user_ids and not content_user_indices:
        return 0.0
    
    cf_similarity = None
    if target_cf_idx is not None and cf_user_ids:
        target_cf_vec = anime_cf_embeddings[target_cf_idx].reshape(1, -1)
        user_cf_vecs = anime_cf_embeddings[[anime_to_cf_idx[aid] for aid in cf_user_ids]]
        cf_similarity = _compute_similarity_with_weights(
            target_cf_vec, user_cf_vecs, cf_user_ids, user_ratings
        )
        cf_similarity = (cf_similarity + 1.0) / 2.0
    
    content_similarity = None
    if content_user_indices:
        target_content_vec = compatibility_embeddings[int(target_content_idx)].reshape(1, -1)
        user_content_vecs = compatibility_embeddings[content_user_indices]
        content_similarity = _compute_similarity_with_weights(
            target_content_vec, user_content_vecs, content_user_indices, user_ratings,
            lambda idx: int(index_to_id[idx])
        )
    
    final_similarity = _compute_final_similarity(cf_similarity, content_similarity, cf_boost)
    
    if final_similarity is None:
        return 0.0
    
    score = _similarity_to_score(final_similarity, score_floor)
    return float(round(score, 2))


def get_most_compatible_from_favourites(user_anime_ids: List[int],
                                        limit: int = 10,
                                        user_ratings: Optional[Dict[int, float]] = None,
                                        score_floor: float = 50.0,
                                        cf_boost: float = 1.2,
                                        min_score: float = 60.0,
                                        exclude_ids: Optional[List[int]] = None) -> List[Tuple[int, float]]:
    """
    Find most compatible anime based on user's favorites
    
    Args:
        user_anime_ids: List of anime IDs in user's list
        limit: Maximum number of recommendations
        user_ratings: Optional dict mapping anime_id -> rating
        score_floor: Minimum score for medium similarity matches
        cf_boost: Boost factor for CF component
        min_score: Minimum compatibility score to include
        exclude_ids: Additional anime IDs to exclude
    
    Returns:
        List of (anime_id, compatibility_score) tuples
    """
    if not user_anime_ids:
        return []
    
    excluded = set(user_anime_ids)
    if exclude_ids:
        excluded.update(exclude_ids)
    
    all_anime_ids = set(index_to_id.values())
    candidate_ids = all_anime_ids - excluded
    
    if not candidate_ids:
        return []
    
    scored_anime = []
    for anime_id in candidate_ids:
        score = calculate_compatibility_score(
            target_anime_id=anime_id,
            user_anime_ids=user_anime_ids,
            user_ratings=user_ratings,
            score_floor=score_floor,
            cf_boost=cf_boost
        )
        if score >= min_score:
            scored_anime.append((anime_id, score))
    
    scored_anime.sort(key=lambda x: x[1], reverse=True)
    return scored_anime[:limit]


def get_cf_recommendations_from_favorites(user_anime_ids: List[int], 
                                          limit: int = 10,
                                          user_ratings: Optional[Dict[int, float]] = None) -> List[Tuple[int, float]]:
    """
    Pure collaborative filtering recommendations based on user's favorites
    """
    cf_user_ids, _ = _validate_and_get_indices(user_anime_ids)
    if not cf_user_ids:
        return []

    user_cf_vecs = anime_cf_embeddings[[anime_to_cf_idx[aid] for aid in cf_user_ids]]
    pseudo_user_vector = _compute_weighted_average(user_cf_vecs, cf_user_ids, user_ratings)
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
    """
    Find similar anime using collaborative filtering
    Based on users who liked the same anime
    """
    cf_idx = anime_to_cf_idx.get(anime_id)
    if cf_idx is None:
        return []

    anime_vector = anime_cf_embeddings[cf_idx].reshape(1, -1)
    similarities = cosine_similarity(anime_vector, anime_cf_embeddings).flatten()

    sorted_indices = similarities.argsort()[::-1][1:limit + 1]
    return [(cf_idx_to_anime[idx], float(similarities[idx])) for idx in sorted_indices]


def get_hybrid_recommendations_from_favorites(user_anime_ids: List[int],
                                              limit: int = 10,
                                              user_ratings: Optional[Dict[int, float]] = None,
                                              cf_weight: float = 0.5,
                                              content_weight: float = 0.5) -> List[Tuple[int, float, Dict]]:
    """
    Hybrid recommendations combining CF and content-based approaches
    
    Returns:
        List of (anime_id, combined_score, score_breakdown) tuples
    """
    total_weight = cf_weight + content_weight
    cf_weight /= total_weight
    content_weight /= total_weight

    cf_recs = get_cf_recommendations_from_favorites(user_anime_ids, limit=limit * 3, user_ratings=user_ratings)
    cf_scores = {aid: score for aid, score in cf_recs}

    _, content_user_indices = _validate_and_get_indices(user_anime_ids)
    content_scores = {}
    if content_user_indices:
        user_content_vecs = compatibility_embeddings[content_user_indices]
        user_profile = _compute_weighted_average(
            user_content_vecs, 
            content_user_indices, 
            user_ratings,
            lambda idx: int(index_to_id[idx])
        )
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
                                                        user_ratings: Optional[Dict[int, float]] = None,
                                                        cf_weight: float = 0.33,
                                                        content_weight: float = 0.33,
                                                        nlp_weight: float = 0.34) -> List[Tuple[int, float, Dict]]:
    """
    Hybrid recommendations with user text query
    Combines CF, content similarity, and natural language search
    
    Args:
        user_anime_ids: User's favorite anime IDs
        text_query: Natural language query (e.g., "dark psychological thriller")
        limit: Number of recommendations
        user_ratings: Optional rating information
        cf_weight: Weight for collaborative filtering
        content_weight: Weight for content similarity
        nlp_weight: Weight for NLP text matching
    
    Returns:
        List of (anime_id, combined_score, score_breakdown) tuples
    """
    total_weight = cf_weight + content_weight + nlp_weight
    cf_weight /= total_weight
    content_weight /= total_weight
    nlp_weight /= total_weight

    cf_recs = get_cf_recommendations_from_favorites(user_anime_ids, limit=limit * 3, user_ratings=user_ratings)
    cf_scores = {aid: score for aid, score in cf_recs}

    _, content_user_indices = _validate_and_get_indices(user_anime_ids)
    content_scores = {}
    if content_user_indices:
        user_content_vecs = compatibility_embeddings[content_user_indices]
        user_profile = _compute_weighted_average(
            user_content_vecs,
            content_user_indices,
            user_ratings,
            lambda idx: int(index_to_id[idx])
        )
        similarities = cosine_similarity(user_profile, compatibility_embeddings).flatten()
        for idx, sim in enumerate(similarities):
            anime_id = int(index_to_id[idx])
            if anime_id not in user_anime_ids:
                content_scores[anime_id] = float(sim)

    query_with_prefix = QUERY_PREFIX + text_query
    query_embedding = model.encode(
        [query_with_prefix], 
        show_progress_bar=False,
        normalize_embeddings=True
    )[0].reshape(1, -1)
    
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


def predict_rating_from_favorites(user_anime_ids: List[int], 
                                   anime_id: int,
                                   user_ratings: Optional[Dict[int, float]] = None) -> Optional[float]:
    """
    Predict user's rating for an anime based on their favorites
    Uses collaborative filtering similarity
    
    Returns:
        Predicted rating (1-10) or None if prediction not possible
    """
    cf_user_ids, _ = _validate_and_get_indices(user_anime_ids)
    if not cf_user_ids or anime_id not in anime_to_cf_idx:
        return None

    user_cf_vecs = anime_cf_embeddings[[anime_to_cf_idx[aid] for aid in cf_user_ids]]
    pseudo_user_vector = _compute_weighted_average(user_cf_vecs, cf_user_ids, user_ratings)
    anime_vector = anime_cf_embeddings[anime_to_cf_idx[anime_id]].reshape(1, -1)
    similarity = cosine_similarity(pseudo_user_vector, anime_vector)[0][0]

    predicted_rating = 1 + similarity * 9
    predicted_rating = np.clip(predicted_rating, 1, 10)
    return float(predicted_rating)


def get_similar_users_from_favorites(user_anime_ids: List[int], 
                                      limit: int = 10,
                                      user_ratings: Optional[Dict[int, float]] = None) -> List[Tuple[int, float]]:
    """
    Find users with similar taste based on favorites
    
    Returns:
        List of (user_index, similarity_score) tuples
    """
    cf_user_ids, _ = _validate_and_get_indices(user_anime_ids)
    if not cf_user_ids:
        return []

    user_cf_vecs = anime_cf_embeddings[[anime_to_cf_idx[aid] for aid in cf_user_ids]]
    pseudo_user_vector = _compute_weighted_average(user_cf_vecs, cf_user_ids, user_ratings)
    similarities = cosine_similarity(pseudo_user_vector, user_embeddings).flatten()

    sorted_indices = similarities.argsort()[::-1][:limit]
    return [(idx, float(similarities[idx])) for idx in sorted_indices]


def hybrid_text_search(
    query: str, 
    limit: int = 10, 
    nlp_weight: float = 0.6, 
    content_weight: float = 0.4
) -> List[Tuple[int, float]]:
    """
    Combines NLP and content embeddings for better recommendations.
    
    Returns a list of (anime_id, combined_score).
    """
    total = nlp_weight + content_weight
    nlp_weight /= total
    content_weight /= total

    query_embedding = model.encode([query], normalize_embeddings=True)[0].reshape(1, -1)

    nlp_similarities = cosine_similarity(query_embedding, nlp_embeddings).flatten()
    content_similarities = cosine_similarity(query_embedding, content_embeddings).flatten()

    nlp_similarities /= np.max(nlp_similarities)
    content_similarities /= np.max(content_similarities)

    combined_scores = nlp_weight * nlp_similarities + content_weight * content_similarities
    similar_indices = combined_scores.argsort()[::-1][:limit]

    return [(int(index_to_id[i]), float(combined_scores[i])) for i in similar_indices]
