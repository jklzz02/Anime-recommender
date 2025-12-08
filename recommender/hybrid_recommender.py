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

model = SentenceTransformer("intfloat/e5-base")

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
