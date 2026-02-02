from typing import List
from fastapi import APIRouter, Query, HTTPException
from data.loader.AnimeDataLoader import enrich_scored_recommendations

from models import (
    CollaborativeRecommendationRequest,
    RecommendedAnime,
    PredictionResponse,
    SimilarUserResponse
)

from recommender import (
    get_cf_recommendations_from_favorites,
    get_cf_similar_anime,
    predict_rating_from_favorites,
    get_similar_users_from_favorites
)

router = APIRouter(prefix="/v1", tags=["Collaborative filtering"])

@router.post("/cf/recommend/user", response_model=List[int])
def cf_recommend_for_user_ids(request: CollaborativeRecommendationRequest):
    """Get collaborative filtering recommendations as anime IDs"""
    try:
        results = get_cf_recommendations_from_favorites(request.user_favourite_ids, request.limit)
        if not results:
            raise HTTPException(status_code=404, detail="User not found in the system.")

        return [aid for aid, _ in results]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting CF recommendations: {str(e)}")

@router.post("/cf/recommend/user/detailed", response_model=List[RecommendedAnime])
def cf_recommend_for_user(request: CollaborativeRecommendationRequest): 
    """Get collaborative filtering recommendations with full anime details"""
    try:
        results = get_cf_recommendations_from_favorites(request.user_favourite_ids, request.limit)
        if not results:
            raise HTTPException(status_code=404, detail="User not found in the system.")

        enriched = enrich_scored_recommendations(results)
        if not enriched:
            raise HTTPException(status_code=500, detail="Failed to enrich recommendations with details.")

        return enriched
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting CF recommendations: {str(e)}")

@router.get("/cf/recommend", response_model=List[int])
def cf_recommend_anime(
        anime_id: int = Query(..., description="Anime ID"),
        limit: int = Query(default=10, ge=1, le=100)
):
    """Get collaborative filtering recommendations based on an anime ID"""
    try:
        results = get_cf_similar_anime(anime_id, limit)
        if not results:
            raise HTTPException(status_code=404, detail="Anime not found in CF system.")

        return [aid for aid, _ in results]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting CF recommendations: {str(e)}")

@router.get("/cf/recommend/detailed", response_model=List[RecommendedAnime])
def cf_recommend_anime_detailed(
        anime_id: int = Query(..., description="Anime ID"),
        limit: int = Query(default=10, ge=1, le=100)
):
    """Find similar anime using collaborative filtering with full details"""
    try:
        results = get_cf_similar_anime(anime_id, limit)
        if not results:
            raise HTTPException(status_code=404, detail="Anime not found in CF system.")

        enriched = enrich_scored_recommendations(results)
        if not enriched:
            raise HTTPException(status_code=500, detail="Failed to enrich recommendations with details.")

        return enriched
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar anime: {str(e)}")

@router.post("/cf/predict", response_model=PredictionResponse)
def predict_rating(request: CollaborativeRecommendationRequest, anime_id: int = Query(..., description="Anime ID")):
    """Predict what rating a user would give to an anime"""
    try:
        prediction = predict_rating_from_favorites(request.user_favourite_ids, anime_id)
        if prediction is None:
            raise HTTPException(status_code=404, detail="User or anime not found in CF system.")

        return PredictionResponse(
            user_favourites=request.user_favourite_ids,
            anime_id=anime_id,
            predicted_rating=prediction
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting rating: {str(e)}")

@router.post("/cf/similar/users", response_model=List[SimilarUserResponse])
def similar_users(request: CollaborativeRecommendationRequest):
    """Find users with similar taste"""
    try:
        results = get_similar_users_from_favorites(request.user_favourite_ids, request.limit)
        if not results:
            raise HTTPException(status_code=404, detail="User not found in the system.")

        return [SimilarUserResponse(user_id=uid, similarity_score=score) for uid, score in results]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar users: {str(e)}")