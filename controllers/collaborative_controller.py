from typing import List
from fastapi import APIRouter, Query, HTTPException
from recommender.AnimeDataLoader import enrich_scored_recommendations

from models.models import (
    RecommendedAnime,
    PredictionResponse,
    SimilarUserResponse
)

from recommender.hybrid_recommender import (
    get_cf_recommendations_from_favorites,
    get_cf_similar_anime,
    predict_rating_from_favorites,
    get_similar_users_from_favorites
)

router = APIRouter(prefix="/v1", tags=["Collaborative filtering"])

@router.get("/cf/recommend/user", response_model=List[RecommendedAnime])
def cf_recommend_for_user(
        favourite_anime_ids: List[int] = Query(..., description="List of user's favourite anime ids."),
        limit: int = Query(default=10, ge=1, le=100)
):
    """Get collaborative filtering recommendations with full anime details"""
    try:
        results = get_cf_recommendations_from_favorites(favourite_anime_ids, limit)
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


@router.get("/cf/similar/anime", response_model=List[RecommendedAnime])
def cf_similar_anime(
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


@router.get("/cf/predict", response_model=PredictionResponse)
def predict_rating(
        favourite_anime_ids: List[int] = Query(..., description="List of user's favourite anime ids."),
        anime_id: int = Query(..., description="Anime ID")
):
    """Predict what rating a user would give to an anime"""
    try:
        prediction = predict_rating_from_favorites(favourite_anime_ids, anime_id)
        if prediction is None:
            raise HTTPException(status_code=404, detail="User or anime not found in CF system.")

        return PredictionResponse(
            user_favourites=favourite_anime_ids,
            anime_id=anime_id,
            predicted_rating=prediction
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting rating: {str(e)}")


@router.get("/cf/similar/users", response_model=List[SimilarUserResponse])
def similar_users(
        favourite_anime_ids: List[int] = Query(..., description="List of user's favourite anime ids."),
        limit: int = Query(default=10, ge=1, le=100)
):
    """Find users with similar taste"""
    try:
        results = get_similar_users_from_favorites(favourite_anime_ids, limit)
        if not results:
            raise HTTPException(status_code=404, detail="User not found in the system.")

        return [SimilarUserResponse(user_id=uid, similarity_score=score) for uid, score in results]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar users: {str(e)}")