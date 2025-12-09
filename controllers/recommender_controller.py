from typing import List
from fastapi import APIRouter, HTTPException, Query
from models.models import CompatibilityRequest, RecommendedAnime, CompatibilityResponse, CompatibilityBatchResponse, CompatibilityBatchRequest
from recommender.AnimeDataLoader import enrich_scored_recommendations, get_anime_details

from recommender.recommender import (
    get_recommendations,
    get_recommendations_by_list,
    get_recommendations_from_text,
    get_recommendations_from_text_with_scores,
    calculate_compatibility_score,
    calculate_compatibility_scores_batch,
    get_high_compatibility_recommendations
)

router = APIRouter(prefix="/v1", tags=["Recommender content based"])

@router.get("/recommend", response_model=List[int])
def recommend(anime_id: int, limit: int = Query(default=10, ge=1, le=100)):
    """Get similar anime IDs based on content embeddings (fast, lightweight)"""
    results = get_recommendations(anime_id, limit)
    if not results:
        raise HTTPException(status_code=404, detail="Anime not found or no similar entries.")
    return results

@router.get("/recommend/detailed", response_model=List[RecommendedAnime])
def recommend_detailed(anime_id: int, limit: int = Query(default=10, ge=1, le=100)):
    """Get similar anime with full details and similarity scores"""
    results = get_recommendations(anime_id, limit)
    if not results:
        raise HTTPException(status_code=404, detail="Anime not found or no similar entries.")

    scored_results = [(aid, 1.0 - (i * 0.05)) for i, aid in enumerate(results)]
    enriched = enrich_scored_recommendations(scored_results)

    if not enriched:
        raise HTTPException(status_code=500, detail="Failed to enrich recommendations with details.")

    return enriched

@router.get("/recommend_batch", response_model=List[int])
def recommend_batch(anime_ids: List[int] = Query(...), limit: int = Query(default=10, ge=1, le=100)):
    """Get recommendations based on a list of anime (averaged profile)"""
    result = get_recommendations_by_list(anime_ids, limit)
    if not result:
        raise HTTPException(status_code=404, detail="No recommendation could be made.")
    return result

@router.get("/recommend_batch/detailed", response_model=List[RecommendedAnime])
def recommend_batch_detailed(anime_ids: List[int] = Query(...), limit: int = Query(default=10, ge=1, le=100)):
    """Get batch recommendations with full anime details"""
    result = get_recommendations_by_list(anime_ids, limit)
    if not result:
        raise HTTPException(status_code=404, detail="No recommendation could be made.")

    scored_results = [(aid, 1.0 - (i * 0.05)) for i, aid in enumerate(result)]
    enriched = enrich_scored_recommendations(scored_results)

    if not enriched:
        raise HTTPException(status_code=500, detail="Failed to enrich recommendations with details.")

    return enriched

@router.get("/recommend/text", response_model=List[int], tags=["NLP Search"])
def recommend_from_text(
        query: str = Query(..., description="Natural language query like 'action anime with magic'"),
        limit: int = Query(default=10, ge=1, le=100)
):
    """Get anime recommendations from natural language text query (IDs only)"""
    try:
        results = get_recommendations_from_text(query, limit)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text query: {str(e)}")

@router.get("/recommend/text/detailed", response_model=List[RecommendedAnime], tags=["NLP Search"])
def recommend_from_text_detailed(
        query: str = Query(..., description="Natural language query"),
        limit: int = Query(default=10, ge=1, le=100)
):
    """Get anime recommendations with full details and similarity scores"""
    try:
        results = get_recommendations_from_text_with_scores(query, limit)
        enriched = enrich_scored_recommendations(results)

        if not enriched:
            raise HTTPException(status_code=500, detail="Failed to enrich recommendations with details.")

        return enriched
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text query: {str(e)}")


@router.post("/compatibility/score", response_model=CompatibilityResponse, tags=["Compatibility"])
def get_compatibility(request: CompatibilityRequest):
    """Calculate compatibility score (1-100) for a target anime"""
    try:
        score = calculate_compatibility_score(request.target_anime_id, request.user_favourite_ids)
        return CompatibilityResponse(
            target_anime_id=request.target_anime_id,
            compatibility_score=score
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating compatibility: {str(e)}")


@router.post("/compatibility/score/detailed", tags=["Compatibility"])
def get_compatibility_detailed(request: CompatibilityRequest):
    """Get compatibility score with full anime details"""
    try:
        score = calculate_compatibility_score(request.target_anime_id, request.user_favourite_ids)
        anime = get_anime_details(request.target_anime_id)
        if not anime:
            raise HTTPException(status_code=404, detail="Anime not found")

        return {
            "anime": anime,
            "compatibility_score": score,
            "scale": "1-100"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating compatibility: {str(e)}")


@router.post("/compatibility/batch", response_model=CompatibilityBatchResponse, tags=["Compatibility"])
def get_compatibility_batch(request: CompatibilityBatchRequest):
    """Calculate compatibility scores for multiple anime at once"""
    try:
        scores = calculate_compatibility_scores_batch(
            request.target_anime_ids,
            request.user_favourite_ids
        )
        return CompatibilityBatchResponse(scores=scores)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating compatibility: {str(e)}")


@router.get("/compatibility/recommendations", response_model=List[RecommendedAnime], tags=["Compatibility"])
def get_high_compatibility(
        user_anime_ids: List[int] = Query(..., description="User's watched/liked anime"),
        limit: int = Query(default=10, ge=1, le=100),
        min_score: float = Query(default=50.0, ge=1, le=100)
):
    """Get recommendations with high compatibility scores and full details"""
    try:
        results = get_high_compatibility_recommendations(user_anime_ids, limit, min_score)
        enriched = enrich_scored_recommendations(results)

        if not enriched:
            raise HTTPException(status_code=404, detail="No recommendations found.")

        return enriched
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")
