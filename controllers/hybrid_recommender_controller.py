from typing import List
from fastapi import APIRouter, HTTPException
from recommender.AnimeDataLoader import enrich_hybrid_recommendations, get_anime_details

from recommender.hybrid_recommender import (
    calculate_compatibility_score,
    get_most_compatible_from_favourites,
    get_hybrid_recommendations_from_favorites,
    get_hybrid_recommendations_with_text_from_favorites
)

from models.models import (
    CollaborativeRecommendationRequest,
    CompatibilityRequest,
    CompatibilityResponse,
    DetailedRecommendedAnime,
    HybridRecommendationRequest,
    HybridTextRecommendationRequest
)

router = APIRouter(prefix="/v1", tags=["Hybrid recommender"])

@router.post("/v1/hybrid/recommend", response_model=List[DetailedRecommendedAnime])
def hybrid_recommend(request: HybridRecommendationRequest):
    """
    Get hybrid recommendations (CF + Content) with full anime details and score breakdown.
    This is the recommended approach for best results.
    """
    try:
        results = get_hybrid_recommendations_from_favorites(
            user_anime_ids=request.user_anime_list,
            limit=request.limit,
            cf_weight=request.cf_weight,
            content_weight=request.content_weight
        )

        if not results:
            raise HTTPException(status_code=404, detail="No recommendations found.")

        enriched = enrich_hybrid_recommendations(results)
        if not enriched:
            raise HTTPException(status_code=500, detail="Failed to enrich recommendations with details.")

        return enriched
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting hybrid recommendations: {str(e)}")

@router.post("/v1/hybrid/recommend/text", response_model=List[DetailedRecommendedAnime])
def hybrid_recommend_with_text(request: HybridTextRecommendationRequest):
    """
    Get hybrid recommendations (CF + Content + NLP) with full details.
    Perfect for queries like "show me action anime like what I usually watch"
    """
    try:
        results = get_hybrid_recommendations_with_text_from_favorites(
            user_anime_ids=request.user_anime_list,
            text_query=request.text_query if request.text_query else "",
            limit=request.limit,
            cf_weight=request.cf_weight,
            content_weight=request.content_weight,
            nlp_weight=request.nlp_weight
        )

        if not results:
            raise HTTPException(status_code=404, detail="No recommendations found.")

        enriched = enrich_hybrid_recommendations(results)
        if not enriched:
            raise HTTPException(status_code=500, detail="Failed to enrich recommendations with details.")

        return enriched
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting hybrid recommendations: {str(e)}")
    

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

@router.post("/compatible", tags=["Compatibility"])
def get_most_compatible(request: CollaborativeRecommendationRequest):
    """Get compatibility scores for multiple anime based on user favourites"""
    try:
        compatibility_result = get_most_compatible_from_favourites(request.user_favourite_ids, request.limit)
        anime_results = [{"anime_id": aid, "compatibility_score": score} for aid, score in compatibility_result]
        anime_results.sort(key=lambda x: x["compatibility_score"], reverse=True)

        return {
            "data": anime_results,
            "scale": "1-100"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating batch compatibility: {str(e)}")
    
@router.post("/compatibile/detailed", tags=["Compatibility"])
def get_compatible_detailed(request: CollaborativeRecommendationRequest):
    """Get compatibility scores with full anime details for multiple anime"""
    try:
        compatibility_result = get_most_compatible_from_favourites(request.user_favourite_ids, request.limit)
        detailed_results = []
        for aid, score in compatibility_result:
            anime = get_anime_details(aid)
            if anime:
                detailed_results.append({
                    "anime": anime,
                    "compatibility_score": score
                })

        detailed_results.sort(key=lambda x: x["compatibility_score"], reverse=True)

        return {
            "data": detailed_results,
            "scale": "1-100"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating batch compatibility: {str(e)}")
