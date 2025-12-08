from typing import List
from fastapi import APIRouter, HTTPException
from recommender.AnimeDataLoader import enrich_hybrid_recommendations

from recommender.hybrid_recommender import (
    get_hybrid_recommendations_from_favorites,
    get_hybrid_recommendations_with_text_from_favorites
)

from models.models import (
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
