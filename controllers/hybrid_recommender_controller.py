from typing import List, Dict
from fastapi import APIRouter, HTTPException
from recommender.AnimeDataLoader import enrich_hybrid_recommendations, enrich_recommendation_with_similarity

from recommender.hybrid_recommender import (
    get_hybrid_recommendations_from_favorites,
    get_hybrid_recommendations_with_text_from_favorites,
    get_cf_similar_anime
)

from models.models import (
    DetailedRecommendedAnime,
    HybridRecommendationRequest,
    HybridTextRecommendationRequest,
    HybridSimilarAnimeResponse
)

router = APIRouter(prefix="/v1/hybrid", tags=["Hybrid recommender"])

@router.get("/similar/{anime_id}", response_model=List[HybridSimilarAnimeResponse])
def hybrid_similar(anime_id: int, limit: int = 10):
    """
    Get hybrid similar anime (CF + Content) based on a single anime ID.
    """
    try:
        results = get_cf_similar_anime(anime_id, limit)
        if not results:
            raise HTTPException(status_code=404, detail="Anime not found or no similar entries.")
        return [HybridSimilarAnimeResponse(anime_id=anime_id, similarity_score=similarity_score) for anime_id, similarity_score in results ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting hybrid similar anime: {str(e)}")


@router.get("/similar/{anime_id}/detailed", response_model=List[Dict])
def hybrid_similar_detailed(anime_id: int, limit: int = 10):
    """
    Get hybrid similar anime (CF + Content) with full details based on a single anime ID.
    """
    try:
        results = get_cf_similar_anime(anime_id, limit)
        if not results:
            raise HTTPException(status_code=404, detail="Anime not found or no similar entries.")

        enriched = enrich_recommendation_with_similarity(results)
        if not enriched:
            raise HTTPException(status_code=500, detail="Failed to enrich recommendations with details.")

        return enriched
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting hybrid similar anime: {str(e)}")


@router.post("/recommend", response_model=List[int])
def hybrid_recommend(request: HybridRecommendationRequest):
    """
    Get hybrid recommendations (CF + Content) as a list of anime IDs.
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

        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting hybrid recommendations: {str(e)}")    

@router.post("/recommend/detailed", response_model=List[DetailedRecommendedAnime])
def hybrid_recommend_detailed(request: HybridRecommendationRequest):
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

@router.post("/recommend/text", response_model=List[int])
def hybrid_recommend_with_text(request: HybridTextRecommendationRequest):
    """
    Get hybrid recommendations (CF + Content + NLP) as a list of anime IDs.
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

        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting hybrid recommendations: {str(e)}")

@router.post("/recommend/text/detailed", response_model=List[DetailedRecommendedAnime])
def hybrid_recommend_with_text_detailed(request: HybridTextRecommendationRequest):
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
