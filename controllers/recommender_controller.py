from typing import List
from functools import lru_cache
from fastapi import APIRouter, HTTPException, Query, Body
from models import  RecommendationResponse, RecommendedAnime
from data.loader import enrich_scored_recommendations

from recommender import (
    get_recommendations,
    get_recommendations_by_list,
    get_recommendations_from_text_with_scores,
    get_recommendations_semantic_search,
)

router = APIRouter(prefix="/v1", tags=["Recommender content based"])

@lru_cache(maxsize=1024)
@router.get("/recommend", response_model=List[int])
def recommend(anime_id: int, limit: int = Body(default=10, ge=1, le=100)):
    """Get similar anime IDs based on content embeddings (fast, lightweight)"""
    results = get_recommendations(anime_id, limit)
    if not results:
        raise HTTPException(status_code=404, detail="Anime not found or no similar entries.")
    return [ aid for aid, _ in results]

@router.get("/recommend/detailed", response_model=List[RecommendedAnime])
def recommend_detailed(anime_id: int, limit: int = Query(default=10, ge=1, le=100)):
    """Get similar anime with full details and similarity scores"""
    results = get_recommendations(anime_id, limit)
    
    if not results:
        raise HTTPException(status_code=404, detail="Anime not found or no similar entries.")

    enriched = enrich_scored_recommendations(results)

    if not enriched:
        raise HTTPException(status_code=500, detail="Failed to enrich recommendations with details.")

    return enriched

@router.post("/recommend_batch", response_model=List[int])
def recommend_batch(anime_ids: List[int], limit: int = Body(default=10, ge=1, le=100)):
    """Get recommendations based on a list of anime (averaged profile)"""
    result = get_recommendations_by_list(anime_ids, limit)
    
    if not result:
        raise HTTPException(status_code=404, detail="No recommendation could be made.")
    
    return [ aid for aid, _ in result ]

@router.post("/recommend_batch/detailed", response_model=List[RecommendedAnime])
def recommend_batch_detailed(anime_ids: List[int], limit: int = Body(default=10, ge=1, le=100)):
    """Get batch recommendations with full anime details"""
    result = get_recommendations_by_list(anime_ids, limit)
    if not result:
        raise HTTPException(status_code=404, detail="No recommendation could be made.")

    enriched = enrich_scored_recommendations(result)

    if not enriched:
        raise HTTPException(status_code=500, detail="Failed to enrich recommendations with details.")

    return enriched

@router.get("/recommend/semantic/search", response_model=List[RecommendationResponse], tags=["NLP Search"])
def recommend_from_text_semantic(
        query: str = Query(..., description="Natural language query like 'action anime with magic'"),
        limit: int = Query(default=10, ge=1, le=100),
        min_similarity: float = Query(default=None, ge=1, le=1.1)
):
    """Get anime recommendations from natural language text query (IDs only)"""
    try:
        results = get_recommendations_semantic_search(query, limit)

        if not min_similarity:
            return [ {"anime_id": aid, "score": score} for aid, score in results ]
        else:
            return [ {"anime_id": aid, "score": score} for aid, score in results if score >= min_similarity ]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text query: {str(e)}")
    

@router.get("/recommend/semantic/search/detailed", response_model=List[RecommendedAnime], tags=["NLP Search"])
def recommend_from_text_semantic_detailed(
        query: str = Query(..., description="Natural language query like 'action anime with magic'"),
        limit: int = Query(default=10, ge=1, le=100),
        min_similarity: float = Query(default=None, ge=1, le=1.1)
):
    """Get anime recommendations from natural language text query (IDs only)"""
    try:
        results = get_recommendations_semantic_search(query, limit)
        enriched = enrich_scored_recommendations(results)

        if not min_similarity:
            return enriched
        else:
            return [ result for result in enriched if result["score"] >= min_similarity ]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text query: {str(e)}")
    
@router.get("/recommend/text", response_model=List[RecommendationResponse], tags=["NLP Search"])
def recommend_from_text(
        query: str = Query(..., description="Natural language query"),
        limit: int = Query(default=10, ge=1, le=100)
):
    """Get anime recommendations with full details and similarity scores"""
    try:
        results = get_recommendations_from_text_with_scores(query, limit)
        return [ {"anime_id": aid, "score": score} for aid, score in results]
    
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
