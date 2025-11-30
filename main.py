from typing import List
from fastapi import FastAPI, HTTPException, Query, status

from models.models import (
    AnimeDetail,
    RecommendedAnime,
    DetailedRecommendedAnime,
    CompatibilityResponse,
    CompatibilityBatchRequest,
    CompatibilityBatchResponse,
    HybridRecommendationRequest,
    HybridTextRecommendationRequest,
    ColdStartRequest,
    PredictionResponse,
    SimilarUserResponse
)

from recommender.recommender import (
    get_recommendations, 
    get_recommendations_by_list,
    get_recommendations_from_text,
    get_recommendations_from_text_with_scores,
    calculate_compatibility_score,
    calculate_compatibility_scores_batch,
    get_high_compatibility_recommendations
)

from recommender.hybrid_recommender import (
    get_cf_recommendations_for_user,
    get_cf_similar_anime,
    predict_user_rating,
    get_hybrid_recommendations,
    get_hybrid_recommendations_with_text,
    get_cold_start_recommendations,
    get_similar_users
)

from recommender.AnimeDataLoader import (
    get_anime_details,
    enrich_scored_recommendations,
    enrich_hybrid_recommendations,
    get_loader_status
)

app = FastAPI(
    title="Anime Recommendation API",
    description="Hybrid recommendation system combining content-based, collaborative filtering, and NLP",
    version="2.0.0"
)

@app.get("/v1", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "version": "2.0.0",
        "features": [
            "content-based recommendations",
            "collaborative filtering",
            "hybrid recommendations",
            "NLP text search",
            "compatibility scoring"
        ]
    }

@app.get("/v1/anime/{anime_id}", response_model=AnimeDetail, tags=["Health"])
async def get_anime(anime_id: int):
    """Get full details for a specific anime by ID"""
    anime = get_anime_details(anime_id)
    if not anime:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Anime with ID {anime_id} not found"
        )
    return anime

@app.get("/v1/health", tags=["Health"])
async def health_check():
    """Detailed health check with system status"""
    loader_status = get_loader_status()
    return {
        "status": "healthy" if loader_status["is_loaded"] else "degraded",
        "version": "2.0.0",
        "anime_loader": loader_status,
        "endpoints": {
            "content_based": ["/v1/recommend", "/v1/recommend/detailed", "/v1/recommend_batch"],
            "nlp": ["/v1/recommend/text", "/v1/recommend/text/detailed"],
            "compatibility": ["/v1/compatibility/score", "/v1/compatibility/recommendations"],
            "collaborative": ["/v1/cf/recommend/user", "/v1/cf/similar/anime", "/v1/cf/predict"],
            "hybrid": ["/v1/hybrid/recommend", "/v1/hybrid/recommend/text"],
            "cold_start": ["/v1/cold_start/recommend"],
            "utility": ["/v1/anime/{id}"]
        }
    }

@app.get("/v1/recommend", response_model=List[int], tags=["Content-Based"])
def recommend(anime_id: int, limit: int = Query(default=10, ge=1, le=100)):
    """Get similar anime IDs based on content embeddings (fast, lightweight)"""
    results = get_recommendations(anime_id, limit)
    if not results:
        raise HTTPException(status_code=404, detail="Anime not found or no similar entries.")
    return results

@app.get("/v1/recommend/detailed", response_model=List[RecommendedAnime], tags=["Content-Based"])
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

@app.get("/v1/recommend_batch", response_model=List[int], tags=["Content-Based"])
def recommend_batch(anime_ids: List[int] = Query(...), limit: int = Query(default=10, ge=1, le=100)):
    """Get recommendations based on a list of anime (averaged profile)"""
    result = get_recommendations_by_list(anime_ids, limit)
    if not result:
        raise HTTPException(status_code=404, detail="No recommendation could be made.")
    return result

@app.get("/v1/recommend_batch/detailed", response_model=List[RecommendedAnime], tags=["Content-Based"])
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

@app.get("/v1/recommend/text", response_model=List[int], tags=["NLP Search"])
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

@app.get("/v1/recommend/text/detailed", response_model=List[RecommendedAnime], tags=["NLP Search"])
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

@app.get("/v1/compatibility/score", response_model=CompatibilityResponse, tags=["Compatibility"])
def get_compatibility(
    target_anime_id: int = Query(..., description="Anime to score"),
    user_anime_ids: List[int] = Query(..., description="User's watched/liked anime")
):
    """Calculate compatibility score (1-100) for a target anime"""
    try:
        score = calculate_compatibility_score(target_anime_id, user_anime_ids)
        return CompatibilityResponse(
            target_anime_id=target_anime_id,
            compatibility_score=score
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating compatibility: {str(e)}")

@app.get("/v1/compatibility/score/detailed", tags=["Compatibility"])
def get_compatibility_detailed(
    target_anime_id: int = Query(..., description="Anime to score"),
    user_anime_ids: List[int] = Query(..., description="User's watched/liked anime")
):
    """Get compatibility score with full anime details"""
    try:
        score = calculate_compatibility_score(target_anime_id, user_anime_ids)
        anime = get_anime_details(target_anime_id)
        
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

@app.post("/v1/compatibility/batch", response_model=CompatibilityBatchResponse, tags=["Compatibility"])
def get_compatibility_batch(request: CompatibilityBatchRequest):
    """Calculate compatibility scores for multiple anime at once"""
    try:
        scores = calculate_compatibility_scores_batch(
            request.target_anime_ids,
            request.user_anime_ids
        )
        return CompatibilityBatchResponse(scores=scores)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating compatibility: {str(e)}")

@app.get("/v1/compatibility/recommendations", response_model=List[RecommendedAnime], tags=["Compatibility"])
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

@app.get("/v1/cf/recommend/user", response_model=List[RecommendedAnime], tags=["Collaborative Filtering"])
def cf_recommend_for_user(
    user_id: int = Query(..., description="User ID"),
    limit: int = Query(default=10, ge=1, le=100)
):
    """Get collaborative filtering recommendations with full anime details"""
    try:
        results = get_cf_recommendations_for_user(user_id, limit)
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

@app.get("/v1/cf/similar/anime", response_model=List[RecommendedAnime], tags=["Collaborative Filtering"])
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

@app.get("/v1/cf/predict", response_model=PredictionResponse, tags=["Collaborative Filtering"])
def predict_rating(
    user_id: int = Query(..., description="User ID"),
    anime_id: int = Query(..., description="Anime ID")
):
    """Predict what rating a user would give to an anime"""
    try:
        prediction = predict_user_rating(user_id, anime_id)
        if prediction is None:
            raise HTTPException(status_code=404, detail="User or anime not found in CF system.")
        
        return PredictionResponse(
            user_id=user_id,
            anime_id=anime_id,
            predicted_rating=prediction
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting rating: {str(e)}")

@app.get("/v1/cf/similar/users", response_model=List[SimilarUserResponse], tags=["Collaborative Filtering"])
def similar_users(
    user_id: int = Query(..., description="User ID"),
    limit: int = Query(default=10, ge=1, le=100)
):
    """Find users with similar taste"""
    try:
        results = get_similar_users(user_id, limit)
        if not results:
            raise HTTPException(status_code=404, detail="User not found in the system.")
        
        return [SimilarUserResponse(user_id=uid, similarity_score=score) for uid, score in results]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar users: {str(e)}")

@app.post("/v1/hybrid/recommend", response_model=List[DetailedRecommendedAnime], tags=["Hybrid"])
def hybrid_recommend(request: HybridRecommendationRequest):
    """
    Get hybrid recommendations (CF + Content) with full anime details and score breakdown.
    This is the recommended approach for best results.
    """
    try:
        results = get_hybrid_recommendations(
            user_id=request.user_id if request.user_id else 0,
            limit=request.limit,
            cf_weight=request.cf_weight,
            content_weight=request.content_weight,
            user_anime_list=request.user_anime_list
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

@app.post("/v1/hybrid/recommend/text", response_model=List[DetailedRecommendedAnime], tags=["Hybrid"])
def hybrid_recommend_with_text(request: HybridTextRecommendationRequest):
    """
    Get hybrid recommendations (CF + Content + NLP) with full details.
    Perfect for queries like "show me action anime like what I usually watch"
    """
    try:
        results = get_hybrid_recommendations_with_text(
            user_id =request.user_id if request.user_id else 0,
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

@app.post("/v1/cold_start/recommend", response_model=List[RecommendedAnime], tags=["Hybrid"])
def cold_start_recommend(request: ColdStartRequest):
    """
    Get recommendations for new users without rating history.
    Uses content-based filtering with text queries and genre preferences.
    """
    try:
        results = get_cold_start_recommendations(
            text_query=request.text_query if request.text_query else "",
            favorite_genres=request.favorite_genres,
            limit=request.limit
        )
        
        if not results:
            raise HTTPException(status_code=404, detail="No recommendations found.")
        
        enriched = enrich_scored_recommendations(results)
        if not enriched:
            raise HTTPException(status_code=500, detail="Failed to enrich recommendations with details.")
        
        return enriched
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting cold start recommendations: {str(e)}")

@app.get("/v1/docs/examples", tags=["Documentation"])
async def get_examples():
    """Get example requests for each endpoint"""
    return {
        "content_based": {
            "description": "Find similar anime based on content",
            "simple": "GET /v1/recommend?anime_id=1&limit=10",
            "detailed": "GET /v1/recommend/detailed?anime_id=1&limit=10"
        },
        "nlp_text": {
            "description": "Search using natural language",
            "simple": "GET /v1/recommend/text?query=dark+fantasy+with+magic&limit=10",
            "detailed": "GET /v1/recommend/text/detailed?query=dark+fantasy+with+magic&limit=10"
        },
        "compatibility": {
            "description": "Score how well an anime matches user's taste",
            "example": "GET /v1/compatibility/score?target_anime_id=100&user_anime_ids=1&user_anime_ids=5"
        },
        "collaborative_filtering": {
            "description": "Personalized recommendations based on user behavior",
            "example": "GET /v1/cf/recommend/user?user_id=1&limit=10"
        },
        "hybrid": {
            "description": "Best approach: CF + content-based",
            "example": {
                "method": "POST",
                "endpoint": "/v1/hybrid/recommend",
                "body": {
                    "user_id": 1,
                    "user_anime_list": [1, 5, 10],
                    "limit": 10,
                    "cf_weight": 0.6,
                    "content_weight": 0.4
                }
            }
        },
        "cold_start": {
            "description": "Recommendations for new users",
            "example": {
                "method": "POST",
                "endpoint": "/v1/cold_start/recommend",
                "body": {
                    "text_query": "action adventure with strong characters",
                    "favorite_genres": ["Action", "Adventure"],
                    "limit": 10
                }
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)