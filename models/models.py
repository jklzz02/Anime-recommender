from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class AnimeDetail(BaseModel):
    """Full anime details"""
    id: int = Field(..., description="Anime ID")
    title: str = Field(..., description="Anime title")
    score: Optional[float] = Field(None, description="Average user rating", ge=0, le=10)
    release_year: Optional[int] = Field(None, description="Year of release")
    synopsis: Optional[str] = Field(None, description="Anime synopsis/description")
    episodes: Optional[int] = Field(None, description="Number of episodes")
    studio: Optional[str] = Field(None, description="Production studio")
    rating: Optional[str] = Field(None, description="Age rating (e.g., PG-13, R)")
    type: Optional[str] = Field(None, description="Type (TV, Movie, OVA, etc.)")
    source: Optional[str] = Field(None, description="Source material (Manga, Original, etc.)")
    genres: Optional[str] = Field(None, description="Comma-separated genre list")
    started_airing: Optional[str] = Field(None, description="Start date of airing")

class RecommendedAnime(AnimeDetail):
    """Anime details with recommendation score"""
    recommendation_score: float = Field(
        ...,
        description="Recommendation strength score (scale varies by endpoint)",
        ge=0
    )

class DetailedRecommendedAnime(RecommendedAnime):
    """Anime details with score breakdown"""
    score_breakdown: Dict[str, float] = Field(
        ...,
        description="Breakdown showing contribution from each recommendation method"
    )

class RecommendationResponse(BaseModel):
    """Simple recommendation response with ID and score"""
    anime_id: int
    score: float

class DetailedRecommendationResponse(BaseModel):
    """Simple response with score breakdown"""
    anime_id: int
    score: float
    score_breakdown: Optional[Dict[str, float]] = None

class CompatibilityBatchRequest(BaseModel):
    target_anime_ids: List[int]
    user_anime_ids: List[int]

class HybridRecommendationRequest(BaseModel):
    user_id: int
    user_anime_list: Optional[List[int]] = None
    limit: int = Field(default=10, ge=1, le=100)
    cf_weight: float = Field(default=0.5, ge=0, le=1)
    content_weight: float = Field(default=0.5, ge=0, le=1)

class HybridTextRecommendationRequest(BaseModel):
    user_id: int
    text_query: str
    limit: int = Field(default=10, ge=1, le=100)
    cf_weight: float = Field(default=0.33, ge=0, le=1)
    content_weight: float = Field(default=0.33, ge=0, le=1)
    nlp_weight: float = Field(default=0.34, ge=0, le=1)

class ColdStartRequest(BaseModel):
    text_query: str
    favorite_genres: Optional[List[str]] = None
    limit: int = Field(default=10, ge=1, le=100)

class CompatibilityResponse(BaseModel):
    """Response for compatibility scoring"""
    target_anime_id: int
    compatibility_score: float = Field(..., ge=1, le=100)
    scale: str = "1-100"

class CompatibilityBatchResponse(BaseModel):
    """Response for batch compatibility scoring"""
    scores: Dict[int, float]
    scale: str = "1-100"

class PredictionResponse(BaseModel):
    """Response for rating prediction"""
    user_id: int
    anime_id: int
    predicted_rating: float = Field(..., ge=1, le=10)
    scale: str = "1-10"

class SimilarUserResponse(BaseModel):
    """Response for similar users"""
    user_id: int
    similarity_score: float = Field(..., ge=0, le=1)