from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class RecommendationResponse(BaseModel):
    anime_id: int = Field(..., description="Unique anime identifier for the recommended item")
    score: float = Field(..., description="Recommendation score (e.g., similarity or relevance; scale may vary)")

class DetailedRecommendationResponse(BaseModel):
    anime_id: int = Field(..., description="Unique anime identifier for the recommended item")
    score: float = Field(..., description="Aggregated recommendation score")
    score_breakdown: Optional[Dict[str, float]] = Field(
        None,
        description="Optional breakdown of component scores keyed by method (e.g., 'cf', 'content', 'nlp')"
    )

class AnimeDetail(BaseModel):
    anime_id: int = Field(..., description="Unique anime identifier")
    title: str = Field(..., description="Primary title")
    synopsis: Optional[str] = Field(None, description="Short summary/description")
    genres: List[str] = Field(default_factory=list, description="List of genres")
    type: Optional[str] = Field(None, description="TV/Movie/OVA/etc")
    episodes: Optional[int] = Field(None, description="Number of episodes if applicable")
    score: Optional[float] = Field(None, description="Average community score")
    members: Optional[int] = Field(None, description="Number of members/users")
    url: Optional[str] = Field(None, description="External link to anime details")
    image_url: Optional[str] = Field(None, description="Poster/cover image URL")

class RecommendedAnime(BaseModel):
    anime_id: int = Field(..., description="Anime ID")
    title: Optional[str] = Field(None, description="Title if available")
    score: Optional[float] = Field(None, description="Similarity or recommendation score (0-1 or 0-100)")
    rank: Optional[int] = Field(None, description="Rank in the returned list (1 = best)")

class DetailedRecommendedAnime(BaseModel):
    anime_id: int = Field(..., description="Anime ID")
    anime: Optional[AnimeDetail] = Field(None, description="Full anime details when available")
    final_score: float = Field(..., description="Final aggregated recommendation score")
    cf_score: Optional[float] = Field(None, description="Collaborative-filtering contribution")
    content_score: Optional[float] = Field(None, description="Content-based contribution")
    nlp_score: Optional[float] = Field(None, description="NLP/text-search contribution")
    explanation: Optional[str] = Field(None, description="Optional human-readable explanation of the score")

class CompatibilityResponse(BaseModel):
    target_anime_id: int = Field(..., description="Target anime ID")
    compatibility_score: float = Field(..., description="Compatibility score (1-100)")

class CompatibilityBatchRequest(BaseModel):
    target_anime_ids: List[int] = Field(..., description="Targets to score")
    user_anime_ids: List[int] = Field(..., description="User's watched/liked anime")

class CompatibilityBatchResponse(BaseModel):
    scores: Dict[int, float] = Field(..., description="Mapping of target anime id to compatibility score")

class HybridRecommendationRequest(BaseModel):
    user_id: Optional[int] = Field(None, description="User id (optional if user_anime_list provided)")
    user_anime_list: List[int] = Field(default_factory=list, description="List of user's anime to base recommendations on")
    limit: int = Field(10, ge=1, le=100, description="Number of recommendations to return")
    cf_weight: float = Field(0.5, ge=0.0, le=1.0, description="Weight for collaborative filtering")
    content_weight: float = Field(0.5, ge=0.0, le=1.0, description="Weight for content-based recommendations")

class HybridTextRecommendationRequest(BaseModel):
    user_id: Optional[int] = Field(None, description="User id (optional)")
    text_query: Optional[str] = Field(None, description="Natural language query to bias results")
    limit: int = Field(10, ge=1, le=100, description="Number of recommendations")
    cf_weight: float = Field(0.4, ge=0.0, le=1.0, description="CF weight")
    content_weight: float = Field(0.4, ge=0.0, le=1.0, description="Content weight")
    nlp_weight: float = Field(0.2, ge=0.0, le=1.0, description="NLP/text relevance weight")

class ColdStartRequest(BaseModel):
    text_query: Optional[str] = Field(None, description="Textual preferences for cold-start users")
    favorite_genres: List[str] = Field(default_factory=list, description="List of preferred genres")
    limit: int = Field(10, ge=1, le=100, description="Number of recommendations")

class PredictionResponse(BaseModel):
    user_id: int = Field(..., description="User ID")
    anime_id: int = Field(..., description="Anime ID")
    predicted_rating: float = Field(..., description="Predicted rating (e.g., 0-10 scale)")

class SimilarUserResponse(BaseModel):
    user_id: int = Field(..., description="Similar user ID")
    similarity_score: float = Field(..., description="Similarity score (higher means more similar)")
