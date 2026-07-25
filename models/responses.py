from pydantic import BaseModel, Field


class DetailedRecommendationResponse(BaseModel):
    """Simple response with score breakdown"""

    anime_id: int
    score: float
    score_breakdown: dict[str, float] | None = None


class RecommendationResponse(BaseModel):
    """Simple recommendation response with ID and score"""

    anime_id: int
    score: float


class CompatibilityResponse(BaseModel):
    """Response for compatibility scoring"""

    target_anime_id: int
    compatibility_score: float = Field(..., ge=0, le=100)
    scale: str = "1-100"


class PredictionResponse(BaseModel):
    """Response for rating prediction"""

    user_favourites: list[int]
    anime_id: int
    predicted_rating: float = Field(..., ge=1, le=10)
    scale: str = "1-10"


class SimilarUserResponse(BaseModel):
    """Response for similar users"""

    user_id: int
    similarity_score: float = Field(..., ge=0, le=1)


class HybridSimilarAnimeResponse(BaseModel):
    """Response for similar anime"""

    anime_id: int
    similarity_score: float
