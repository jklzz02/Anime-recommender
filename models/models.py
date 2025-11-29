from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class RecommendationResponse(BaseModel):
    anime_id: int
    score: float

class DetailedRecommendationResponse(BaseModel):
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