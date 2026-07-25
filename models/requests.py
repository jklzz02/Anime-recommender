from pydantic import BaseModel, Field


class CollaborativeRecommendationRequest(BaseModel):
    user_favourite_ids: list[int] = []
    limit: int = Field(default=10, ge=1)


class CompatibilityRequest(BaseModel):
    target_anime_id: int
    user_favourite_ids: list[int]


class HybridRecommendationRequest(BaseModel):
    user_anime_list: list[int] = []
    limit: int = Field(default=10, ge=1, le=100)
    cf_weight: float = Field(default=0.5, ge=0, le=1)
    content_weight: float = Field(default=0.5, ge=0, le=1)


class HybridTextRecommendationRequest(BaseModel):
    text_query: str
    user_anime_list: list[int] = []
    limit: int = Field(default=10, ge=1, le=100)
    cf_weight: float = Field(default=0.33, ge=0, le=1)
    content_weight: float = Field(default=0.33, ge=0, le=1)
    nlp_weight: float = Field(default=0.34, ge=0, le=1)
