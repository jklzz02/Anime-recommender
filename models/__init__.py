from .recommender import AnimeDetail, DetailedRecommendedAnime, RecommendedAnime
from .requests import (
    CollaborativeRecommendationRequest,
    CompatibilityRequest,
    HybridRecommendationRequest,
    HybridTextRecommendationRequest,
)
from .responses import (
    CompatibilityResponse,
    DetailedRecommendationResponse,
    HybridSimilarAnimeResponse,
    PredictionResponse,
    RecommendationResponse,
    SimilarUserResponse,
)

__all__ = [
    "AnimeDetail",
    "CollaborativeRecommendationRequest",
    "CompatibilityRequest",
    "CompatibilityResponse",
    "DetailedRecommendationResponse",
    "DetailedRecommendedAnime",
    "HybridRecommendationRequest",
    "HybridSimilarAnimeResponse",
    "HybridTextRecommendationRequest",
    "PredictionResponse",
    "RecommendationResponse",
    "RecommendedAnime",
    "SimilarUserResponse",
]
