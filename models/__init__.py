
from .requests import (CollaborativeRecommendationRequest, CompatibilityRequest,
                       HybridRecommendationRequest, HybridTextRecommendationRequest)

from .responses import (DetailedRecommendationResponse, PredictionResponse,
                        RecommendationResponse, CompatibilityResponse, SimilarUserResponse,
                        HybridSimilarAnimeResponse)

from .recommender import AnimeDetail, RecommendedAnime, DetailedRecommendedAnime

__all__ = [
    "CollaborativeRecommendationRequest",
    "CompatibilityRequest",
    "HybridRecommendationRequest",
    "HybridTextRecommendationRequest",
    "DetailedRecommendationResponse",
    "PredictionResponse",
    "RecommendationResponse",
    "CompatibilityResponse",
    "SimilarUserResponse",
    "HybridSimilarAnimeResponse",
    "AnimeDetail",
    "RecommendedAnime",
    "DetailedRecommendedAnime",
]