
from typing import Dict, Optional
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