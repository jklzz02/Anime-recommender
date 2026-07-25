from pydantic import BaseModel, Field


class AnimeDetail(BaseModel):
    """Full anime details"""

    id: int = Field(..., description="Anime ID")
    title: str = Field(..., description="Anime title")
    score: float | None = Field(None, description="Average user rating", ge=0, le=10)
    release_year: int | None = Field(None, description="Year of release")
    synopsis: str | None = Field(None, description="Anime synopsis/description")
    episodes: int | None = Field(None, description="Number of episodes")
    studio: str | None = Field(None, description="Production studio")
    rating: str | None = Field(None, description="Age rating (e.g., PG-13, R)")
    type: str | None = Field(None, description="Type (TV, Movie, OVA, etc.)")
    source: str | None = Field(
        None, description="Source material (Manga, Original, etc.)"
    )
    genres: str | None = Field(None, description="Comma-separated genre list")
    started_airing: str | None = Field(None, description="Start date of airing")


class RecommendedAnime(AnimeDetail):
    """Anime details with recommendation score"""

    recommendation_score: float = Field(
        ...,
        description="Recommendation strength score (scale varies by endpoint)",
        ge=0,
    )


class DetailedRecommendedAnime(RecommendedAnime):
    """Anime details with score breakdown"""

    score_breakdown: dict[str, float] = Field(
        ...,
        description="Breakdown showing contribution from each recommendation method",
    )
