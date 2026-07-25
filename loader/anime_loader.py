import logging

from pandas import DataFrame

from .AnimeDataLoader import AnimeDataLoader, AnimeDataLoaderError

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

__anime_data_loader: AnimeDataLoader | None = None


def get_anime_data_loader() -> AnimeDataLoader:
    """Get or create the global AnimeDataLoader instance"""
    global __anime_data_loader
    if __anime_data_loader is None:
        __anime_data_loader = AnimeDataLoader()
        try:
            __anime_data_loader.load()
            logger.info(f"✓ Loaded {__anime_data_loader.anime_count} anime entries")
        except AnimeDataLoaderError as e:
            logger.error(f"✗ Failed to load anime data: {e}")
    return __anime_data_loader


def get_anime_data_frame() -> DataFrame | None:
    """Get the raw anime DataFrame (if loaded)"""
    try:
        return get_anime_data_loader().anime_data_frame
    except Exception as e:
        logger.error(f"Error in get_anime_data_frame: {e}")
        return None


def get_anime_details(anime_id: int) -> dict | None:
    """Get full anime details by ID (cached)"""
    try:
        return get_anime_data_loader().get_anime(anime_id)
    except Exception as e:
        logger.error(f"Error in get_anime_details({anime_id}): {e}")
        return None


def enrich_simple_recommendations(anime_ids: list[int]) -> list[dict]:
    """Enrich a list of anime IDs with full details"""
    try:
        return get_anime_data_loader().get_anime_batch(anime_ids)
    except Exception as e:
        logger.error(f"Error in enrich_simple_recommendations: {e}")
        return []


def enrich_recommendation_with_similarity(
    anime_tuple: list[tuple[int, float]],
) -> list[dict]:
    """Enrich a list of anime IDs with full details"""
    try:
        anime = get_anime_data_loader().get_anime_batch([aid for aid, _ in anime_tuple])
        for item in anime:
            score = next((s for aid, s in anime_tuple if aid == item["id"]), 0.0)
            item["similarity_score"] = score

        return anime
    except Exception as e:
        logger.error(f"Error in enrich_recommendation_with_similarity: {e}")
        return []


def enrich_scored_recommendations(
    recommendations: list[tuple[int, float]],
) -> list[dict]:
    """Enrich recommendations with scores: [(anime_id, score), ...]"""
    try:
        return get_anime_data_loader().enrich_recommendations(
            recommendations, include_score=True
        )
    except Exception as e:
        logger.error(f"Error in enrich_scored_recommendations: {e}")
        return []


def enrich_hybrid_recommendations(
    recommendations: list[tuple[int, float, dict]],
) -> list[dict]:
    """Enrich hybrid recommendations: [(anime_id, score, breakdown), ...]"""
    try:
        return get_anime_data_loader().enrich_with_breakdown(recommendations)
    except Exception as e:
        logger.error(f"Error in enrich_hybrid_recommendations: {e}")
        return []


def get_loader_status() -> dict:
    """Get current status of the data loader including cache stats"""
    try:
        return get_anime_data_loader().get_stats()
    except Exception as e:
        return {
            "is_loaded": False,
            "anime_count": 0,
            "has_error": True,
            "error_message": str(e),
            "data_path": "unknown",
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_size": 0,
            "cache_max_size": 0,
        }
