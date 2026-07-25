import logging
from functools import lru_cache

import pandas as pd

from .embeddings_loader import load_anime_dataset, load_id_to_index

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class AnimeDataLoaderError(Exception):
    """Custom exception for AnimeDataLoader errors"""


class AnimeDataLoader:
    """
    anime data loader using existing embeddings and mappings.

    Loads anime data from the same CSV used to build embeddings, ensuring
    consistency between embeddings and anime details.
    """

    def __init__(self):
        """Initialize the data loader with standard paths"""

        self._anime_dict: dict[int, dict] | None = None
        self._id_to_index: dict[str, int] | None = None
        self._is_loaded = False
        self._load_error: str | None = None

    def _load_id_mappings(self):
        """Load id_to_index mapping for validation"""
        try:
            self._id_to_index = load_id_to_index()
        except Exception as e:
            logger.warning(f"Could not load id_to_index.json: {e}")
            self._id_to_index = None

    def _load_dataframe(self) -> None:
        """Load the anime dataset into a DataFrame"""

        if hasattr(self, "_df") and self._df is not None:
            return

        self._df = load_anime_dataset()

        self._df.columns = [
            "Id",
            "Name",
            "Started_airing",
            "Score",
            "Release_year",
            "Synopsis",
            "Episodes",
            "Studio",
            "Rating",
            "Type",
            "Source",
            "Genres",
        ]

    def _safe_convert(self, value, converter, default=None):
        """Safely convert a value with a fallback"""
        try:
            if pd.isna(value):
                return default
            return converter(value)
        except (ValueError, TypeError):
            return default

    def load(self) -> dict[int, dict]:
        """
        Load anime dataset into memory.

        Returns:
            Dictionary mapping anime_id to anime details

        Raises:
            AnimeDataLoaderError: If loading fails
        """
        if self._is_loaded and self._anime_dict is not None:
            return self._anime_dict

        if self._load_error:
            raise AnimeDataLoaderError(f"Cannot load data: {self._load_error}")

        try:
            logger.info("Loading anime dataset and id mappings")

            self._load_id_mappings()
            self._load_dataframe()

            self._anime_dict = {}
            skipped_count = 0
            embedding_mismatch = 0

            for idx, row in self._df.iterrows():
                try:
                    anime_id = self._safe_convert(row["Id"], int)

                    if anime_id is None:
                        skipped_count += 1
                        continue

                    if self._id_to_index and str(anime_id) not in self._id_to_index:
                        embedding_mismatch += 1

                    self._anime_dict[anime_id] = {
                        "id": anime_id,
                        "title": self._safe_convert(row["Name"], str, "Unknown"),
                        "started_airing": self._safe_convert(
                            row["Started_airing"], str
                        ),
                        "score": self._safe_convert(row["Score"], float),
                        "release_year": self._safe_convert(row["Release_year"], int),
                        "synopsis": self._safe_convert(row["Synopsis"], str),
                        "episodes": self._safe_convert(row["Episodes"], int),
                        "studio": self._safe_convert(row["Studio"], str),
                        "rating": self._safe_convert(row["Rating"], str),
                        "type": self._safe_convert(row["Type"], str),
                        "source": self._safe_convert(row["Source"], str),
                        "genres": self._safe_convert(row["Genres"], str),
                    }

                except Exception as e:
                    skipped_count += 1
                    logger.warning(f"Error processing row {idx}: {e}")
                    continue

            self._is_loaded = True

            logger.info(
                f"Successfully loaded {len(self._anime_dict)} anime entries "
                f"({skipped_count} rows skipped)"
            )

            if embedding_mismatch > 0:
                logger.warning(
                    f"{embedding_mismatch} anime IDs in dataset not found in embeddings. "
                    f"Consider rebuilding embeddings if data has changed."
                )

            return self._anime_dict

        except Exception as e:
            error_msg = f"Failed to load anime data: {e!s}"
            logger.error(error_msg, exc_info=True)
            self._load_error = error_msg
            raise AnimeDataLoaderError(error_msg) from e

    @lru_cache(maxsize=1024)
    def get_anime(self, anime_id: int) -> dict | None:
        """
        Get anime details by ID (cached).

        Args:
            anime_id: The anime ID

        Returns:
            Dictionary with anime details or None if not found
        """
        try:
            if self._anime_dict is None:
                self.load()

            anime = self._anime_dict.get(anime_id)  # type: ignore
            return anime.copy() if anime else None

        except Exception as e:
            logger.error(f"Error getting anime {anime_id}: {e}")
            return None

    def get_anime_batch(self, anime_ids: list[int]) -> list[dict]:
        """
        Get multiple anime details at once.

        Args:
            anime_ids: List of anime IDs

        Returns:
            List of anime detail dictionaries
        """
        if not isinstance(anime_ids, (list, tuple)):
            logger.error(f"Invalid anime_ids type: {type(anime_ids)}")
            return []

        try:
            if self._anime_dict is None:
                self.load()

            results = []
            for anime_id in anime_ids:
                if not isinstance(anime_id, int):
                    continue

                anime = self.get_anime(anime_id)
                if anime:
                    results.append(anime)

            return results

        except Exception as e:
            logger.error(f"Error in batch lookup: {e}")
            return []

    def enrich_recommendations(
        self, recommendations: list[tuple[int, float]], include_score: bool = True
    ) -> list[dict]:
        """
        Enrich recommendations with full anime details.

        Args:
            recommendations: List of (anime_id, score) tuples
            include_score: Whether to include recommendation score

        Returns:
            List of dicts with full anime details and optional scores
        """
        if not isinstance(recommendations, (list, tuple)):
            logger.error(f"Invalid recommendations type: {type(recommendations)}")
            return []

        try:
            if self._anime_dict is None:
                self.load()

            enriched = []

            for item in recommendations:
                try:
                    if not isinstance(item, (tuple, list)) or len(item) < 2:
                        continue

                    anime_id = int(item[0])
                    score = float(item[1])

                    anime = self.get_anime(anime_id)
                    if anime:
                        result = anime.copy()
                        if include_score:
                            result["recommendation_score"] = score
                        enriched.append(result)

                except (ValueError, TypeError, IndexError):
                    continue

            return enriched

        except Exception as e:
            logger.error(f"Error enriching recommendations: {e}")
            return []

    def enrich_with_breakdown(
        self, recommendations: list[tuple[int, float, dict]]
    ) -> list[dict]:
        """
        Enrich hybrid recommendations with full details and score breakdown.

        Args:
            recommendations: List of (anime_id, score, breakdown) tuples

        Returns:
            List of dicts with full anime details and score breakdown
        """
        if not isinstance(recommendations, (list, tuple)):
            logger.error(f"Invalid recommendations type: {type(recommendations)}")
            return []

        try:
            if self._anime_dict is None:
                self.load()

            enriched = []

            for item in recommendations:
                try:
                    if not isinstance(item, (tuple, list)) or len(item) < 3:
                        continue

                    anime_id = int(item[0])
                    score = float(item[1])
                    breakdown = item[2] if isinstance(item[2], dict) else {}

                    anime = self.get_anime(anime_id)
                    if anime:
                        result = anime.copy()
                        result["recommendation_score"] = score
                        result["score_breakdown"] = breakdown
                        enriched.append(result)

                except (ValueError, TypeError, IndexError):
                    continue

            return enriched

        except Exception as e:
            logger.error(f"Error enriching hybrid recommendations: {e}")
            return []

    def get_stats(self) -> dict:
        """Get loader statistics"""
        cache_info = self.get_anime.cache_info()
        return {
            "is_loaded": self._is_loaded,
            "anime_count": self.anime_count,
            "has_error": self._load_error is not None,
            "error_message": self._load_error,
            "cache_hits": cache_info.hits,
            "cache_misses": cache_info.misses,
            "cache_size": cache_info.currsize,
            "cache_max_size": cache_info.maxsize,
        }

    @property
    def anime_data_frame(self) -> pd.DataFrame | None:
        """Get the raw anime DataFrame (if loaded)"""
        self._load_dataframe()
        return self._df

    @property
    def is_loaded(self) -> bool:
        """Check if data is loaded"""
        return self._is_loaded

    @property
    def anime_count(self) -> int:
        """Get total count of anime in database"""
        if self._anime_dict is None:
            return 0
        return len(self._anime_dict)
