import grpc

from grpc_server.pb2 import recommender_pb2, recommender_pb2_grpc
from grpc_server.pb2.health import recommender_health_pb2, recommender_health_pb2_grpc

from loader import (
    get_loader_status,
    get_data_status,
)

from recommender import (
    get_most_compatible_from_favourites,
    calculate_compatibility_score,
    get_recommendations,
    get_cf_recommendations_from_favorites
)

class AnimeRecommenderServicer(
    recommender_pb2_grpc.AnimeRecommenderServicer,
    recommender_health_pb2_grpc.AnimeRecommenderHealthServicer):

    def HealthCheck(self, request, context):
        try:
            loader_status = get_loader_status()
            data_status = get_data_status()

            anime_loader_status = recommender_health_pb2.AnimeLoaderStatus(
                is_loaded=loader_status["is_loaded"],
                has_error=loader_status["has_error"],
                anime_count=loader_status["anime_count"],
                error_message=loader_status["error_message"],
                cache_hits=loader_status["cache_hits"],
                cache_misses=loader_status["cache_misses"],
                cache_size=loader_status["cache_size"],
                cache_max_size=loader_status["cache_max_size"]
            )

            statuses = [
                recommender_health_pb2.DataHealth(
                    file=item["file"],
                    status=item["status"]
                )
                for item in data_status["set_status"]
            ]

            data_info = recommender_health_pb2.DataSetStatus(
                is_healthy=data_status["is_healthy"],
                set_status=statuses 
            )

            return recommender_health_pb2.HealthCheckResponse(
                anime_loader_status=anime_loader_status,
                data_sets_status=data_info,
                version=1,
                service_status="available"
            )

        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, str(e))
            return recommender_health_pb2.HealthCheckResponse()

    def GetCompatible(self, request, context):
        try:
            results = get_most_compatible_from_favourites(
                user_anime_ids=list(request.user_favourite_ids),
                limit=request.limit if request.limit > 0 else 10,
            )
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, str(e))
            return recommender_pb2.CompatibleResponse()

        if not results:
            context.abort(grpc.StatusCode.NOT_FOUND, "No compatible anime found.")
            return recommender_pb2.CompatibleResponse()

        data = [
            recommender_pb2.CompatibleAnime(
                anime_id=anime_id, compatibility_score=score
            )
            for anime_id, score in results
        ]
        return recommender_pb2.CompatibleResponse(data=data, scale="1-100")
    
    def GetCfRecommendations(self, request, context):
        try:
            result = get_cf_recommendations_from_favorites(
                user_anime_ids=request.user_favourite_ids,
                limit=request.limit if request.limit > 0 else 10
                )
            
            if not result:
                context.abort(grpc.StatusCode.NOT_FOUND, "No anime found to recommend.")
                return recommender_pb2.CollaborativeRecommendationResponse()
            
            return recommender_pb2.CollaborativeRecommendationResponse(
                recommended=[anime_id for anime_id, _ in result]
            )
            
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, str(e))
            return recommender_pb2.CollaborativeRecommendationResponse()

    def GetCompatibility(self, request, context):
        try:
            score = calculate_compatibility_score(
                target_anime_id=request.target_anime_id,
                user_anime_ids=list(request.user_favourite_ids),
            )
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, str(e))
            return recommender_pb2.CompatibilityResponse()

        return recommender_pb2.CompatibilityResponse(
            target_anime_id=request.target_anime_id,
            compatibility_score=score,
            scale="1-100",
        )

    def GetCompatibilityScores(self, request, context):
        scores = []
        try:
            for target_id in request.target_anime_ids:
                score = calculate_compatibility_score(
                    target_anime_id=target_id,
                    user_anime_ids=list(request.user_favourite_ids),
                )
                scores.append(
                    recommender_pb2.CompatibilityResponse(
                        target_anime_id=target_id,
                        compatibility_score=score,
                        scale="1-100",
                    )
                )
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, str(e))
            return recommender_pb2.CompatibilityBatchResponse()

        return recommender_pb2.CompatibilityBatchResponse(scores=scores)

    def GetRelated(self, request, context):
        try:
            results = get_recommendations(
                anime_id=request.anime_id,
                limit=request.limit if request.limit > 0 else 10,
            )
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, str(e))
            return recommender_pb2.RelatedResponse()

        if not results:
            context.abort(grpc.StatusCode.NOT_FOUND, "Anime not found or no similar entries.")
            return recommender_pb2.RelatedResponse()

        anime_ids = [aid for aid, _ in results]
        return recommender_pb2.RelatedResponse(anime_ids=anime_ids)
