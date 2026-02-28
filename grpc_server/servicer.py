import grpc

from grpc_server.pb2 import recommender_pb2, recommender_pb2_grpc

from recommender import (
    get_most_compatible_from_favourites,
    calculate_compatibility_score,
    get_recommendations,
    get_cf_recommendations_from_favorites
)

class AnimeRecommenderServicer(recommender_pb2_grpc.AnimeRecommenderServicer):

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
