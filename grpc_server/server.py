from concurrent import futures

import grpc
from colorama import Fore

from grpc_server.pb2 import recommender_pb2_grpc
from grpc_server.pb2.health import recommender_health_pb2_grpc
from grpc_server.servicer import AnimeRecommenderServicer


def start_grpc_server(port: int) -> grpc.Server:
    """
    Create, configure and start the gRPC server.

    The server uses its own thread pool and is non-blocking — it runs
    in the background while uvicorn (if enabled) occupies the main thread.

    Returns the running grpc.Server instance so the caller can call
    wait_for_termination() or stop() on it.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    recommender_pb2_grpc.add_AnimeRecommenderServicer_to_server(
        AnimeRecommenderServicer(), server
    )
    recommender_health_pb2_grpc.add_AnimeRecommenderHealthServicer_to_server(
        AnimeRecommenderServicer(), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"{Fore.GREEN}[gRPC]{Fore.RESET} server listening on port {port}")
    return server
