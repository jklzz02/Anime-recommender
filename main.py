from http import HTTPMethod

from colorama import Fore
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from controllers import (
    collaborative_controller,
    healt_controller,
    hybrid_recommender_controller,
    recommender_controller,
)
from settings import settings

app = FastAPI(
    title="Anime Recommendation API",
    description="Hybrid recommendation system combining content-based, collaborative filtering, and NLP",
    version=settings.version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_cors_origins,
    allow_credentials=False,
    allow_methods=[HTTPMethod.GET, HTTPMethod.POST],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(healt_controller.router)
app.include_router(hybrid_recommender_controller.router)
app.include_router(recommender_controller.router)
app.include_router(collaborative_controller.router)

if __name__ == "__main__":
    grpc_server_instance = None

    if settings.enable_grpc:
        from grpc_server import start_grpc_server

        grpc_server_instance = start_grpc_server(settings.grpc_port)

    if settings.enable_rest:
        import uvicorn

        try:
            uvicorn.run(
                app,
                host=settings.host,
                port=settings.port,
                access_log=settings.environment.is_development,
                ssl_keyfile=settings.ssl_keyfile_path,
                ssl_certfile=settings.ssl_certfile_path,
                log_level="info" if settings.environment.is_production else "debug",
            )
        finally:
            if grpc_server_instance:
                grpc_server_instance.stop(grace=5)

    elif grpc_server_instance:
        print(f"{Fore.GREEN}[INFO]{Fore.RESET} REST disabled. Running gRPC only.")
        grpc_server_instance.wait_for_termination()
    else:
        print(
            f"{Fore.YELLOW}[WARNING]{Fore.RESET} Both enable_rest and enable_grpc are False. Nothing to run."
        )
