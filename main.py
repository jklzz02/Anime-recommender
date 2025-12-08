from fastapi import FastAPI
from settings import settings

from controllers import (
    healt_controller,
    recommender_controller,
    hybrid_recommender_controller,
    collaborative_controller
)
app = FastAPI(
    title="Anime Recommendation API",
    description="Hybrid recommendation system combining content-based, collaborative filtering, and NLP",
    version="1.0.0"
)

app.include_router(healt_controller.router)
app.include_router(hybrid_recommender_controller.router)
app.include_router(recommender_controller.router)
app.include_router(collaborative_controller.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,
                host=settings.host,
                port=settings.port,
                access_log=settings.environment.is_development,
                log_level="info" if settings.environment.is_production else "debug")