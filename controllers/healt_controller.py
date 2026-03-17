from fastapi import  APIRouter, Request
from settings import settings
from loader import get_loader_status, get_data_status

router = APIRouter(prefix="/v1", tags=["Health"])

@router.get("", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "version": settings.version,
        "features": [
            "content-based recommendations",
            "collaborative filtering",
            "hybrid recommendations",
            "NLP text search",
            "compatibility scoring"
        ]
    }


@router.get("/health", tags=["Health"])
async def health_check(request: Request):
    """Detailed health check with system status"""
    endpoints = {}

    for route in request.app.routes:
        if hasattr(route, "tags") and route.tags:
            for tag in route.tags:
                endpoints.setdefault(tag, []).append(route.path)

    loader_status = get_loader_status()
    data_status = get_data_status()

    return {
        "status": "healthy" if loader_status["is_loaded"] and data_status["is_healthy"] else "degraded",
        "version": settings.version,
        "anime_loader": loader_status,
        "datasets" : data_status,
        "endpoints": endpoints
    }