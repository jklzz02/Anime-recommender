from fastapi import APIRouter, Request

from loader import get_data_status, get_loader_status
from settings import settings

router = APIRouter(prefix="/v1", tags=["Health"])


@router.get("", tags=["Health"])
def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "version": settings.version,
        "features": [
            "content-based recommendations",
            "collaborative filtering",
            "hybrid recommendations",
            "NLP text search",
            "compatibility scoring",
        ],
    }


@router.get("/health", tags=["Health"])
def health_check(request: Request):
    """Detailed health check with system status"""
    endpoints = {}

    for route in request.app.routes:
        if hasattr(route, "tags") and route.tags:
            for tag in route.tags:
                endpoints.setdefault(tag, []).append(route.path)

    loader_status = get_loader_status()
    data_status = get_data_status()

    return {
        "status": "healthy"
        if loader_status["is_loaded"] and data_status["is_healthy"]
        else "degraded",
        "version": settings.version,
        "anime_loader": loader_status,
        "datasets": data_status,
        "endpoints": endpoints,
    }
