from typing import List

from fastapi import FastAPI, HTTPException
from recommender import get_recommendations
app = FastAPI()


@app.get("/v1")
async def root():
    return {"status": "running"}


@app.get("/v1/recommend", response_model=List[int])
def recommend(anime_id: int, limit: int = 10):
    results = get_recommendations(anime_id, limit)
    if not results:
        raise HTTPException(status_code=404, detail="Anime not found or no similar entries.")
    return results
