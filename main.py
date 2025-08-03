from typing import List

from fastapi import FastAPI, HTTPException, Query
from recommender import get_recommendations, get_recommendations_by_list
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

@app.get("/v1/recommend_batch/", response_model=List[int])
def recommend_batch(anime_ids: List[int] = Query(...), limit: int = 10):

    result = get_recommendations_by_list(anime_ids, limit)
    if not result:
        raise HTTPException(status_code=404, detail="No recommendation could be made.")
    return result
