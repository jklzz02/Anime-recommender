import numpy as np
import pandas as pd
import json
import os
from sentence_transformers import SentenceTransformer
from logging import getLogger

logger = getLogger(__name__)

data_dir_path = os.path.dirname(os.path.abspath(__file__))
json_dir_path = os.path.join(data_dir_path, "json")
embeddings_dir_path = os.path.join(data_dir_path, "embeddings")
data_path = os.path.join(data_dir_path, "anime-dataset.csv")

embeddings_path = os.path.join(embeddings_dir_path, "anime_embeddings.npy")

MODEL_NAME = "BAAI/bge-base-en-v1.5"
DOC_PREFIX = "Represent this passage for retrieval: "


def clean_text(text):
    if pd.isna(text) or str(text).lower() == 'nan':
        return ''
    return str(text).strip()

def create_document(row):
    name = clean_text(row["Name"])
    synopsis = clean_text(row["Synopsis"])
    genres = clean_text(row["Genres"])
    studio = clean_text(row["Studio"])
    source = clean_text(row["Source"])
    rating = clean_text(row["Rating"])

    parts = [
        f"Title: {name}",
        f"Genres: {genres}",
        f"Studio: {studio}",
        f"Source: {source}",
        f"Rating: {rating}",
        f"Synopsis: {synopsis}",
    ]

    return " ".join(p for p in parts if p.strip())

def main():
    os.makedirs(embeddings_dir_path, exist_ok=True)
    os.makedirs(json_dir_path, exist_ok=True)

    if not os.path.exists(data_path):
        logger.error(f"Dataset not found: {data_path}")
        return

    anime_df = pd.read_csv(data_path, delimiter="\t")
    anime_df.columns = [
        "Id", "Name", "Started_airing", "Score", "Release_year",
        "Synopsis", "Episodes", "Studio", "Rating", "Type", "Source", "Genres"
    ]
    print(f"Loaded {len(anime_df):,} anime")

    model = SentenceTransformer(MODEL_NAME)
    print(f"Model: {MODEL_NAME}")
    
    documents = [
        DOC_PREFIX + create_document(row)
        for _, row in anime_df.iterrows()
    ]
    
    print(f"Encoding {len(documents):,} documents...")
    embeddings = model.encode(
        documents,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True
    )
    
    np.save(embeddings_path, embeddings)
    print(f"Saved: {embeddings_path}")

    id_to_index = {int(row["Id"]): i for i, row in anime_df.iterrows()}
    index_to_id = {i: int(row["Id"]) for i, row in anime_df.iterrows()}

    with open(os.path.join(json_dir_path, "id_to_index.json"), "w") as f:
        json.dump(id_to_index, f)
    with open(os.path.join(json_dir_path, "index_to_id.json"), "w") as f:
        json.dump(index_to_id, f)
    
    with open(os.path.join(json_dir_path, "embeddings_config.json"), "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "dimensions": model.get_sentence_embedding_dimension(),
            "total": len(anime_df),
        }, f, indent=2)

    print("Done")

if __name__ == "__main__":
    main()