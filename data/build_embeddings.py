import numpy as np
import pandas as pd
import json
import os
from sentence_transformers import SentenceTransformer
from logging import getLogger

logger = getLogger()

data_dir_path = os.path.dirname(os.path.abspath(__file__))
json_dir_path = os.path.join(data_dir_path, "json")
embeddings_dir_path = os.path.join(data_dir_path, "embeddings")
data_path = os.path.join(data_dir_path, "anime-dataset.csv")

content_embeddings_path = os.path.join(embeddings_dir_path, "anime_embeddings.npy")
nlp_embeddings_path = os.path.join(embeddings_dir_path, "anime_nlp_embeddings.npy")
compatibility_embeddings_path = os.path.join(embeddings_dir_path, "anime_compatibility_embeddings.npy")

def main():
    if not os.path.exists(embeddings_dir_path):
        os.makedirs(embeddings_dir_path)

    if not os.path.exists(json_dir_path):
        os.makedirs(json_dir_path)

    if not os.path.exists(data_path) or not os.path.isfile(data_path):
        logger.error("Unable to find csv dataset")
        return

    anime_df = pd.read_csv(data_path, delimiter="\t")

    anime_df.columns = [
        "Id", "Name", "Started_airing", "Score", "Release_year",
        "Synopsis", "Episodes", "Studio", "Rating", "Type", "Source", "Genres"
    ]

    anime_df['content'] = (
        "Title: " + anime_df['Name'].astype(str) + ". " +
        "Synopsis: " + anime_df['Synopsis'].astype(str) + ". " +
        "Genres: " + anime_df['Genres'].astype(str) + ". " +
        "Studio: " + anime_df['Studio'].astype(str) + ". " +
        "Source: " + anime_df['Source'].astype(str) + ". " +
        "Type: " + anime_df['Type'].astype(str) + ". " +
        "Rating: " + anime_df['Rating'].astype(str) + ". " +
        "Release Year: " + anime_df['Release_year'].astype(str)
    )

    model = SentenceTransformer("intfloat/e5-base")
    texts = ["query: " + text for text in anime_df['content'].tolist()]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    try:
        np.save(content_embeddings_path, embeddings)
        logger.info(f"Content embeddings saved to {content_embeddings_path}")
    except Exception as e:
        fallback_path = os.path.join(os.getcwd(), "embeddings.npy")
        np.save(fallback_path, embeddings)
        logger.error(f"Error: {e}. Embeddings saved at: {fallback_path}")

    anime_df['nlp_content'] = (
        anime_df['Name'].astype(str) + ". " +
        anime_df['Synopsis'].astype(str) + ". " +
        "This is a " + anime_df['Genres'].astype(str) + " anime. " +
        "It is a " + anime_df['Type'].astype(str) + " series. " +
        "Produced by " + anime_df['Studio'].astype(str) + ". " +
        "Source material: " + anime_df['Source'].astype(str) + ". " +
        "Rating: " + anime_df['Rating'].astype(str)
    )

    nlp_texts = ["query: " + text for text in anime_df['nlp_content'].tolist()]
    nlp_embeddings = model.encode(nlp_texts, show_progress_bar=True, batch_size=64)

    try:
        np.save(nlp_embeddings_path, nlp_embeddings)
        logger.info(f"NLP embeddings saved to {nlp_embeddings_path}")
    except Exception as e:
        fallback_path = os.path.join(os.getcwd(), "nlp_embeddings.npy")
        np.save(fallback_path, nlp_embeddings)
        logger.error(f"Error: {e}. NLP embeddings saved at: {fallback_path}")

    anime_df['compatibility_content'] = (
        "Genres: " + anime_df['Genres'].astype(str) + ". " +
        "Rating: " + anime_df['Rating'].astype(str) + ". " +
        "Type: " + anime_df['Type'].astype(str) + ". " +
        "Studio: " + anime_df['Studio'].astype(str) + ". " +
        "Source: " + anime_df['Source'].astype(str) + ". " +
        "Score: " + anime_df['Score'].astype(str) + ". " +
        "Theme: " + anime_df['Synopsis'].astype(str)
    )

    compatibility_texts = ["query: " + text for text in anime_df['compatibility_content'].tolist()]
    compatibility_embeddings = model.encode(compatibility_texts, show_progress_bar=True, batch_size=64)

    try:
        np.save(compatibility_embeddings_path, compatibility_embeddings)
        logger.info(f"Compatibility embeddings saved to {compatibility_embeddings_path}")
    except Exception as e:
        fallback_path = os.path.join(os.getcwd(), "compatibility_embeddings.npy")
        np.save(fallback_path, compatibility_embeddings)
        logger.error(f"Error: {e}. Compatibility embeddings saved at: {fallback_path}")

    id_to_index = {int(row["Id"]): i for i, row in anime_df.iterrows()}
    index_to_id = {i: int(row["Id"]) for i, row in anime_df.iterrows()}

    with open(os.path.join(json_dir_path, "id_to_index.json"), "w") as f:
        json.dump(id_to_index, f)

    with open(os.path.join(json_dir_path, "index_to_id.json"), "w") as f:
        json.dump(index_to_id, f)

    print("All embeddings created and cached successfully!")
    print(f"  - Content embeddings: {content_embeddings_path}")
    print(f"  - NLP embeddings: {nlp_embeddings_path}")
    print(f"  - Compatibility embeddings: {compatibility_embeddings_path}")

if __name__ == "__main__":
    main()