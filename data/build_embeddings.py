import pandas as pd
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from logging import getLogger

logger = getLogger();

script_path = os.path.dirname(os.path.abspath(__file__))
json_dir_path = os.path.join(script_path, "json")
embeddings_dir_path = os.path.join(script_path, "embeddings")
data_path = os.path.join(script_path, "anime-dataset.csv")

embeddings_path = os.path.join(embeddings_dir_path, "anime_embeddings.npy")

def main():

    if not os.path.exists(embeddings_dir_path):
        os.makedirs(embeddings_dir_path)

    if not os.path.exists(json_dir_path):
        os.makedirs(json_dir_path)

    if not os.path.exists(data_path) and not os.path.isfile(data_path):
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
        np.save(embeddings_path, embeddings)
    except Exception as e:
        fallback_path = os.path.join(os.getcwd(), "embeddings.npy")
        np.save(fallback_path, embeddings)
        logger.error(f"An error occurred: {e}.\nEmbeddings will be saved at: {fallback_path}")

    id_to_index = {int(row["Id"]): i for i, row in anime_df.iterrows()}
    index_to_id = {i: int(row["Id"]) for i, row in anime_df.iterrows()}

    with open(os.path.join(json_dir_path, "id_to_index.json"), "w") as f:
        json.dump(id_to_index, f)

    with open(os.path.join(json_dir_path, "index_to_id.json"), "w") as f:
        json.dump(index_to_id, f)

    print("Embeddings created and cached.")

if __name__ == "__main__":
    main()