import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer

def main():
    anime_df = pd.read_csv("anime-dataset.csv", delimiter="\t")

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

    np.save("embeddings/anime_embeddings.npy", embeddings)

    id_to_index = {int(row["Id"]): i for i, row in anime_df.iterrows()}
    index_to_id = {i: int(row["Id"]) for i, row in anime_df.iterrows()}

    with open("json/id_to_index.json", "w") as f:
        json.dump(id_to_index, f)

    with open("json/index_to_id.json", "w") as f:
        json.dump(index_to_id, f)

    print("Embeddings created and cached.")


if __name__ == "__main__":
    main()