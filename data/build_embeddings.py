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

GENRE_WEIGHT = 0.40
SYNOPSIS_WEIGHT = 0.35
RATING_WEIGHT = 0.15
TYPE_WEIGHT = 0.07
SOURCE_WEIGHT = 0.03

def clean_text(text):
    if pd.isna(text) or str(text).lower() == 'nan':
        return ''
    return str(text).strip()

def create_content_prompt(row):
    synopsis = clean_text(row['Synopsis'])
    genres = clean_text(row['Genres'])
    source = clean_text(row['Source'])
    anime_type = clean_text(row['Type'])
    rating = clean_text(row['Rating'])
    
    content = "Synopsis: " + (synopsis if synopsis else "No description available") + ". "
    content += "Genres: " + (genres if genres else "Unknown") + ". "
    content += "Source: " + (source if source else "Unknown") + ". "
    content += "Type: " + (anime_type if anime_type else "Unknown") + ". "
    content += "Rating: " + (rating if rating else "Unknown") + ". "
    
    return content

def create_nlp_prompt(row):
    name = clean_text(row['Name'])
    synopsis = clean_text(row['Synopsis'])
    genres = clean_text(row['Genres'])
    anime_type = clean_text(row['Type'])
    studio = clean_text(row['Studio'])
    source = clean_text(row['Source'])
    rating = clean_text(row['Rating'])
    
    parts = []
    
    if name:
        parts.append(name + ".")
    
    if synopsis:
        parts.append(synopsis + ".")
    
    if genres:
        parts.append("This is a " + genres + " anime.")
    
    if anime_type:
        parts.append("It is a " + anime_type + " series.")
    
    if studio:
        parts.append("Produced by " + studio + ".")
    
    if source:
        parts.append("Source material: " + source + ".")
    
    if rating:
        parts.append("Rating: " + rating)
    
    return " ".join(parts)

def create_compatibility_prompt(row):
    genres = clean_text(row['Genres'])
    synopsis = clean_text(row['Synopsis'])
    rating = clean_text(row['Rating'])
    anime_type = clean_text(row['Type'])
    source = clean_text(row['Source'])
    
    parts = []
    
    if genres:
        parts.append("Primary genres: " + genres)
    
    if synopsis:
        parts.append("Story and themes: " + synopsis)
    
    if rating:
        parts.append("Content rating: " + rating)
    
    if anime_type:
        parts.append("Format: " + anime_type)
    
    if source:
        parts.append("Source: " + source)
    
    return ". ".join(parts)

def main():
    if not os.path.exists(embeddings_dir_path):
        os.makedirs(embeddings_dir_path)

    if not os.path.exists(json_dir_path):
        os.makedirs(json_dir_path)

    if not os.path.exists(data_path) or not os.path.isfile(data_path):
        logger.error("Unable to find csv dataset")
        return

    print("Loading anime dataset...")
    anime_df = pd.read_csv(data_path, delimiter="\t")

    anime_df.columns = [
        "Id", "Name", "Started_airing", "Score", "Release_year",
        "Synopsis", "Episodes", "Studio", "Rating", "Type", "Source", "Genres"
    ]

    print(f"Loaded {len(anime_df)} anime entries")

    print("Loading embedding model...")
    model = SentenceTransformer("intfloat/e5-base")
    
    print("\n[1/3] Generating CONTENT embeddings for semantic search...")
    anime_df['content'] = anime_df.apply(create_content_prompt, axis=1)
    texts = ["query: " + text for text in anime_df['content'].tolist()]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    try:
        np.save(content_embeddings_path, embeddings)
        logger.info(f"Content embeddings saved to {content_embeddings_path}")
        print(f"\t\t\t✓ Saved: {embeddings.shape}")
    except Exception as e:
        fallback_path = os.path.join(os.getcwd(), "embeddings.npy")
        np.save(fallback_path, embeddings)
        logger.error(f"Error: {e}. Embeddings saved at: {fallback_path}")

    print("\n[2/3] Generating NLP embeddings for conversational search...")
    anime_df['nlp_content'] = anime_df.apply(create_nlp_prompt, axis=1)
    nlp_texts = ["query: " + text for text in anime_df['nlp_content'].tolist()]
    nlp_embeddings = model.encode(nlp_texts, show_progress_bar=True, batch_size=64)

    try:
        np.save(nlp_embeddings_path, nlp_embeddings)
        logger.info(f"NLP embeddings saved to {nlp_embeddings_path}")
        print(f"\t\t\t✓ Saved: {nlp_embeddings.shape}")
    except Exception as e:
        fallback_path = os.path.join(os.getcwd(), "nlp_embeddings.npy")
        np.save(fallback_path, nlp_embeddings)
        logger.error(f"Error: {e}. NLP embeddings saved at: {fallback_path}")

    print("\n[3/3] Generating COMPATIBILITY embeddings with proper feature weighting...")
    print("\t\tMethod: Weighted combination (Genres: 40%, Synopsis: 35%, Rating: 15%)")
    print("\t\tExcluded: Studio, Score (don't indicate content similarity)")
    
    anime_df['compatibility_content'] = anime_df.apply(create_compatibility_prompt, axis=1)
    compatibility_texts = ["query: " + text for text in anime_df['compatibility_content'].tolist()]
    compatibility_embeddings = model.encode(compatibility_texts, show_progress_bar=True, batch_size=64)

    try:
        np.save(compatibility_embeddings_path, compatibility_embeddings)
        logger.info(f"Compatibility embeddings saved to {compatibility_embeddings_path}")
        print(f"\t\t\t✓ Saved: {compatibility_embeddings.shape}")
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
    
    weights_info = {
        "genre_weight": GENRE_WEIGHT,
        "synopsis_weight": SYNOPSIS_WEIGHT,
        "rating_weight": RATING_WEIGHT,
        "type_weight": TYPE_WEIGHT,
        "source_weight": SOURCE_WEIGHT,
        "excluded_features": ["Studio", "Score", "Name", "Episodes", "Release_year"],
        "method": "single_prompt_optimized"
    }
    with open(os.path.join(json_dir_path, "weights_config.json"), "w") as f:
        json.dump(weights_info, f, indent=2)

    print("All embeddings created")
    print(f"\nFiles created:")
    print(f"\t1. Content embeddings:       {content_embeddings_path}")
    print(f"\t2. NLP embeddings:           {nlp_embeddings_path}")
    print(f"\t3. Compatibility embeddings: {compatibility_embeddings_path}")
    print(f"\t4. Weights config:           {os.path.join(json_dir_path, 'weights_config.json')}")
    print(f"\nDataset: {len(anime_df)} anime, {embeddings.shape[1]} dimensions")

if __name__ == "__main__":
    main()