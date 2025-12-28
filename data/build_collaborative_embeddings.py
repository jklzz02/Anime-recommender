import numpy as np
import pandas as pd
import json
import os
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from logging import getLogger

logger = getLogger()

script_path = os.path.dirname(os.path.abspath(__file__))
json_dir_path = os.path.join(script_path, "json")
embeddings_dir_path = os.path.join(script_path, "embeddings")
ratings_path = os.path.join(script_path, "anime-ratings.csv")

user_embeddings_path = os.path.join(embeddings_dir_path, "user_embeddings.npy")
anime_cf_embeddings_path = os.path.join(embeddings_dir_path, "anime_cf_embeddings.npy")
user_mapping_path = os.path.join(json_dir_path, "user_mappings.json")
rating_stats_path = os.path.join(json_dir_path, "rating_stats.json")

def build_collaborative_embeddings(n_factors=100, min_ratings_per_user=5, min_ratings_per_anime=5):
    if not os.path.exists(embeddings_dir_path):
        os.makedirs(embeddings_dir_path)

    if not os.path.exists(json_dir_path):
        os.makedirs(json_dir_path)

    if not os.path.exists(ratings_path) or not os.path.isfile(ratings_path):
        logger.error("Unable to find ratings dataset")
        return

    logger.info("Loading ratings data...")
    ratings_df = pd.read_csv(ratings_path)
    
    logger.info(f"Initial dataset: {len(ratings_df)} ratings from {ratings_df['user_id'].nunique()} users")
    
    user_counts = ratings_df['user_id'].value_counts()
    anime_counts = ratings_df['anime_id'].value_counts()
    
    valid_users = user_counts[user_counts >= min_ratings_per_user].index
    valid_anime = anime_counts[anime_counts >= min_ratings_per_anime].index
    
    filtered_df = ratings_df[
        (ratings_df['user_id'].isin(valid_users)) & 
        (ratings_df['anime_id'].isin(valid_anime))
    ].copy()
    
    logger.info(f"Filtered dataset: {len(filtered_df)} ratings from {filtered_df['user_id'].nunique()} users")
    
    unique_users = sorted(filtered_df['user_id'].unique())
    unique_anime = sorted(filtered_df['anime_id'].unique())
    
    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
    
    anime_to_idx = {anime_id: idx for idx, anime_id in enumerate(unique_anime)}
    idx_to_anime = {idx: anime_id for anime_id, idx in anime_to_idx.items()}
    
    logger.info("Building user-item matrix...")
    n_users = len(unique_users)
    n_anime = len(unique_anime)
    
    row_indices = filtered_df['user_id'].map(user_to_idx).values
    col_indices = filtered_df['anime_id'].map(anime_to_idx).values
    scores = filtered_df['score'].values
    
    user_item_matrix = csr_matrix(
        (scores, (row_indices, col_indices)),
        shape=(n_users, n_anime),
        dtype=np.float32
    )
    
    logger.info("Normalizing ratings...")
    user_means = np.array(user_item_matrix.sum(axis=1) / (user_item_matrix != 0).sum(axis=1)).flatten() # type: ignore
    user_means = np.nan_to_num(user_means)
    
    normalized_matrix = user_item_matrix.toarray()
    for i in range(n_users):
        mask = normalized_matrix[i] != 0
        normalized_matrix[i][mask] -= user_means[i]
    
    logger.info(f"Applying SVD with {n_factors} factors...")
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    user_embeddings = svd.fit_transform(normalized_matrix)
    anime_embeddings = svd.components_.T
    
    user_norms = np.linalg.norm(user_embeddings, axis=1, keepdims=True)
    user_norms[user_norms == 0] = 1
    user_embeddings = user_embeddings / user_norms
    
    anime_norms = np.linalg.norm(anime_embeddings, axis=1, keepdims=True)
    anime_norms[anime_norms == 0] = 1
    anime_embeddings = anime_embeddings / anime_norms
    
    logger.info("Saving embeddings...")
    try:
        np.save(user_embeddings_path, user_embeddings)
        np.save(anime_cf_embeddings_path, anime_embeddings)
        logger.info(f"User embeddings saved to {user_embeddings_path}")
        logger.info(f"Anime CF embeddings saved to {anime_cf_embeddings_path}")
    except Exception as e:
        logger.error(f"Error saving embeddings: {e}")
        return
    
    mappings = {
        "user_to_idx": {str(k): int(v) for k, v in user_to_idx.items()},
        "idx_to_user": {str(k): int(v) for k, v in idx_to_user.items()},
        "anime_to_idx": {str(k): int(v) for k, v in anime_to_idx.items()},
        "idx_to_anime": {str(k): int(v) for k, v in idx_to_anime.items()},
        "n_users": int(n_users),
        "n_anime": int(n_anime),
        "n_factors": int(n_factors)
    }
    
    with open(user_mapping_path, "w") as f:
        json.dump(mappings, f)
    
    logger.info("Calculating rating statistics...")
    
    global_mean = filtered_df['score'].mean()
    global_std = filtered_df['score'].std()
    
    anime_stats = filtered_df.groupby('anime_id')['score'].agg(['mean', 'std', 'count']).to_dict('index')
    
    user_stats = {
        user_id: {
            'mean': float(user_means[user_to_idx[user_id]]),
            'count': int(user_counts[user_id])
        }
        for user_id in unique_users
    }
    
    stats = {
        "global_mean": float(global_mean),
        "global_std": float(global_std),
        "anime_stats": {
            str(k): {
                'mean': float(v['mean']),
                'std': float(v['std']),
                'count': int(v['count'])
            }
            for k, v in anime_stats.items()
        },
        "user_stats": {str(k): v for k, v in user_stats.items()}
    }
    
    with open(rating_stats_path, "w") as f:
        json.dump(stats, f)
    
    logger.info("Collaborative filtering embeddings created successfully!")
    print(f"\nSummary:")
    print(f"\t- Users: {n_users}")
    print(f"\t- Anime: {n_anime}")
    print(f"\t- Ratings: {len(filtered_df)}")
    print(f"\t- Embedding dimensions: {n_factors}")
    print(f"\t- Global mean rating: {global_mean:.2f}")
    print(f"\t- User embeddings shape: {user_embeddings.shape}")
    print(f"\t- Anime CF embeddings shape: {anime_embeddings.shape}")

def main():
    build_collaborative_embeddings(
        n_factors=100,
        min_ratings_per_user=5,
        min_ratings_per_anime=5
    )

if __name__ == "__main__":
    main()