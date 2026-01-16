import numpy as np
import pandas as pd
import json
import os
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from logging import getLogger

logger = getLogger(__name__)

script_path = os.path.dirname(os.path.abspath(__file__))
json_dir_path = os.path.join(script_path, "json")
embeddings_dir_path = os.path.join(script_path, "embeddings")
ratings_path = os.path.join(script_path, "anime-ratings.csv")

user_embeddings_path = os.path.join(embeddings_dir_path, "user_embeddings.npy")
anime_cf_embeddings_path = os.path.join(embeddings_dir_path, "anime_cf_embeddings.npy")
user_mapping_path = os.path.join(json_dir_path, "user_mappings.json")
rating_stats_path = os.path.join(json_dir_path, "rating_stats.json")


def build_collaborative_embeddings(
    n_factors: int = 100,
    min_ratings_per_user: int = 5,
    min_ratings_per_anime: int = 5,
    popularity_damping: float = 0.5,
) -> None:
    """
    Build collaborative filtering embeddings using tuned SVD:
    - user mean-centering
    - popularity damping
    - cosine-normalized latent factors
    """

    os.makedirs(embeddings_dir_path, exist_ok=True)
    os.makedirs(json_dir_path, exist_ok=True)

    if not os.path.exists(ratings_path):
        logger.error("Ratings dataset not found")
        return

    logger.info("Loading ratings...")
    ratings_df = pd.read_csv(ratings_path)

    logger.info(
        f"Initial dataset: {len(ratings_df)} ratings | "
        f"{ratings_df['user_id'].nunique()} users | "
        f"{ratings_df['anime_id'].nunique()} anime"
    )

    user_counts = ratings_df["user_id"].value_counts()
    anime_counts = ratings_df["anime_id"].value_counts()

    valid_users = user_counts[user_counts >= min_ratings_per_user].index
    valid_anime = anime_counts[anime_counts >= min_ratings_per_anime].index

    df = ratings_df[
        ratings_df["user_id"].isin(valid_users)
        & ratings_df["anime_id"].isin(valid_anime)
    ].copy()

    logger.info(
        f"Filtered dataset: {len(df)} ratings | "
        f"{df['user_id'].nunique()} users | "
        f"{df['anime_id'].nunique()} anime"
    )

    unique_users = sorted(df["user_id"].unique())
    unique_anime = sorted(df["anime_id"].unique())

    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    idx_to_user = {i: u for u, i in user_to_idx.items()}

    anime_to_idx = {a: i for i, a in enumerate(unique_anime)}
    idx_to_anime = {i: a for a, i in anime_to_idx.items()}

    n_users = len(unique_users)
    n_anime = len(unique_anime)

    rows = df["user_id"].map(user_to_idx).values
    cols = df["anime_id"].map(anime_to_idx).values
    values = df["score"].astype(np.float32).values

    user_item = csr_matrix(
        (values, (rows, cols)),
        shape=(n_users, n_anime),
        dtype=np.float32
    )

    logger.info("Mean-centering ratings by user...")

    user_rating_counts = user_item.getnnz(axis=1)
    user_rating_sums = np.asarray(user_item.sum(axis=1)).ravel()

    user_means = np.zeros(n_users, dtype=np.float32)
    nonzero_mask = user_rating_counts > 0
    user_means[nonzero_mask] = (
        user_rating_sums[nonzero_mask] / user_rating_counts[nonzero_mask]
    )

    dense_matrix = user_item.toarray()
    for u in range(n_users):
        mask = dense_matrix[u] != 0
        dense_matrix[u, mask] -= user_means[u]

    logger.info("Applying popularity damping...")

    anime_popularity = user_item.getnnz(axis=0)
    popularity_weights = 1.0 / np.power(anime_popularity + 1.0, popularity_damping)

    dense_matrix *= popularity_weights

    logger.info(f"Running TruncatedSVD ({n_factors} factors)...")

    svd = TruncatedSVD(
        n_components=n_factors,
        n_iter=10,
        random_state=42
    )

    user_embeddings = svd.fit_transform(dense_matrix)
    anime_embeddings = svd.components_.T

    logger.info("Normalizing embeddings...")

    def normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return x / norms

    user_embeddings = normalize(user_embeddings)
    anime_embeddings = normalize(anime_embeddings)

    np.save(user_embeddings_path, user_embeddings)
    np.save(anime_cf_embeddings_path, anime_embeddings)

    logger.info("Saved CF embeddings")

    mappings = {
        "user_to_idx": {str(k): int(v) for k, v in user_to_idx.items()},
        "idx_to_user": {str(k): int(v) for k, v in idx_to_user.items()},
        "anime_to_idx": {str(k): int(v) for k, v in anime_to_idx.items()},
        "idx_to_anime": {str(k): int(v) for k, v in idx_to_anime.items()},
        "n_users": n_users,
        "n_anime": n_anime,
        "n_factors": n_factors,
    }

    with open(user_mapping_path, "w") as f:
        json.dump(mappings, f, indent=2)

    global_mean = float(df["score"].mean())
    global_std = float(df["score"].std())

    anime_stats = (
        df.groupby("anime_id")["score"]
        .agg(["mean", "std", "count"])
        .fillna(0)
        .to_dict("index")
    )

    user_stats = {
        str(u): {
            "mean": float(user_means[user_to_idx[u]]),
            "count": int(user_counts[u]),
        }
        for u in unique_users
    }

    stats = {
        "global_mean": global_mean,
        "global_std": global_std,
        "anime_stats": {
            str(k): {
                "mean": float(v["mean"]),
                "std": float(v["std"]),
                "count": int(v["count"]),
            }
            for k, v in anime_stats.items()
        },
        "user_stats": user_stats,
    }

    with open(rating_stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\nSummary")
    print(f"- Users: {n_users}")
    print(f"- Anime: {n_anime}")
    print(f"- Ratings: {len(df)}")
    print(f"- Factors: {n_factors}")
    print(f"- Global mean rating: {global_mean:.2f}")
    print(f"- User embeddings shape: {user_embeddings.shape}")
    print(f"- Anime embeddings shape: {anime_embeddings.shape}")


if __name__ == "__main__":
    build_collaborative_embeddings(
        n_factors=100,
        min_ratings_per_user=5,
        min_ratings_per_anime=5,
        popularity_damping=0.5,
    )
