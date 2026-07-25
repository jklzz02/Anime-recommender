from .embeddings_loader import (
    load_anime_cf_embeddings,
    load_anime_compatibility_embeddings,
    load_anime_embeddings,
    load_anime_nlp_embeddings,
    load_user_embeddings,
)


class Embeddings:
    @property
    def anime_cf(self):
        return load_anime_cf_embeddings()

    @property
    def anime_content(self):
        return load_anime_embeddings()

    @property
    def anime_nlp(self):
        return load_anime_nlp_embeddings()

    @property
    def anime_compatibility(self):
        return load_anime_compatibility_embeddings()

    @property
    def user_embeddings(self):
        return load_user_embeddings()
