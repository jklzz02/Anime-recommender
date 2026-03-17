from .embeddings_loader import(
    load_id_to_index,
    load_index_to_id,
    load_user_mappings,
    load_rating_stats,
)

class Mappings:

    @property
    def id_to_index(self):
        return load_id_to_index()
    
    @property
    def index_to_id(self):
        return load_index_to_id()
    
    @property
    def user_mappings(self):
        return load_user_mappings()
    
    @property
    def ratings(self):
        return load_rating_stats()