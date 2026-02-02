from sentence_transformers import SentenceTransformer

__model = None

def get_transformer():
    global __model
    if __model is None:
        __model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return __model