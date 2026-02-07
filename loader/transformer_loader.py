from logging import getLogger
from sentence_transformers import SentenceTransformer
from huggingface_hub.errors import HFValidationError
from typing import Dict

DEFAULT_MODEL: str = "BAAI/bge-base-en-v1.5"

__logger = getLogger(__name__)
__loaded_models: Dict[str, SentenceTransformer]= {}

def get_transformer(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    try:

        if model_name not in __loaded_models:
            __loaded_models[model_name] = SentenceTransformer(model_name)

        return __loaded_models[model_name]

    except HFValidationError as e:
        __logger.error(f"Failed to load model '{model_name}' no such model found: {e}")
        raise e

    except Exception as e:
        __logger.error(f"Failed to load model '{model_name}': {e}")
        raise e