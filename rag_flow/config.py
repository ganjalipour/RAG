from dataclasses import dataclass

from logger import LOG
from utils import get_conf


@dataclass
class EmbeddingConfig:
    """Configuration for embedding manager"""

    weaviate_url: str
    weaviate_api_key: str
    openai_api_key: str
    openai_model_name: str
    separator: str = "\n"
    chunk_size: int = 200
    chunk_overlap: int = 0
    collection_name: str = "general"
    cache_embed_dir: str = "./cache"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""    

    @staticmethod
    def load_from_env() -> "EmbeddingConfig":
        """Load configuration from environment variables"""
        try:
            return EmbeddingConfig(
                weaviate_url=get_conf("WEAVIATE_URL", ""),
                weaviate_api_key=get_conf("WEAVIATE_API_KEY", ""),
                openai_api_key=get_conf("OPENAI_API_KEY", ""),
                openai_model_name=get_conf("OPENAI_MODEL_NAME", "gpt-4o-mini"),
                qdrant_url="http://localhost:6333",
                collection_name="rag_embeddings",
            )
        except Exception as e:
            LOG.error(f"Error loading configuration: {e}")
            raise
