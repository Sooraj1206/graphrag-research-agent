"""
Centralized configuration. All environment-dependent values live here so
nothing is hardcoded in business logic. Loaded once, imported everywhere.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM (Ollama — free, runs locally, no API key)
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "llama3.1"
    llm_temperature: float = 0.0

    # Embeddings (HuggingFace sentence-transformers — free, runs locally, no API key)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Vector store
    chroma_persist_dir: str = "./data/chroma"
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Agent behavior
    max_retries: int = 2
    retriever_k: int = 5


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — avoids re-parsing env on every import."""
    return Settings()
