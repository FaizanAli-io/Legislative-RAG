"""
Configuration management for RAG backend
"""

import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Chat_Model
    chat_model: str = os.getenv("CHAT_MODEL", "")

    # API Security
    api_key: str = os.getenv("API_KEY", "secure-default-key")

    # Database
    database_url: str = os.getenv("DATABASE_URL", "")

    # Processing Configuration
    max_chunk_size: int = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    min_chunk_size: int = int(os.getenv("MIN_CHUNK_SIZE", "100"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    batch_size: int = int(os.getenv("BATCH_SIZE", "100"))

    # Vector Search
    top_k_retrieval: int = 50
    similarity_threshold: float = 0.7
    max_context_tokens: int = 16000
    reserved_tokens: int = 2000

    class Config:
        env_file = ".env"


settings = Settings()
