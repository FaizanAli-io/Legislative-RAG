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
    
    #new things
    chat_model: str = "gpt-4-turbo-preview"
    log_level: str = "INFO"
    log_file: str = "rag_system.log"
    enable_query_logging: bool = True
    enable_performance_logging: bool = True
    test_database_url: str = "postgresql://user:password@localhost:5432/rag_test_db"
    test_api_key: str = "test-api-key-12345"

    class Config:
        env_file = ".env"

settings = Settings()