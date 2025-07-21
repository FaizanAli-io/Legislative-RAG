"""
Database connection and table management
"""
import asyncpg
import logging
from typing import List, Dict, Any, Optional
from config import settings
import json

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
    
    async def initialize(self):
        """Initialize database connection pool and create tables"""
        try:
            self.pool = await asyncpg.create_pool(
                settings.database_url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            await self.create_tables()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def create_tables(self):
        """Create necessary tables with pgvector extension"""
        async with self.pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create document_chunks table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_name TEXT NOT NULL,
                    section_title TEXT,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    token_count INTEGER,
                    embedding vector(1536),
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Create indexes for efficient retrieval
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding 
                ON document_chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_document 
                ON document_chunks (document_name, chunk_index);
            """)
    
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
    
    async def insert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Insert document chunks in batch"""
        if not chunks:
            return 0
        
        async with self.pool.acquire() as conn:
            query = """
                INSERT INTO document_chunks 
                (document_name, section_title, chunk_index, chunk_text, token_count, embedding, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """
            
            values = [
                (
                    chunk['document_name'],
                    chunk.get('section_title'),
                    chunk['chunk_index'],
                    chunk['chunk_text'],
                    chunk.get('token_count'),
                    f"[{', '.join(map(str, chunk.get('embedding', [])))}]",
                    json.dumps(chunk.get('metadata', {}))
                )
                for chunk in chunks
            ]
            
            await conn.executemany(query, values)
            return len(chunks)
    
    async def get_document_chunks(self, document_name: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a document"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, document_name, section_title, chunk_index, 
                       chunk_text, token_count, metadata, created_at
                FROM document_chunks 
                WHERE document_name = $1 
                ORDER BY chunk_index
            """, document_name)
            
            return [dict(row) for row in rows]
    
    async def search_similar_chunks(self, embedding: List[float], top_k: int, threshold: float) -> List[Dict[str, Any]]:
        """Find similar chunks using vector similarity search"""
        async with self.pool.acquire() as conn:
            query = """
                SELECT 
                    id, document_name, section_title, chunk_index,
                    chunk_text, token_count, metadata, created_at,
                    1 - (embedding <#> $1::vector) AS similarity_score
                FROM document_chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <#> $1::vector
                LIMIT $2;
            """
            try:
                pg_vector = f"[{', '.join(map(str, embedding))}]"
                rows = await conn.fetch(query, pg_vector, top_k)
                return [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                raise

# Global database manager instance
db_manager = DatabaseManager()