"""
Document embedding service with batch processing for large documents
"""
import asyncio
import openai
from typing import List, Dict, Any
import logging
import time
from config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        self.batch_size = settings.batch_size
        self.max_retries = 3
        self.retry_delay = 1.0
    
    async def create_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text"""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise
    
    async def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=texts
                )
                return [data.embedding for data in response.data]
            
            except openai.RateLimitError as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Rate limit exceeded after {self.max_retries} attempts")
                    raise
            
            except Exception as e:
                logger.error(f"Error in batch embedding (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise
    
    async def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Embed all chunks with batch processing"""
        total_chunks = len(chunks)
        logger.info(f"Starting embedding process for {total_chunks} chunks")
        
        embedded_chunks = []
        start_time = time.time()
        
        for i in range(0, total_chunks, self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_texts = [chunk['chunk_text'] for chunk in batch]
            
            logger.info(f"Processing batch {i//self.batch_size + 1}/{(total_chunks + self.batch_size - 1)//self.batch_size}")
            
            try:
                embeddings = await self.create_embeddings_batch(batch_texts)
                
                for chunk, embedding in zip(batch, embeddings):
                    chunk_with_embedding = chunk.copy()
                    chunk_with_embedding['embedding'] = embedding
                    embedded_chunks.append(chunk_with_embedding)
                
                if i + self.batch_size < total_chunks:
                    await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to process batch starting at index {i}: {e}")
                for chunk in batch:
                    chunk_with_embedding = chunk.copy()
                    chunk_with_embedding['embedding'] = None
                    embedded_chunks.append(chunk_with_embedding)
        
        total_time = time.time() - start_time
        successful_embeddings = sum(1 for chunk in embedded_chunks if chunk['embedding'] is not None)
        
        logger.info(f"Embedding completed: {successful_embeddings}/{total_chunks} successful")
        logger.info(f"Total processing time: {total_time:.2f}s")
        
        return embedded_chunks

# Global embedding service instance
embedding_service = EmbeddingService()