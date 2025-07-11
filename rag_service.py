"""
RAG (Retrieval-Augmented Generation) service for query processing
"""
import asyncio
import openai
import time
import logging
from typing import List, Dict, Any, Optional
from config import settings
from database import db_manager
from embedder import embedding_service

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        self.chat_model = settings.chat_model
        self.max_context_tokens = settings.max_context_tokens
        self.reserved_tokens = settings.reserved_tokens
    
    async def search_similar_chunks(self, query: str, top_k: int = None, 
                                  similarity_threshold: float = None) -> Dict[str, Any]:
        """Search for similar chunks using vector similarity"""
        start_time = time.time()
        
        # Use defaults if not provided
        if top_k is None:
            top_k = settings.top_k_retrieval
        if similarity_threshold is None:
            similarity_threshold = settings.similarity_threshold
        
        try:
            # Create query embedding
            query_embedding = await embedding_service.create_embedding(query)
            
            # Search similar chunks
            chunks = await db_manager.search_similar_chunks(query_embedding, top_k)
            
            # Filter by similarity threshold
            filtered_chunks = [
                chunk for chunk in chunks 
                if chunk['similarity_score'] >= similarity_threshold
            ]
            
            processing_time = time.time() - start_time
            
            logger.info(f"Found {len(filtered_chunks)} relevant chunks for query")
            
            return {
                'chunks': filtered_chunks,
                'total_found': len(chunks),
                'filtered_count': len(filtered_chunks),
                'processing_time_seconds': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise
    
    def build_context(self, chunks: List[Dict[str, Any]], max_tokens: int) -> str:
        """Build context string from chunks within token limit"""
        context_parts = []
        current_tokens = 0
        
        for chunk in chunks:
            chunk_text = f"Document: {chunk['document_name']}\n"
            if chunk['section_title']:
                chunk_text += f"Section: {chunk['section_title']}\n"
            chunk_text += f"Content: {chunk['chunk_text']}\n\n"
            
            chunk_tokens = chunk.get('token_count', len(chunk_text.split()) * 1.3)
            
            if current_tokens + chunk_tokens <= max_tokens:
                context_parts.append(chunk_text)
                current_tokens += chunk_tokens
            else:
                break
        
        return "".join(context_parts)
    
    async def generate_response(self, query: str, context: str) -> str:
        """Generate response using OpenAI chat completion"""
        system_prompt = """You are an expert assistant for R&D Tax Incentive legislation and compliance. 
        Use the provided context to answer questions accurately and comprehensively.
        
        Guidelines:
        - Base your answers on the provided context
        - If the context doesn't contain enough information, say so
        - Provide specific references to legislation sections when available
        - Be precise about eligibility criteria and requirements
        - Use clear, professional language suitable for business users
        """
        
        user_prompt = f"""Context from R&D Tax Incentive documents:
        {context}
        
        Question: {query}
        
        Please provide a comprehensive answer based on the context above."""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    async def query(self, query: str, top_k: int = None, 
                   similarity_threshold: float = None) -> Dict[str, Any]:
        """Main RAG query method"""
        start_time = time.time()
        
        logger.info(f"Processing RAG query: {query[:100]}...")
        
        try:
            # Step 1: Search for relevant chunks
            search_result = await self.search_similar_chunks(
                query, top_k, similarity_threshold
            )
            
            relevant_chunks = search_result['chunks']
            
            if not relevant_chunks:
                return {
                    'answer': "I couldn't find relevant information in the R&D Tax Incentive documents to answer your question. Please try rephrasing your query or ask about specific aspects of R&D eligibility, compliance, or legislation.",
                    'sources': [],
                    'processing_time_seconds': time.time() - start_time
                }
            
            # Step 2: Build context within token limits
            available_tokens = self.max_context_tokens - self.reserved_tokens
            context = self.build_context(relevant_chunks, available_tokens)
            
            # Step 3: Generate response
            answer = await self.generate_response(query, context)
            
            # Step 4: Prepare sources
            sources = []
            for chunk in relevant_chunks[:10]:  # Limit sources to top 10
                sources.append({
                    'document_name': chunk['document_name'],
                    'section_title': chunk.get('section_title'),
                    'chunk_index': chunk['chunk_index'],
                    'similarity_score': round(chunk['similarity_score'], 3),
                    'content_preview': chunk['chunk_text'][:200] + "..." if len(chunk['chunk_text']) > 200 else chunk['chunk_text']
                })
            
            processing_time = time.time() - start_time
            
            logger.info(f"RAG query completed in {processing_time:.2f}s")
            logger.info(f"Used {len(relevant_chunks)} chunks, generated {len(answer)} character response")
            
            return {
                'answer': answer,
                'sources': sources,
                'processing_time_seconds': processing_time,
                'chunks_used': len(relevant_chunks),
                'context_tokens_used': len(context.split()) * 1.3  # Rough estimate
            }
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            raise

# Global RAG service instance
rag_service = RAGService()