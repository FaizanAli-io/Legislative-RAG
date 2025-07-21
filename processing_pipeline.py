"""
Main processing pipeline for document ingestion, chunking, and embedding
"""
import asyncio
import time
import logging
from typing import List, Dict, Any
from document_loader import document_loader
from text_chunker import SemanticChunker
from embedder import embedding_service
from database import db_manager
from config import settings

logger = logging.getLogger(__name__)

class ProcessingPipeline:
    def __init__(self):
        self.chunker = SemanticChunker(
            max_chunk_size=settings.max_chunk_size,
            overlap_size=settings.chunk_overlap,
            min_chunk_size=settings.min_chunk_size
        )
    
    async def process_single_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single document through the complete pipeline"""
        doc_name = document['filename']
        start_time = time.time()
        
        logger.info(f"Starting processing pipeline for: {doc_name}")
        
        try:
            # Step 1: Chunk the document
            chunks = self.chunker.chunk_document(document['content'], doc_name)
            
            # Step 2: Create embeddings
            embedded_chunks = await embedding_service.embed_chunks(chunks)
            
            # Step 3: Store in database
            stored_count = await db_manager.insert_chunks(embedded_chunks)
            
            total_time = time.time() - start_time
            successful_embeddings = sum(1 for chunk in embedded_chunks if chunk['embedding'] is not None)
            
            return {
                'document_name': doc_name,
                'status': 'success',
                'total_chunks': len(chunks),
                'successful_embeddings': successful_embeddings,
                'stored_chunks': stored_count,
                'processing_time_seconds': total_time
            }
            
        except Exception as e:
            error_msg = f"Processing failed for {doc_name}: {str(e)}"
            logger.error(error_msg)
            
            return {
                'document_name': doc_name,
                'status': 'error',
                'error': str(e),
                'processing_time_seconds': time.time() - start_time
            }
    
    async def process_documents_from_directory(self, directory: str = None) -> Dict[str, Any]:
        """Process all documents from a directory"""
        pipeline_start = time.time()
        
        logger.info("Starting document processing pipeline")
        
        documents = await document_loader.load_documents_from_directory(directory)
        
        if not documents:
            logger.warning("No documents found to process")
            return {
                'status': 'warning',
                'message': 'No documents found',
                'processed_documents': []
            }
        
        results = []
        successful_count = 0
        total_chunks = 0
        total_embeddings = 0
        
        for i, document in enumerate(documents, 1):
            logger.info(f"Processing document {i}/{len(documents)}: {document['filename']}")
            
            result = await self.process_single_document(document)
            results.append(result)
            
            if result['status'] == 'success':
                successful_count += 1
                total_chunks += result['total_chunks']
                total_embeddings += result['successful_embeddings']
        
        total_time = time.time() - pipeline_start
        
        summary = {
            'status': 'completed',
            'total_documents': len(documents),
            'successful_documents': successful_count,
            'failed_documents': len(documents) - successful_count,
            'total_chunks_generated': total_chunks,
            'total_embeddings_created': total_embeddings,
            'total_processing_time_seconds': total_time,
            'processed_documents': results
        }
        
        logger.info("=== PROCESSING PIPELINE COMPLETED ===")
        logger.info(f"Documents processed: {successful_count}/{len(documents)}")
        logger.info(f"Total chunks generated: {total_chunks:,}")
        logger.info(f"Total embeddings created: {total_embeddings:,}")
        
        return summary

# Global processing pipeline instance
processing_pipeline = ProcessingPipeline()