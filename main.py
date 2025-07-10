"""
FastAPI application for RAG backend - Milestone 2
Adds RAG query loop and API endpoints
"""
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from config import settings
from database import db_manager
from processing_pipeline import processing_pipeline
from document_loader import document_loader
from rag_service import rag_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key authentication"""
    if credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return credentials.credentials

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    similarity_threshold: Optional[float] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    processing_time_seconds: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting RAG Backend - Milestone 2")
    await db_manager.initialize()
    logger.info("Database initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Backend")
    await db_manager.close()

# Create FastAPI app
app = FastAPI(
    title="RAG Backend - R&D Tax Incentive",
    description="Document processing, embedding, and RAG query system for R&D legislation",
    version="2.0.0 - Milestone 2",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "RAG Backend - R&D Tax Incentive System",
        "version": "2.0.0",
        "milestone": "2 - RAG Query Loop",
        "status": "active",
        "features": [
            "Document loading and preprocessing",
            "Semantic text chunking",
            "OpenAI embeddings generation",
            "Vector storage in pgvector",
            "RAG query processing",
            "Context-aware responses"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "database": "connected",
            "embedding_service": "ready",
            "rag_service": "ready"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )

# Milestone 1 endpoints (document processing)
@app.post("/process/documents")
async def process_documents(
    directory: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
):
    """Process all documents from the specified directory"""
    try:
        result = await processing_pipeline.process_documents_from_directory(directory)
        return {
            "success": True,
            "message": "Document processing completed",
            "data": result
        }
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}"
        )

@app.post("/process/document")
async def process_single_document(
    filename: str,
    api_key: str = Depends(verify_api_key)
):
    """Process a single document by filename"""
    try:
        document = await document_loader.load_document(filename)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {filename}"
            )
        
        result = await processing_pipeline.process_single_document(document)
        return {
            "success": True,
            "message": f"Document {filename} processed successfully",
            "data": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Single document processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}"
        )

# Milestone 2 endpoints (RAG query)
@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """Query documents using RAG (Retrieval-Augmented Generation)"""
    try:
        result = await rag_service.query(
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        return QueryResponse(
            query=request.query,
            answer=result['answer'],
            sources=result['sources'],
            processing_time_seconds=result['processing_time_seconds']
        )
        
    except Exception as e:
        logger.error(f"RAG query failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )

@app.post("/search")
async def search_documents(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """Search documents using vector similarity (without generation)"""
    try:
        result = await rag_service.search_similar_chunks(
            query=request.query,
            top_k=request.top_k or settings.top_k_retrieval,
            similarity_threshold=request.similarity_threshold or settings.similarity_threshold
        )
        
        return {
            "success": True,
            "query": request.query,
            "results": result['chunks'],
            "total_results": len(result['chunks']),
            "processing_time_seconds": result['processing_time_seconds']
        }
        
    except Exception as e:
        logger.error(f"Document search failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@app.get("/documents/{document_name}/chunks")
async def get_document_chunks(
    document_name: str,
    api_key: str = Depends(verify_api_key)
):
    """Get all chunks for a specific document"""
    try:
        chunks = await db_manager.get_document_chunks(document_name)
        return {
            "success": True,
            "message": f"Retrieved chunks for {document_name}",
            "data": {
                "document_name": document_name,
                "total_chunks": len(chunks),
                "chunks": chunks
            }
        }
    except Exception as e:
        logger.error(f"Chunk retrieval failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chunk retrieval failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )