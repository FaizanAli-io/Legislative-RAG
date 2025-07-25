"""
Advanced text chunking for very long documents with semantic awareness
"""
import re
import tiktoken
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class SemanticChunker:
    def __init__(self, max_chunk_size: int = 1000, overlap_size: int = 200, 
                 min_chunk_size: int = 100, model: str = "gpt-4"):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        self.encoding = tiktoken.encoding_for_model(model)
        
        # Section patterns for R&D legislation documents
        self.section_patterns = [
            r'^#{1,6}\s+(.+)$',
            r'^(\d+\.?\d*\.?\d*)\s+(.+)$',
            r'^([A-Z][A-Z\s]+)$',
            r'^(SECTION\s+\d+.*?)$',
            r'^(PART\s+[IVX]+.*?)$',
            r'^(CHAPTER\s+\d+.*?)$',
        ]
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.encoding.encode(text))
    
    def split_by_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs while preserving structure"""
        paragraphs = re.split(r'\n\s*\n', text)
        
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if para and len(para) > 20:
                cleaned_paragraphs.append(para)
        
        return cleaned_paragraphs
    
    def create_overlapping_chunks(self, paragraphs: List[str], 
                                section_title: str = None) -> List[Dict[str, Any]]:
        """Create overlapping chunks from paragraphs"""
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            
            if current_tokens + para_tokens > self.max_chunk_size:
                if current_chunk.strip() and current_tokens >= self.min_chunk_size:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'token_count': current_tokens,
                        'section_title': section_title
                    })
                
                current_chunk = para
                current_tokens = para_tokens
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_tokens += para_tokens
        
        if current_chunk.strip() and current_tokens >= self.min_chunk_size:
            chunks.append({
                'text': current_chunk.strip(),
                'token_count': current_tokens,
                'section_title': section_title
            })
        
        return chunks
    
    def chunk_document(self, text: str, document_name: str) -> List[Dict[str, Any]]:
        """Main chunking method for very long documents"""
        logger.info(f"Starting chunking for document: {document_name}")
        
        paragraphs = self.split_by_paragraphs(text)
        chunks = self.create_overlapping_chunks(paragraphs)
        
        all_chunks = []
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                'document_name': document_name,
                'section_title': chunk.get('section_title'),
                'chunk_index': i,
                'chunk_text': chunk['text'],
                'token_count': chunk['token_count'],
                'metadata': {
                    'has_section': chunk.get('section_title') is not None
                }
            })
        
        logger.info(f"Generated {len(all_chunks)} chunks")
        return all_chunks