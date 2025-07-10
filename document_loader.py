"""
Document loader with preprocessing for various text formats
"""
import os
import re
import logging
from typing import List, Dict, Any, Optional
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self, documents_dir: str = "documents"):
        self.documents_dir = Path(documents_dir)
        self.supported_extensions = {'.txt', '.md'}
        
    def normalize_text(self, text: str) -> str:
        """Normalize text formatting for consistent processing"""
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r' +\n', '\n', text)
        text = re.sub(r'(\d+)\.\s*\n\s*(\w)', r'\1. \2', text)
        text = re.sub(r'([a-z])\n([a-z])', r'\1 \2', text)
        return text.strip()
    
    def extract_metadata(self, text: str, filename: str) -> Dict[str, Any]:
        lines = text.split('\n')
        metadata = {
            'filename': filename,
            'character_count': len(text),
            'line_count': len(lines),
            'estimated_pages': len(text) // 2000,
        }
        for line in lines[:10]:
            line = line.strip()
            if line and len(line) > 10 and not line.startswith('#'):
                if line.isupper() or line.istitle():
                    metadata['title'] = line
                    break
        return metadata

    async def load_document(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """Load and preprocess a single document"""
        file_path = filepath if isinstance(filepath, Path) else Path(filepath)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        if file_path.suffix.lower() not in self.supported_extensions:
            logger.warning(f"Unsupported file type: {file_path}")
            return None

        try:
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            text = None
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text = f.read()
                    logger.info(f"Successfully read {file_path} with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue

            if text is None:
                logger.error(f"Could not decode file: {file_path}")
                return None

            normalized_text = self.normalize_text(text)
            metadata = self.extract_metadata(normalized_text, file_path.name)

            return {
                'filename': file_path.name,
                'filepath': str(file_path),
                'content': normalized_text,
                'metadata': metadata
            }

        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return None

    async def load_documents_from_directory(self, directory: str = None) -> List[Dict[str, Any]]:
        """Load all supported documents from directory"""
        docs_dir = self.documents_dir / directory if directory else self.documents_dir

        if not docs_dir.exists():
            logger.error(f"Documents directory not found: {docs_dir}")
            return []

        logger.info(f"Loading documents from: {docs_dir}")

        files = []
        for ext in self.supported_extensions:
            files.extend(docs_dir.glob(f"*{ext}"))

        logger.info(f"Found {len(files)} document files")

        documents = []
        for filepath in files:
            doc = await self.load_document(filepath)
            if doc:
                documents.append(doc)

        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents

# Global document loader instance
document_loader = DocumentLoader()
