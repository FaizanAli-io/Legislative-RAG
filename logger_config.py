"""
Comprehensive logging configuration for RAG system
"""
import logging
import logging.handlers
import json
from datetime import datetime
from typing import Dict, Any
from config import settings

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_entry["extra"] = record.extra_data
        
        return json.dumps(log_entry)

class RAGLoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter for RAG-specific logging"""
    
    def process(self, msg, kwargs):
        # Add context information to all log messages
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        kwargs['extra']['service'] = 'rag_backend'
        kwargs['extra']['version'] = '3.0.0'
        
        return msg, kwargs

def setup_logging():
    """Setup comprehensive logging configuration"""
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colored output
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # File handler with JSON formatting
    file_handler = logging.handlers.RotatingFileHandler(
        settings.log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(JSONFormatter())
    file_handler.setLevel(logging.DEBUG)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Setup specific loggers
    setup_query_logger()
    setup_performance_logger()
    
    logging.info("Logging system initialized")

def setup_query_logger():
    """Setup dedicated query logging"""
    if not settings.enable_query_logging:
        return
    
    query_logger = logging.getLogger('rag.queries')
    
    # Query-specific file handler
    query_handler = logging.handlers.RotatingFileHandler(
        'queries.log',
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    
    query_formatter = logging.Formatter(
        '%(asctime)s - QUERY - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    query_handler.setFormatter(query_formatter)
    query_logger.addHandler(query_handler)
    query_logger.setLevel(logging.INFO)

def setup_performance_logger():
    """Setup performance monitoring logger"""
    if not settings.enable_performance_logging:
        return
    
    perf_logger = logging.getLogger('rag.performance')
    
    # Performance-specific file handler
    perf_handler = logging.handlers.RotatingFileHandler(
        'performance.log',
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    
    perf_handler.setFormatter(JSONFormatter())
    perf_logger.addHandler(perf_handler)
    perf_logger.setLevel(logging.INFO)

def log_query_performance(query: str, processing_time: float, 
                         chunks_used: int, tokens_used: int):
    """Log query performance metrics"""
    perf_logger = logging.getLogger('rag.performance')
    
    perf_data = {
        "type": "query_performance",
        "query_length": len(query),
        "processing_time_seconds": processing_time,
        "chunks_used": chunks_used,
        "tokens_used": tokens_used,
        "queries_per_second": 1 / processing_time if processing_time > 0 else 0
    }
    
    perf_logger.info("Query performance", extra={'extra_data': perf_data})

def log_embedding_performance(batch_size: int, processing_time: float, 
                            success_count: int):
    """Log embedding performance metrics"""
    perf_logger = logging.getLogger('rag.performance')
    
    perf_data = {
        "type": "embedding_performance",
        "batch_size": batch_size,
        "processing_time_seconds": processing_time,
        "success_count": success_count,
        "success_rate": success_count / batch_size if batch_size > 0 else 0,
        "embeddings_per_second": success_count / processing_time if processing_time > 0 else 0
    }
    
    perf_logger.info("Embedding performance", extra={'extra_data': perf_data})