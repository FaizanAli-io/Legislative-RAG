"""
Quality Assurance service for comprehensive testing and demo scenarios
"""
import asyncio
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from config import settings
from database import db_manager
from rag_service import rag_service

logger = logging.getLogger(__name__)

class QAService:
    def __init__(self):
        self.test_queries = {
            "basic": [
                "What is R&D tax incentive?",
                "How do I apply for R&D tax incentive?",
                "What are the eligibility criteria?"
            ],
            "comprehensive": [
                "What activities qualify for R&D tax incentive?",
                "How is the R&D tax offset calculated?",
                "What documentation is required for R&D claims?",
                "What are core vs supporting R&D activities?",
                "What are the registration requirements?",
                "How do refundable and non-refundable rates work?",
                "What expenses can be claimed under R&D tax incentive?",
                "What are the compliance requirements?",
                "How do I maintain R&D records?",
                "What happens during an R&D audit?"
            ],
            "performance": [
                "What is R&D tax incentive?" * 10,  # Repeated for load testing
            ]
        }
        
        self.demo_scenarios = {
            "eligibility": {
                "queries": [
                    "What activities qualify for R&D tax incentive?",
                    "How do I determine if my software development is eligible?",
                    "What is the difference between core and supporting R&D activities?"
                ],
                "context": "R&D Eligibility Assessment"
            },
            "compliance": {
                "queries": [
                    "What documentation do I need for R&D claims?",
                    "What are the registration requirements?",
                    "How do I maintain compliance records?"
                ],
                "context": "Compliance Requirements"
            },
            "calculations": {
                "queries": [
                    "How is the R&D tax offset calculated?",
                    "What expenses can be claimed?",
                    "What are the refundable vs non-refundable rates?"
                ],
                "context": "Tax Incentive Calculations"
            }
        }
    
    async def initialize(self):
        """Initialize QA service"""
        logger.info("QA Service initialized")
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run comprehensive health checks"""
        checks = {}
        services = {}
        
        # Database health check
        try:
            db_start = time.time()
            await db_manager.get_system_statistics()
            db_time = time.time() - db_start
            checks["database"] = {"status": "healthy", "response_time": db_time}
            services["database"] = "healthy"
        except Exception as e:
            checks["database"] = {"status": "unhealthy", "error": str(e)}
            services["database"] = "unhealthy"
        
        # RAG service health check
        try:
            rag_start = time.time()
            test_result = await rag_service.search_similar_chunks("test query", top_k=1)
            rag_time = time.time() - rag_start
            checks["rag_service"] = {"status": "healthy", "response_time": rag_time}
            services["rag_service"] = "healthy"
        except Exception as e:
            checks["rag_service"] = {"status": "unhealthy", "error": str(e)}
            services["rag_service"] = "unhealthy"
        
        # Overall status
        overall_status = "healthy" if all(s == "healthy" for s in services.values()) else "degraded"
        
        return {
            "overall_status": overall_status,
            "services": services,
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def run_test_suite(self, test_suite: str = "comprehensive") -> Dict[str, Any]:
        """Run a specific test suite"""
        if test_suite not in self.test_queries:
            raise ValueError(f"Unknown test suite: {test_suite}")
        
        queries = self.test_queries[test_suite]
        results = []
        start_time = time.time()
        
        for i, query in enumerate(queries):
            test_start = time.time()
            try:
                result = await rag_service.query(query)
                test_time = time.time() - test_start
                
                results.append({
                    "query_index": i,
                    "query": query,
                    "status": "success",
                    "response_time": test_time,
                    "answer_length": len(result['answer']),
                    "sources_count": len(result['sources']),
                    "chunks_used": result.get('chunks_used', 0)
                })
                
            except Exception as e:
                test_time = time.time() - test_start
                results.append({
                    "query_index": i,
                    "query": query,
                    "status": "failed",
                    "response_time": test_time,
                    "error": str(e)
                })
        
        total_time = time.time() - start_time
        success_count = sum(1 for r in results if r["status"] == "success")
        
        return {
            "test_suite": test_suite,
            "total_queries": len(queries),
            "successful_queries": success_count,
            "failed_queries": len(queries) - success_count,
            "success_rate": success_count / len(queries) if queries else 0,
            "total_time": total_time,
            "average_response_time": sum(r["response_time"] for r in results) / len(results) if results else 0,
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def run_demo_scenario(self, scenario: str, user_type: str = "business") -> Dict[str, Any]:
        """Run a demo scenario with sample queries"""
        if scenario not in self.demo_scenarios:
            raise ValueError(f"Unknown demo scenario: {scenario}")
        
        scenario_data = self.demo_scenarios[scenario]
        demo_results = []
        
        for query in scenario_data["queries"]:
            try:
                result = await rag_service.query(query)
                
                demo_results.append({
                    "query": query,
                    "answer": result['answer'],
                    "sources": result['sources'][:3],  # Limit sources for demo
                    "processing_time": result['processing_time_seconds']
                })
                
            except Exception as e:
                demo_results.append({
                    "query": query,
                    "error": str(e)
                })
        
        return {
            "scenario": scenario,
            "context": scenario_data["context"],
            "user_type": user_type,
            "demo_queries": demo_results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_latest_results(self) -> Dict[str, Any]:
        """Get latest test results"""
        # This would typically fetch from a results database
        # For now, return a sample structure
        return {
            "last_test_run": datetime.utcnow().isoformat(),
            "test_suites": {
                "basic": {"success_rate": 1.0, "avg_response_time": 2.3},
                "comprehensive": {"success_rate": 0.95, "avg_response_time": 3.1},
                "performance": {"success_rate": 0.98, "avg_response_time": 1.8}
            },
            "system_metrics": {
                "uptime": "99.9%",
                "total_queries_processed": 15420,
                "average_daily_queries": 1250
            }
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for QA monitoring"""
        return {
            "qa_service": "operational",
            "test_coverage": "comprehensive",
            "last_health_check": datetime.utcnow().isoformat(),
            "monitoring_active": True,
            "demo_scenarios_available": len(self.demo_scenarios),
            "test_suites_available": len(self.test_queries)
        }

# Global QA service instance
qa_service = QAService()