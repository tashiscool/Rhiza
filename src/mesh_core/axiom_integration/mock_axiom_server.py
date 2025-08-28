"""
Mock axiom server module for development/testing
"""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class MockAxiomServer:
    """Mock axiom server"""
    def __init__(self):
        self.logger = logger
        self.facts = {}
    
    async def semantic_search_ledger(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Mock semantic search"""
        self.logger.info(f"Mock semantic search: {query}")
        return [
            {
                "id": "mock_fact_001",
                "content": query,
                "confidence": 0.85,
                "source": "mock_source",
                "timestamp": "2024-01-01T00:00:00Z"
            }
        ]
    
    async def verify_claim(self, claim: str, **kwargs) -> Dict[str, Any]:
        """Mock claim verification"""
        self.logger.info(f"Mock claim verification: {claim}")
        return {
            "verified": True,
            "confidence": 0.80,
            "evidence": ["mock_evidence_1", "mock_evidence_2"],
            "sources": ["mock_source_1", "mock_source_2"]
        }

# Mock modules
api_query = MockAxiomServer()
common = type('MockCommon', (), {'NLP_MODEL': 'mock_model'})()
