"""
Truth Layer - AxiomEngine Integration for Objective Fact Verification
"Truth Without Gatekeepers" - No central authority, only verifiable facts
"""

import asyncio
import sys
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

# Add AxiomEngine to path
sys.path.append('/Users/admin/AI/AxiomEngine/src')

try:
    from axiom_server.api_query import semantic_search_ledger
except ImportError:
    from .axiom_integration.mock_axiom_server import api_query
    semantic_search_ledger = api_query.semantic_search_ledger

try:
    from axiom_server.ledger import get_session
except ImportError:
    # Mock get_session function
    def get_session():
        """Mock session getter for development"""
        from .axiom_integration.axiom_processor import MockSession
        return MockSession()

try:
    from axiom_server.common import NLP_MODEL
except ImportError:
    from .axiom_integration.mock_axiom_server import common
    NLP_MODEL = common.NLP_MODEL


@dataclass
class TruthResponse:
    """Response from truth layer processing"""
    content: str
    facts: List[Dict[str, Any]]
    confidence: float
    sources: List[str]
    processing_time: float
    summary: str


class TruthLayer:
    """
    Truth Layer integrates with AxiomEngine for objective fact verification
    
    Core Principles:
    - Default to Skepticism: Network's primary state is disbelief
    - Show, Don't Tell: Every fact is traceable to sources
    - Radical Transparency: All logic is open-source
    - Resilience over Speed: Patient, long-term historian
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # AxiomEngine connection
        self.axiom_session = None
        self._initialize_axiom_connection()
        
        # Truth processing parameters
        self.min_confidence = config.get('min_confidence', 0.65)
        self.min_status = config.get('min_status', 'corroborated')
        self.max_facts = config.get('max_facts', 10)
        
        self.logger.info("Truth Layer initialized - Connected to AxiomEngine")
    
    def _initialize_axiom_connection(self):
        """Initialize connection to AxiomEngine"""
        try:
            self.axiom_session = get_session()
            self.logger.info("AxiomEngine connection established")
        except Exception as e:
            self.logger.error(f"Failed to connect to AxiomEngine: {e}")
            raise RuntimeError(f"Truth Layer requires AxiomEngine: {e}")
    
    async def process_query(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> TruthResponse:
        """
        Process query through AxiomEngine for objective truth verification
        
        Returns confidence-ranked insights, not absolute truth.
        Tagged with factual alignment scores and source traces.
        """
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Query AxiomEngine for relevant facts
            facts = await self._query_axiom_facts(query)
            
            # Analyze fact reliability and sources
            analyzed_facts = self._analyze_fact_reliability(facts)
            
            # Generate truth summary
            summary = self._generate_truth_summary(analyzed_facts, query)
            
            # Calculate overall confidence
            confidence = self._calculate_truth_confidence(analyzed_facts)
            
            # Extract source information
            sources = self._extract_sources(analyzed_facts)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return TruthResponse(
                content=summary,
                facts=[self._fact_to_dict(fact) for fact in analyzed_facts],
                confidence=confidence,
                sources=sources,
                processing_time=processing_time,
                summary=summary
            )
            
        except Exception as e:
            self.logger.error(f"Truth processing error: {e}")
            return self._create_error_response(str(e), start_time)
    
    async def _query_axiom_facts(self, query: str) -> List[Any]:
        """Query AxiomEngine for relevant facts"""
        try:
            # Use AxiomEngine's semantic search
            facts = semantic_search_ledger(
                session=self.axiom_session,
                search_term=query,
                min_status=self.min_status,
                top_n=self.max_facts,
                similarity_threshold=self.min_confidence
            )
            
            self.logger.debug(f"Retrieved {len(facts)} facts from AxiomEngine")
            return facts
            
        except Exception as e:
            self.logger.error(f"AxiomEngine query failed: {e}")
            return []
    
    def _analyze_fact_reliability(self, facts: List[Any]) -> List[Dict[str, Any]]:
        """
        Analyze reliability of facts using AxiomEngine's verification system
        
        Facts are ranked by:
        - Corroboration level (how many independent sources confirm)
        - Source credibility scores  
        - Temporal relevance
        - Contradiction detection
        """
        
        analyzed_facts = []
        
        for fact in facts:
            analysis = {
                'fact': fact,
                'reliability_score': self._calculate_reliability_score(fact),
                'corroboration_level': self._get_corroboration_level(fact),
                'source_credibility': self._get_source_credibility(fact),
                'temporal_relevance': self._calculate_temporal_relevance(fact),
                'disputed': fact.disputed if hasattr(fact, 'disputed') else False
            }
            
            analyzed_facts.append(analysis)
        
        # Sort by reliability score
        analyzed_facts.sort(key=lambda x: x['reliability_score'], reverse=True)
        
        return analyzed_facts
    
    def _calculate_reliability_score(self, fact: Any) -> float:
        """Calculate overall reliability score for a fact"""
        base_score = 0.5  # Neutral starting point
        
        # Boost for corroborated facts
        if hasattr(fact, 'status') and fact.status.value == 'corroborated':
            base_score += 0.3
        
        # Boost for empirically verified facts  
        if hasattr(fact, 'status') and fact.status.value == 'empirically_verified':
            base_score += 0.4
        
        # Penalty for disputed facts
        if hasattr(fact, 'disputed') and fact.disputed:
            base_score -= 0.5
        
        # Clamp to valid range
        return max(0.0, min(1.0, base_score))
    
    def _get_corroboration_level(self, fact: Any) -> str:
        """Get corroboration level description"""
        if hasattr(fact, 'status'):
            status = fact.status.value
            if status == 'empirically_verified':
                return 'High - Multiple independent primary sources'
            elif status == 'corroborated':
                return 'Medium - Confirmed by independent sources'  
            elif status == 'proposed':
                return 'Low - Single source, awaiting confirmation'
            else:
                return 'Very Low - Ingested but not verified'
        return 'Unknown'
    
    def _get_source_credibility(self, fact: Any) -> str:
        """Get source credibility assessment"""
        # This would integrate with AxiomEngine's source scoring system
        # For now, provide basic assessment
        if hasattr(fact, 'source') and fact.source:
            return f"Source: {fact.source[:50]}..."
        return "Source information unavailable"
    
    def _calculate_temporal_relevance(self, fact: Any) -> float:
        """Calculate how temporally relevant the fact is"""
        # Simple implementation - could be enhanced with temporal analysis
        return 1.0  # Default to fully relevant
    
    def _generate_truth_summary(
        self, 
        analyzed_facts: List[Dict[str, Any]], 
        original_query: str
    ) -> str:
        """
        Generate human-readable summary of truth findings
        
        Emphasizes confidence levels and source traceability
        """
        
        if not analyzed_facts:
            return f"No verified facts found for query: '{original_query}'"
        
        high_confidence_facts = [
            fact for fact in analyzed_facts 
            if fact['reliability_score'] >= 0.7
        ]
        
        if high_confidence_facts:
            summary = f"Based on {len(high_confidence_facts)} high-confidence fact(s):\n\n"
            
            for i, fact_analysis in enumerate(high_confidence_facts[:3], 1):
                fact = fact_analysis['fact']
                confidence = fact_analysis['reliability_score']
                
                summary += f"{i}. {fact.content}\n"
                summary += f"   Confidence: {confidence:.1%} | {fact_analysis['corroboration_level']}\n"
                summary += f"   {fact_analysis['source_credibility']}\n\n"
        
        else:
            summary = f"Limited verified information available for: '{original_query}'\n\n"
            summary += "Lower confidence findings:\n"
            
            for i, fact_analysis in enumerate(analyzed_facts[:2], 1):
                fact = fact_analysis['fact']
                confidence = fact_analysis['reliability_score']
                
                summary += f"{i}. {fact.content}\n"
                summary += f"   Confidence: {confidence:.1%} | Requires additional verification\n\n"
        
        return summary.strip()
    
    def _calculate_truth_confidence(self, analyzed_facts: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence in truth response"""
        if not analyzed_facts:
            return 0.0
        
        # Weight confidence by fact reliability
        total_confidence = sum(fact['reliability_score'] for fact in analyzed_facts)
        average_confidence = total_confidence / len(analyzed_facts)
        
        # Boost confidence if multiple high-reliability facts agree
        high_reliability_count = len([
            fact for fact in analyzed_facts 
            if fact['reliability_score'] >= 0.7
        ])
        
        if high_reliability_count >= 2:
            average_confidence = min(1.0, average_confidence * 1.2)
        
        return average_confidence
    
    def _extract_sources(self, analyzed_facts: List[Dict[str, Any]]) -> List[str]:
        """Extract source information from facts"""
        sources = []
        
        for fact_analysis in analyzed_facts:
            fact = fact_analysis['fact']
            if hasattr(fact, 'source') and fact.source:
                sources.append(fact.source)
        
        # Return unique sources
        return list(set(sources))
    
    def _fact_to_dict(self, fact_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Convert fact to dictionary for JSON serialization"""
        fact = fact_analysis['fact']
        
        return {
            'id': fact.id if hasattr(fact, 'id') else None,
            'content': fact.content if hasattr(fact, 'content') else str(fact),
            'status': fact.status.value if hasattr(fact, 'status') else 'unknown',
            'reliability_score': fact_analysis['reliability_score'],
            'corroboration_level': fact_analysis['corroboration_level'],
            'source': fact.source if hasattr(fact, 'source') else 'Unknown',
            'disputed': fact_analysis['disputed'],
            'timestamp': fact.created_at.isoformat() if hasattr(fact, 'created_at') else None
        }
    
    def _create_error_response(self, error: str, start_time: float) -> TruthResponse:
        """Create error response"""
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return TruthResponse(
            content=f"Truth verification error: {error}",
            facts=[],
            confidence=0.0,
            sources=[],
            processing_time=processing_time,
            summary=f"Unable to verify facts: {error}"
        )
    
    async def verify_fact_against_network(self, fact_claim: str) -> Dict[str, Any]:
        """
        Verify a specific fact claim against the AxiomEngine network
        
        Used for real-time fact checking during conversations
        """
        
        response = await self.process_query(fact_claim, {})
        
        verification_result = {
            'claim': fact_claim,
            'verified': response.confidence >= 0.6,
            'confidence': response.confidence,
            'supporting_facts': len(response.facts),
            'contradicting_facts': len([
                fact for fact in response.facts 
                if fact.get('disputed', False)
            ]),
            'verification_summary': response.summary
        }
        
        return verification_result
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get status of truth network connection"""
        try:
            # Test connection to AxiomEngine
            test_facts = semantic_search_ledger(
                session=self.axiom_session,
                search_term="test",
                top_n=1
            )
            
            return {
                'connected': True,
                'axiom_session_active': self.axiom_session is not None,
                'total_facts_available': len(test_facts) if test_facts else 0,
                'status': 'Operational'
            }
            
        except Exception as e:
            return {
                'connected': False,
                'error': str(e),
                'status': 'Error'
            }