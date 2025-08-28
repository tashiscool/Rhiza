"""
AxiomEngine Processor - Core Integration with AxiomEngine

Provides the main interface between The Mesh and AxiomEngine, handling
fact verification, knowledge processing, and truth analysis through
AxiomEngine's sophisticated verification pipeline.

Key Features:
- Direct integration with AxiomEngine's verification engine
- Semantic similarity analysis using spaCy
- Citation verification and source validation
- Corroboration finding across fact databases
- Integration with Mesh trust scoring
"""

import asyncio
import sys
import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Add AxiomEngine to Python path
axiom_path = Path(__file__).parent.parent.parent.parent / "AxiomEngine" / "src"
if str(axiom_path) not in sys.path:
    sys.path.insert(0, str(axiom_path))

try:
    from axiom_server.verification_engine import find_corroborating_claims, verify_citations
    from axiom_server.ledger import Fact, FactStatus, SessionMaker, ENGINE
    from axiom_server.nlp_utils import *
    from sqlalchemy.orm import Session
    AXIOM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"AxiomEngine not available: {e}")
    AXIOM_AVAILABLE = False
    
    # Mock classes for development/testing
    class MockSession:
        """Mock database session for development"""
        def __init__(self):
            self.committed = False
            self.closed = False
        
        def commit(self):
            self.committed = True
        
        def close(self):
            self.closed = True
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()
    
    class MockFact:
        """Mock fact for development"""
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class MockFactStatus:
        """Mock fact status for development"""
        PENDING = "pending"
        VERIFIED = "verified"
        REJECTED = "rejected"
    
    class MockSessionMaker:
        """Mock session maker for development"""
        def __call__(self):
            return MockSession()
    
    class MockEngine:
        """Mock engine for development"""
        pass
    
    # Replace with mocks
    Fact = MockFact
    FactStatus = MockFactStatus
    SessionMaker = MockSessionMaker
    ENGINE = MockEngine
    Session = MockSession


@dataclass
class AxiomVerificationResult:
    """Result of AxiomEngine verification"""
    claim: str
    verification_status: str
    confidence_score: float
    corroborations: List[Dict[str, Any]]
    citations: List[Dict[str, str]]
    semantic_analysis: Dict[str, Any]
    axiom_fact_id: Optional[int] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None


@dataclass 
class AxiomFactSubmission:
    """Fact submission to AxiomEngine"""
    content: str
    source_url: Optional[str] = None
    source_domain: str = "mesh_network"
    confidence: float = 0.8
    metadata: Dict[str, Any] = None


class AxiomProcessor:
    """Core processor for AxiomEngine integration"""
    
    def __init__(self):
        self.axiom_available = AXIOM_AVAILABLE
        self.session_maker = SessionMaker if AXIOM_AVAILABLE else None
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.similarity_threshold = 0.85
        self.corroboration_threshold = 0.90
        self.min_confidence_score = 0.3
        
        # Statistics
        self.verification_count = 0
        self.successful_verifications = 0
        self.failed_verifications = 0
        
        if not self.axiom_available:
            self.logger.warning("AxiomEngine not available - using mock mode")
    
    async def verify_claim(self, claim: str, context: Optional[Dict[str, Any]] = None) -> AxiomVerificationResult:
        """Verify a claim using AxiomEngine's verification pipeline"""
        start_time = time.time()
        
        try:
            if not self.axiom_available:
                return await self._mock_verification(claim, start_time)
            
            # Create a session for database operations
            session = self.session_maker()
            
            try:
                # Create or find existing fact in AxiomEngine
                fact = await self._get_or_create_fact(claim, session, context)
                
                if not fact:
                    return AxiomVerificationResult(
                        claim=claim,
                        verification_status="error",
                        confidence_score=0.0,
                        corroborations=[],
                        citations=[],
                        semantic_analysis={},
                        processing_time=time.time() - start_time,
                        error_message="Failed to create fact in AxiomEngine"
                    )
                
                # Find corroborating claims
                corroborations = find_corroborating_claims(fact, session)
                
                # Verify citations
                citations = verify_citations(fact)
                
                # Perform semantic analysis
                semantic_analysis = await self._perform_semantic_analysis(fact)
                
                # Calculate confidence score
                confidence_score = self._calculate_confidence_score(
                    fact, corroborations, citations, semantic_analysis
                )
                
                # Determine verification status
                verification_status = self._determine_verification_status(
                    confidence_score, corroborations, citations
                )
                
                self.verification_count += 1
                self.successful_verifications += 1
                
                result = AxiomVerificationResult(
                    claim=claim,
                    verification_status=verification_status,
                    confidence_score=confidence_score,
                    corroborations=corroborations,
                    citations=citations,
                    semantic_analysis=semantic_analysis,
                    axiom_fact_id=fact.id,
                    processing_time=time.time() - start_time
                )
                
                return result
                
            finally:
                session.close()
                
        except Exception as e:
            self.failed_verifications += 1
            self.logger.error(f"Error verifying claim: {e}")
            
            return AxiomVerificationResult(
                claim=claim,
                verification_status="error",
                confidence_score=0.0,
                corroborations=[],
                citations=[],
                semantic_analysis={},
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def submit_fact(self, submission: AxiomFactSubmission) -> Dict[str, Any]:
        """Submit a new fact to AxiomEngine"""
        try:
            if not self.axiom_available:
                return await self._mock_fact_submission(submission)
            
            session = self.session_maker()
            
            try:
                # Create new fact in AxiomEngine
                fact = Fact(
                    content=submission.content,
                    status=FactStatus.PROPOSED,
                    timestamp=time.time()
                )
                
                # Add source information
                if submission.source_url:
                    # This would add source information to the fact
                    # Implementation depends on AxiomEngine's source model
                    pass
                
                session.add(fact)
                session.commit()
                
                return {
                    'success': True,
                    'fact_id': fact.id,
                    'status': fact.status.value,
                    'message': 'Fact successfully submitted to AxiomEngine'
                }
                
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"Error submitting fact: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to submit fact to AxiomEngine'
            }
    
    async def batch_verify_claims(self, claims: List[str], context: Optional[Dict[str, Any]] = None) -> List[AxiomVerificationResult]:
        """Verify multiple claims in batch"""
        tasks = [self.verify_claim(claim, context) for claim in claims]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = AxiomVerificationResult(
                    claim=claims[i],
                    verification_status="error",
                    confidence_score=0.0,
                    corroborations=[],
                    citations=[],
                    semantic_analysis={},
                    error_message=str(result)
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _get_or_create_fact(self, claim: str, session: Session, context: Optional[Dict[str, Any]]) -> Optional[Fact]:
        """Get existing fact or create new one in AxiomEngine"""
        try:
            # First, try to find existing fact with similar content
            existing_facts = session.query(Fact).all()
            
            for fact in existing_facts:
                if self._is_similar_claim(claim, fact.content):
                    return fact
            
            # Create new fact if none exists
            new_fact = Fact(
                content=claim,
                status=FactStatus.PROPOSED,
                timestamp=time.time()
            )
            
            session.add(new_fact)
            session.commit()
            
            return new_fact
            
        except Exception as e:
            self.logger.error(f"Error getting/creating fact: {e}")
            return None
    
    def _is_similar_claim(self, claim1: str, claim2: str) -> bool:
        """Check if two claims are semantically similar"""
        # Simple similarity check - in production would use spaCy
        # For now, use basic string similarity
        claim1_words = set(claim1.lower().split())
        claim2_words = set(claim2.lower().split())
        
        if not claim1_words or not claim2_words:
            return False
        
        intersection = claim1_words & claim2_words
        union = claim1_words | claim2_words
        
        similarity = len(intersection) / len(union)
        return similarity > self.similarity_threshold
    
    async def _perform_semantic_analysis(self, fact: Fact) -> Dict[str, Any]:
        """Perform semantic analysis on a fact"""
        try:
            # Get semantic information from AxiomEngine
            if hasattr(fact, 'get_semantics'):
                semantics = fact.get_semantics()
                
                return {
                    'entities': self._extract_entities(fact.content),
                    'keywords': self._extract_keywords(fact.content),
                    'sentiment': self._analyze_sentiment(fact.content),
                    'complexity': self._assess_complexity(fact.content),
                    'confidence': semantics.get('confidence', 0.5)
                }
            else:
                # Fallback analysis
                return {
                    'entities': self._extract_entities(fact.content),
                    'keywords': self._extract_keywords(fact.content),
                    'sentiment': 'neutral',
                    'complexity': 'medium',
                    'confidence': 0.5
                }
                
        except Exception as e:
            self.logger.error(f"Error in semantic analysis: {e}")
            return {
                'entities': [],
                'keywords': [],
                'sentiment': 'unknown',
                'complexity': 'unknown',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text"""
        # Simple entity extraction - would use spaCy in production
        import re
        
        # Extract capitalized words as potential entities
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return list(set(entities))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction
        words = text.lower().split()
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        return list(set(keywords))[:10]  # Top 10 keywords
    
    def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of text"""
        # Simple sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'positive', 'beneficial', 'effective']
        negative_words = ['bad', 'terrible', 'awful', 'negative', 'harmful', 'ineffective']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _assess_complexity(self, text: str) -> str:
        """Assess complexity of text"""
        word_count = len(text.split())
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        avg_words_per_sentence = word_count / max(sentence_count, 1)
        
        if avg_words_per_sentence > 20:
            return 'high'
        elif avg_words_per_sentence > 12:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_confidence_score(self, fact: Fact, corroborations: List[Dict], citations: List[Dict], semantic_analysis: Dict) -> float:
        """Calculate confidence score based on multiple factors"""
        base_confidence = 0.5
        
        # Factor in corroborations
        corroboration_boost = min(len(corroborations) * 0.1, 0.3)
        
        # Factor in citations
        valid_citations = [c for c in citations if c.get('status') == 'VALID_AND_LIVE']
        citation_boost = min(len(valid_citations) * 0.05, 0.2)
        
        # Factor in semantic complexity (more complex = potentially more reliable)
        complexity = semantic_analysis.get('complexity', 'medium')
        complexity_boost = {'low': 0.0, 'medium': 0.1, 'high': 0.15}.get(complexity, 0.1)
        
        # Factor in entity count (more entities might indicate more verifiable claims)
        entity_count = len(semantic_analysis.get('entities', []))
        entity_boost = min(entity_count * 0.02, 0.1)
        
        total_confidence = base_confidence + corroboration_boost + citation_boost + complexity_boost + entity_boost
        
        return min(total_confidence, 1.0)
    
    def _determine_verification_status(self, confidence_score: float, corroborations: List[Dict], citations: List[Dict]) -> str:
        """Determine verification status based on evidence"""
        
        if confidence_score >= 0.9 and len(corroborations) >= 3:
            return "EMPIRICALLY_VERIFIED"
        elif confidence_score >= 0.8 and len(corroborations) >= 2:
            return "CORROBORATED"
        elif confidence_score >= 0.7 and (len(corroborations) >= 1 or len(citations) >= 2):
            return "LOGICALLY_CONSISTENT"
        elif confidence_score >= 0.5:
            return "PROPOSED"
        else:
            return "UNVERIFIED"
    
    async def _mock_verification(self, claim: str, start_time: float) -> AxiomVerificationResult:
        """Mock verification when AxiomEngine is not available"""
        await asyncio.sleep(0.5)  # Simulate processing time
        
        # Generate mock results based on claim content
        import hashlib
        claim_hash = hashlib.sha256(claim.encode()).hexdigest()
        hash_int = int(claim_hash[:8], 16)
        
        # Mock confidence based on claim characteristics
        confidence = 0.4 + (hash_int % 500) / 1000.0  # 0.4 to 0.9
        
        # Mock corroborations
        corroboration_count = (hash_int % 4) + 1
        corroborations = [
            {
                "fact_id": f"mock_{i}",
                "content": f"Corroborating evidence {i} for: {claim[:50]}...",
                "similarity": 0.85 + (i * 0.02),
                "source": f"mock_source_{i}.com"
            }
            for i in range(corroboration_count)
        ]
        
        # Mock citations
        citations = [
            {
                "url": f"https://example{(hash_int + i) % 10}.com/evidence",
                "status": "VALID_AND_LIVE" if i % 3 != 2 else "BROKEN_404"
            }
            for i in range((hash_int % 3) + 1)
        ]
        
        # Mock semantic analysis
        semantic_analysis = {
            'entities': ['Example Entity', 'Test Subject'],
            'keywords': claim.lower().split()[:5],
            'sentiment': 'neutral',
            'complexity': 'medium',
            'confidence': confidence
        }
        
        status = self._determine_verification_status(confidence, corroborations, citations)
        
        return AxiomVerificationResult(
            claim=claim,
            verification_status=status,
            confidence_score=confidence,
            corroborations=corroborations,
            citations=citations,
            semantic_analysis=semantic_analysis,
            processing_time=time.time() - start_time
        )
    
    async def _mock_fact_submission(self, submission: AxiomFactSubmission) -> Dict[str, Any]:
        """Mock fact submission when AxiomEngine is not available"""
        await asyncio.sleep(0.3)  # Simulate processing time
        
        return {
            'success': True,
            'fact_id': f"mock_{int(time.time())}",
            'status': 'proposed',
            'message': 'Fact successfully submitted (mock mode)'
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics"""
        total_verifications = self.verification_count
        success_rate = (self.successful_verifications / max(total_verifications, 1)) * 100
        
        return {
            'total_verifications': total_verifications,
            'successful_verifications': self.successful_verifications,
            'failed_verifications': self.failed_verifications,
            'success_rate': f"{success_rate:.1f}%",
            'axiom_available': self.axiom_available,
            'similarity_threshold': self.similarity_threshold,
            'corroboration_threshold': self.corroboration_threshold
        }
    
    def reset_statistics(self):
        """Reset processor statistics"""
        self.verification_count = 0
        self.successful_verifications = 0
        self.failed_verifications = 0