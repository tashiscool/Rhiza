"""
Relevance Scorer - Enhanced with Sentient's Memory Relevance Concepts

Integrates Sentient's proven memory relevance patterns:
- Query-fact matching and relevance scoring
- Context-aware relevance determination
- Boolean relevance classification
- Intelligent fact retrieval optimization

Following the same integration pattern used for Leon, Empathy, and AxiomEngine.
"""

import asyncio
import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import sys

# Add Sentient to path for concept extraction
try:
    sys.path.append('/Users/admin/AI/Sentient/src/server/main/memories')
    from prompts import (
        fact_relevance_system_prompt_template,
        fact_relevance_user_prompt_template
    )
    SENTIENT_MEMORY_AVAILABLE = True
except ImportError:
    SENTIENT_MEMORY_AVAILABLE = False

@dataclass
class RelevanceScore:
    """Relevance score for a fact-query pair"""
    fact_id: str
    fact_content: str
    query: str
    is_relevant: bool
    confidence: float
    relevance_reason: str
    scoring_method: str
    processing_time: float
    metadata: Dict[str, Any]

@dataclass
class RelevanceResult:
    """Result of relevance scoring process"""
    relevant_facts: List[RelevanceScore]
    irrelevant_facts: List[RelevanceScore]
    total_facts: int
    relevance_ratio: float
    processing_time: float
    metadata: Dict[str, Any]

@dataclass
class RelevanceConfig:
    """Configuration for relevance scoring"""
    enable_semantic_scoring: bool = True
    enable_keyword_scoring: bool = True
    enable_context_analysis: bool = True
    enable_confidence_scoring: bool = True
    min_confidence_threshold: float = 0.7
    max_facts_per_query: int = 100
    enable_debug_logging: bool = False
    scoring_weights: Dict[str, float] = None

class RelevanceScorer:
    """
    Enhanced relevance scorer integrating Sentient's memory relevance concepts
    
    Provides:
    - Query-fact matching and relevance scoring
    - Context-aware relevance determination
    - Boolean relevance classification
    - Intelligent fact retrieval optimization
    """
    
    def __init__(self, config: RelevanceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize scoring components
        self.semantic_scorer = None
        self.keyword_scorer = None
        self.context_analyzer = None
        
        # Performance tracking
        self.scoring_count = 0
        self.total_facts_scored = 0
        self.total_processing_time = 0.0
        
        # Default scoring weights
        if self.config.scoring_weights is None:
            self.config.scoring_weights = {
                "semantic": 0.6,
                "keyword": 0.3,
                "context": 0.1
            }
        
        # Initialize components
        self._initialize_scoring_components()
        
        self.logger.info("Relevance Scorer initialized with Sentient concepts")
    
    def _initialize_scoring_components(self):
        """Initialize relevance scoring components using Sentient patterns"""
        
        try:
            # Initialize semantic scorer
            if self.config.enable_semantic_scoring:
                self._initialize_semantic_scorer()
            
            # Initialize keyword scorer
            if self.config.enable_keyword_scoring:
                self._initialize_keyword_scorer()
            
            # Initialize context analyzer
            if self.config.enable_context_analysis:
                self._initialize_context_analyzer()
            
            self.logger.info("Relevance scoring components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize scoring components: {e}")
            self.logger.warning("Relevance scoring will use basic methods")
    
    def _initialize_semantic_scorer(self):
        """Initialize semantic scoring component"""
        
        try:
            self.semantic_scorer = SemanticScorer()
            self.logger.info("Semantic scorer initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize semantic scorer: {e}")
            self.semantic_scorer = None
    
    def _initialize_keyword_scorer(self):
        """Initialize keyword scoring component"""
        
        try:
            self.keyword_scorer = KeywordScorer()
            self.logger.info("Keyword scorer initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize keyword scorer: {e}")
            self.keyword_scorer = None
    
    def _initialize_context_analyzer(self):
        """Initialize context analysis component"""
        
        try:
            self.context_analyzer = ContextAnalyzer()
            self.logger.info("Context analyzer initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize context analyzer: {e}")
            self.context_analyzer = None
    
    async def score_relevance(self, query: str, facts: List[Dict[str, Any]], 
                             metadata: Optional[Dict[str, Any]] = None) -> RelevanceResult:
        """
        Score relevance of facts to a query using Sentient's relevance patterns
        
        Args:
            query: User query to match against facts
            facts: List of facts to score (each should have 'id' and 'content' keys)
            metadata: Additional metadata for scoring
            
        Returns:
            RelevanceResult with scored facts and relevance information
        """
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            if not facts:
                return RelevanceResult(
                    relevant_facts=[],
                    irrelevant_facts=[],
                    total_facts=0,
                    relevance_ratio=0.0,
                    processing_time=0.0,
                    metadata=metadata or {}
                )
            
            # Score each fact
            scored_facts = []
            
            for fact in facts:
                fact_id = fact.get('id', str(hash(fact.get('content', ''))))
                fact_content = fact.get('content', '')
                
                # Score relevance
                relevance_score = await self._score_single_fact(query, fact_content, fact_id)
                scored_facts.append(relevance_score)
            
            # Separate relevant and irrelevant facts
            relevant_facts = [f for f in scored_facts if f.is_relevant]
            irrelevant_facts = [f for f in scored_facts if not f.is_relevant]
            
            # Sort by confidence
            relevant_facts.sort(key=lambda x: x.confidence, reverse=True)
            irrelevant_facts.sort(key=lambda x: x.confidence, reverse=True)
            
            # Limit results
            relevant_facts = relevant_facts[:self.config.max_facts_per_query]
            
            # Calculate processing time
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Update performance tracking
            self.scoring_count += 1
            self.total_facts_scored += len(facts)
            self.total_processing_time += processing_time
            
            # Calculate relevance ratio
            relevance_ratio = len(relevant_facts) / len(facts) if facts else 0.0
            
            result = RelevanceResult(
                relevant_facts=relevant_facts,
                irrelevant_facts=irrelevant_facts,
                total_facts=len(facts),
                relevance_ratio=relevance_ratio,
                processing_time=processing_time,
                metadata={
                    "query": query,
                    "scoring_methods_used": self._get_used_scoring_methods(),
                    **(metadata or {})
                }
            )
            
            self.logger.info(f"Relevance scoring completed in {processing_time:.3f}s: {len(relevant_facts)}/{len(facts)} facts relevant")
            return result
            
        except Exception as e:
            self.logger.error(f"Relevance scoring failed: {e}")
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return RelevanceResult(
                relevant_facts=[],
                irrelevant_facts=[],
                total_facts=len(facts),
                relevance_ratio=0.0,
                processing_time=processing_time,
                metadata={
                    "error": str(e),
                    "query": query,
                    **(metadata or {})
                }
            )
    
    async def _score_single_fact(self, query: str, fact_content: str, fact_id: str) -> RelevanceScore:
        """Score relevance of a single fact to a query"""
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Initialize scoring components
            semantic_score = 0.0
            keyword_score = 0.0
            context_score = 0.0
            scoring_methods = []
            
            # Semantic scoring
            if self.semantic_scorer:
                semantic_score = await self.semantic_scorer.score(query, fact_content)
                scoring_methods.append("semantic")
            
            # Keyword scoring
            if self.keyword_scorer:
                keyword_score = await self.keyword_scorer.score(query, fact_content)
                scoring_methods.append("keyword")
            
            # Context analysis
            if self.context_analyzer:
                context_score = await self.context_analyzer.analyze(query, fact_content)
                scoring_methods.append("context")
            
            # Calculate weighted score
            weighted_score = self._calculate_weighted_score(semantic_score, keyword_score, context_score)
            
            # Determine relevance
            is_relevant = weighted_score >= self.config.min_confidence_threshold
            
            # Generate relevance reason
            relevance_reason = self._generate_relevance_reason(
                semantic_score, keyword_score, context_score, weighted_score, is_relevant
            )
            
            # Calculate processing time
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return RelevanceScore(
                fact_id=fact_id,
                fact_content=fact_content,
                query=query,
                is_relevant=is_relevant,
                confidence=weighted_score,
                relevance_reason=relevance_reason,
                scoring_method=scoring_methods[0] if scoring_methods else "basic",
                processing_time=processing_time,
                metadata={
                    "semantic_score": semantic_score,
                    "keyword_score": keyword_score,
                    "context_score": context_score,
                    "scoring_methods": scoring_methods
                }
            )
            
        except Exception as e:
            self.logger.error(f"Single fact scoring failed: {e}")
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return RelevanceScore(
                fact_id=fact_id,
                fact_content=fact_content,
                query=query,
                is_relevant=False,
                confidence=0.0,
                relevance_reason=f"Scoring failed: {str(e)}",
                scoring_method="error",
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
    def _calculate_weighted_score(self, semantic_score: float, keyword_score: float, 
                                 context_score: float) -> float:
        """Calculate weighted relevance score"""
        
        try:
            weights = self.config.scoring_weights
            
            weighted_score = (
                semantic_score * weights.get("semantic", 0.6) +
                keyword_score * weights.get("keyword", 0.3) +
                context_score * weights.get("context", 0.1)
            )
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, weighted_score))
            
        except Exception as e:
            self.logger.error(f"Weighted score calculation failed: {e}")
            return 0.0
    
    def _generate_relevance_reason(self, semantic_score: float, keyword_score: float,
                                  context_score: float, weighted_score: float, 
                                  is_relevant: bool) -> str:
        """Generate human-readable reason for relevance score"""
        
        try:
            if is_relevant:
                reasons = []
                
                if semantic_score > 0.7:
                    reasons.append("high semantic similarity")
                if keyword_score > 0.7:
                    reasons.append("strong keyword matches")
                if context_score > 0.7:
                    reasons.append("good context alignment")
                
                if reasons:
                    return f"Relevant due to: {', '.join(reasons)}"
                else:
                    return f"Relevant with {weighted_score:.2f} confidence"
            else:
                if weighted_score < 0.3:
                    return "Low overall relevance score"
                elif semantic_score < 0.3 and keyword_score < 0.3:
                    return "Poor semantic and keyword matching"
                else:
                    return f"Below threshold ({weighted_score:.2f} < {self.config.min_confidence_threshold})"
                    
        except Exception as e:
            self.logger.error(f"Relevance reason generation failed: {e}")
            return "Relevance scoring completed"
    
    def _get_used_scoring_methods(self) -> List[str]:
        """Get list of scoring methods currently in use"""
        
        methods = []
        
        if self.semantic_scorer:
            methods.append("semantic")
        if self.keyword_scorer:
            methods.append("keyword")
        if self.context_analyzer:
            methods.append("context")
        
        return methods
    
    async def get_relevant_facts(self, query: str, facts: List[Dict[str, Any]], 
                                limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get relevant facts for a query (convenience method)
        
        Args:
            query: User query
            facts: List of facts to search
            limit: Maximum number of relevant facts to return
            
        Returns:
            List of relevant facts sorted by relevance
        """
        
        try:
            result = await self.score_relevance(query, facts)
            
            # Convert to simple fact format
            relevant_facts = []
            for score in result.relevant_facts:
                fact = {
                    'id': score.fact_id,
                    'content': score.fact_content,
                    'relevance_score': score.confidence,
                    'relevance_reason': score.relevance_reason
                }
                relevant_facts.append(fact)
            
            # Apply limit if specified
            if limit is not None:
                relevant_facts = relevant_facts[:limit]
            
            return relevant_facts
            
        except Exception as e:
            self.logger.error(f"Get relevant facts failed: {e}")
            return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for relevance scoring"""
        
        avg_processing_time = (self.total_processing_time / self.scoring_count 
                             if self.scoring_count > 0 else 0)
        
        avg_facts_per_scoring = (self.total_facts_scored / self.scoring_count 
                               if self.scoring_count > 0 else 0)
        
        return {
            "scoring_count": self.scoring_count,
            "total_facts_scored": self.total_facts_scored,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "average_facts_per_scoring": avg_facts_per_scoring,
            "semantic_scorer_available": self.semantic_scorer is not None,
            "keyword_scorer_available": self.keyword_scorer is not None,
            "context_analyzer_available": self.context_analyzer is not None,
            "scoring_weights": self.config.scoring_weights
        }
    
    async def cleanup(self):
        """Clean up relevance scoring resources"""
        
        try:
            if hasattr(self.semantic_scorer, 'cleanup'):
                await self.semantic_scorer.cleanup()
            
            if hasattr(self.keyword_scorer, 'cleanup'):
                await self.keyword_scorer.cleanup()
            
            if hasattr(self.context_analyzer, 'cleanup'):
                await self.context_analyzer.cleanup()
                
            self.logger.info("Relevance scorer cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


class SemanticScorer:
    """Semantic scoring component following Sentient patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def score(self, query: str, fact_content: str) -> float:
        """Score semantic similarity between query and fact"""
        
        try:
            # Simple semantic scoring based on word overlap
            # In the future, we can integrate with embeddings or LLMs
            
            query_words = set(re.findall(r'\b\w+\b', query.lower()))
            fact_words = set(re.findall(r'\b\w+\b', fact_content.lower()))
            
            if not query_words or not fact_words:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(query_words.intersection(fact_words))
            union = len(query_words.union(fact_words))
            
            if union == 0:
                return 0.0
            
            jaccard_similarity = intersection / union
            
            # Boost score for exact phrase matches
            if query.lower() in fact_content.lower():
                jaccard_similarity = min(1.0, jaccard_similarity + 0.3)
            
            return jaccard_similarity
            
        except Exception as e:
            self.logger.error(f"Semantic scoring failed: {e}")
            return 0.0
    
    async def cleanup(self):
        """Clean up semantic scorer resources"""
        pass


class KeywordScorer:
    """Keyword scoring component following Sentient patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def score(self, query: str, fact_content: str) -> float:
        """Score keyword relevance between query and fact"""
        
        try:
            # Extract important keywords from query
            query_keywords = self._extract_keywords(query)
            fact_keywords = self._extract_keywords(fact_content)
            
            if not query_keywords:
                return 0.0
            
            # Calculate keyword match score
            matches = 0
            total_keywords = len(query_keywords)
            
            for query_keyword in query_keywords:
                if any(self._keywords_similar(query_keyword, fact_keyword) 
                      for fact_keyword in fact_keywords):
                    matches += 1
            
            keyword_score = matches / total_keywords if total_keywords > 0 else 0.0
            
            return keyword_score
            
        except Exception as e:
            self.logger.error(f"Keyword scoring failed: {e}")
            return 0.0
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        
        try:
            # Remove common stop words
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
            }
            
            words = re.findall(r'\b\w+\b', text.lower())
            keywords = [word for word in words if word not in stop_words and len(word) > 2]
            
            return keywords
            
        except Exception as e:
            self.logger.error(f"Keyword extraction failed: {e}")
            return []
    
    def _keywords_similar(self, keyword1: str, keyword2: str) -> bool:
        """Check if two keywords are similar"""
        
        try:
            # Exact match
            if keyword1 == keyword2:
                return True
            
            # Stemming (simple approach)
            if keyword1.endswith('ing') and keyword2 == keyword1[:-3]:
                return True
            if keyword2.endswith('ing') and keyword1 == keyword2[:-3]:
                return True
            
            # Plural forms
            if keyword1.endswith('s') and keyword2 == keyword1[:-1]:
                return True
            if keyword2.endswith('s') and keyword1 == keyword2[:-1]:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Keyword similarity check failed: {e}")
            return False
    
    async def cleanup(self):
        """Clean up keyword scorer resources"""
        pass


class ContextAnalyzer:
    """Context analysis component following Sentient patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def analyze(self, query: str, fact_content: str) -> float:
        """Analyze context relevance between query and fact"""
        
        try:
            # Simple context analysis based on domain matching
            # In the future, we can integrate more sophisticated context analysis
            
            query_context = self._extract_context(query)
            fact_context = self._extract_context(fact_content)
            
            if not query_context or not fact_context:
                return 0.5  # Neutral score if no context detected
            
            # Check for context overlap
            context_overlap = len(query_context.intersection(fact_context))
            total_context = len(query_context.union(fact_context))
            
            if total_context == 0:
                return 0.5
            
            context_score = context_overlap / total_context
            
            return context_score
            
        except Exception as e:
            self.logger.error(f"Context analysis failed: {e}")
            return 0.5
    
    def _extract_context(self, text: str) -> set:
        """Extract context indicators from text"""
        
        try:
            context_indicators = set()
            text_lower = text.lower()
            
            # Time context
            time_words = ['today', 'tomorrow', 'yesterday', 'week', 'month', 'year', 'morning', 'afternoon', 'evening']
            for word in time_words:
                if word in text_lower:
                    context_indicators.add('time')
            
            # Location context
            location_words = ['home', 'work', 'office', 'meeting', 'room', 'building', 'city', 'country']
            for word in location_words:
                if word in text_lower:
                    context_indicators.add('location')
            
            # Action context
            action_words = ['meeting', 'call', 'email', 'task', 'project', 'deadline', 'reminder']
            for word in action_words:
                if word in text_lower:
                    context_indicators.add('action')
            
            # Person context
            person_words = ['manager', 'colleague', 'friend', 'family', 'team', 'client']
            for word in person_words:
                if word in text_lower:
                    context_indicators.add('person')
            
            return context_indicators
            
        except Exception as e:
            self.logger.error(f"Context extraction failed: {e}")
            return set()
    
    async def cleanup(self):
        """Clean up context analyzer resources"""
        pass


# Factory function for easy integration
def create_relevance_scorer(config: Optional[RelevanceConfig] = None) -> RelevanceScorer:
    """Create a relevance scorer with default or custom configuration"""
    
    if config is None:
        config = RelevanceConfig()
    
    return RelevanceScorer(config)
