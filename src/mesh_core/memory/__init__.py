"""
Memory Module - Enhanced with Sentient Memory Concepts

This module integrates Sentient's proven memory patterns into The Mesh:
- Fact extraction and atomic decomposition
- Relevance scoring and intelligent retrieval
- Knowledge graph organization and relationships
- Memory type classification and management

Following the same integration pattern used for Leon, Empathy, and AxiomEngine.
"""

# Import memory components
from .fact_extractor import (
    FactExtractor,
    FactExtractionConfig,
    ExtractedFact,
    FactExtractionResult,
    create_fact_extractor
)

from .relevance_scorer import (
    RelevanceScorer,
    RelevanceConfig,
    RelevanceScore,
    RelevanceResult,
    create_relevance_scorer
)

from .knowledge_graph import (
    KnowledgeGraph,
    KnowledgeGraphConfig,
    KnowledgeNode,
    KnowledgeRelationship,
    KnowledgeGraphResult,
    create_knowledge_graph
)

# Export all memory capabilities
__all__ = [
    # Fact Extraction
    'FactExtractor',
    'FactExtractionConfig', 
    'ExtractedFact',
    'FactExtractionResult',
    'create_fact_extractor',
    
    # Relevance Scoring
    'RelevanceScorer',
    'RelevanceConfig',
    'RelevanceScore',
    'RelevanceResult',
    'create_relevance_scorer',
    
    # Knowledge Graph
    'KnowledgeGraph',
    'KnowledgeGraphConfig',
    'KnowledgeNode',
    'KnowledgeRelationship',
    'KnowledgeGraphResult',
    'create_knowledge_graph'
]

# Version information
__version__ = "2.0.0"
__description__ = "Enhanced memory systems with Sentient concepts"
__author__ = "The Mesh Development Team"
