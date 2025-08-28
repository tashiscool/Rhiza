"""
LLM Integration Module for The Mesh

This module provides comprehensive integration between modern local LLMs 
(GGUF models, KoboldCpp, Ollama) and The Mesh's trust validation system.

Key Components:
- LLMTrustValidator: Social consensus validation for LLM outputs
- EnhancedKoboldClient: Advanced KoboldCpp integration with trust validation
- Model inspection and verification tools
- Apple M4 Pro optimizations

Philosophy:
Traditional ML models provide computational intelligence.
The Mesh provides social intelligence and trust validation.
Together they create trustworthy, verified AI responses.
"""

from .llm_trust_validator import (
    LLMTrustValidator,
    LLMResponse,
    TrustMetrics,
    ValidationContext
)

from .enhanced_kobold_client import (
    EnhancedKoboldClient,
    KoboldConfig,
    ModelStatus
)

__all__ = [
    'LLMTrustValidator',
    'LLMResponse', 
    'TrustMetrics',
    'ValidationContext',
    'EnhancedKoboldClient',
    'KoboldConfig',
    'ModelStatus'
]