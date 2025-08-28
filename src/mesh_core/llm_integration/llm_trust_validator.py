"""
LLM Trust Validator - Integrates modern local LLMs with Mesh trust validation

This module bridges traditional neural network models (GGUF, KoboldCpp, Ollama) 
with The Mesh's social consensus and trust validation system.

Core Philosophy:
- LLMs provide computational intelligence 
- Mesh provides social validation and trust scoring
- Combined system offers verified, trustworthy AI responses
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Structured LLM response with metadata"""
    content: str
    model_name: str
    model_version: str
    response_time: float
    token_count: int
    confidence_score: float
    generation_params: Dict[str, Any]
    timestamp: datetime
    response_hash: str

@dataclass
class TrustMetrics:
    """Trust validation metrics for LLM responses"""
    social_consensus: float      # Peer agreement score (0-1)
    factual_alignment: float     # Truth verification score (0-1) 
    bias_detection: float        # Bias/manipulation detection (0-1, lower is better)
    source_credibility: float    # Model source credibility (0-1)
    historical_accuracy: float   # Past performance score (0-1)
    context_relevance: float     # Response relevance score (0-1)
    mesh_confidence: float       # Overall Mesh trust score (0-1)

@dataclass
class ValidationContext:
    """Context for LLM response validation"""
    query: str
    user_intent: str
    privacy_level: str
    required_confidence: float
    validation_peers: List[str]
    biometric_verified: bool
    coercion_detected: bool

class LLMTrustValidator:
    """
    Validates LLM outputs through Mesh social consensus and trust networks
    
    Integration Flow:
    1. LLM generates response locally (KoboldCpp/Ollama)
    2. Response analyzed for bias/manipulation
    3. Cross-validated with peer nodes
    4. Trust metrics calculated
    5. Social consensus formed
    6. Verified response returned to user
    """
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.trust_history: Dict[str, List[TrustMetrics]] = {}
        self.model_registry: Dict[str, Dict] = {}
        self.peer_validators: List[str] = []
        self.validation_cache: Dict[str, Tuple[LLMResponse, TrustMetrics]] = {}
        
    async def validate_llm_response(
        self, 
        response: LLMResponse, 
        context: ValidationContext
    ) -> Tuple[LLMResponse, TrustMetrics]:
        """
        Main validation pipeline for LLM responses
        """
        logger.info(f"Validating LLM response from {response.model_name}")
        
        # 1. Check cache first
        cache_key = self._generate_cache_key(response, context)
        if cache_key in self.validation_cache:
            logger.debug("Returning cached validation result")
            return self.validation_cache[cache_key]
        
        # 2. Initialize trust metrics
        trust_metrics = TrustMetrics(
            social_consensus=0.0,
            factual_alignment=0.0, 
            bias_detection=0.0,
            source_credibility=0.0,
            historical_accuracy=0.0,
            context_relevance=0.0,
            mesh_confidence=0.0
        )
        
        # 3. Validate response through multiple channels
        trust_metrics.source_credibility = await self._assess_model_credibility(response.model_name)
        trust_metrics.bias_detection = await self._detect_bias_manipulation(response, context)
        trust_metrics.context_relevance = await self._assess_context_relevance(response, context)
        trust_metrics.factual_alignment = await self._validate_factual_claims(response)
        trust_metrics.social_consensus = await self._gather_peer_consensus(response, context)
        trust_metrics.historical_accuracy = await self._assess_historical_performance(response.model_name)
        
        # 4. Calculate overall mesh confidence
        trust_metrics.mesh_confidence = self._calculate_mesh_confidence(trust_metrics)
        
        # 5. Update trust history
        await self._update_trust_history(response.model_name, trust_metrics)
        
        # 6. Cache result
        self.validation_cache[cache_key] = (response, trust_metrics)
        
        logger.info(f"Validation complete - Mesh confidence: {trust_metrics.mesh_confidence:.3f}")
        return response, trust_metrics
    
    async def _assess_model_credibility(self, model_name: str) -> float:
        """Assess the credibility of the source model"""
        if model_name not in self.model_registry:
            # Unknown model - lower credibility
            return 0.5
            
        model_info = self.model_registry[model_name]
        credibility = 0.7  # Base credibility
        
        # Boost for verified models
        if model_info.get('verified', False):
            credibility += 0.2
            
        # Boost for models with good track record
        if model_info.get('accuracy_score', 0) > 0.8:
            credibility += 0.1
            
        return min(credibility, 1.0)
    
    async def _detect_bias_manipulation(self, response: LLMResponse, context: ValidationContext) -> float:
        """Detect potential bias or manipulation in response"""
        bias_indicators = 0
        total_checks = 5
        
        # Check 1: Intent alignment - does response match user's true intent?
        if context.user_intent and not self._check_intent_alignment(response.content, context.user_intent):
            bias_indicators += 1
            
        # Check 2: Coercion detection - is user being coerced?
        if context.coercion_detected:
            bias_indicators += 2  # Heavy penalty for coercion
            
        # Check 3: Extreme language detection
        if self._contains_extreme_language(response.content):
            bias_indicators += 1
            
        # Check 4: Factual consistency
        if self._check_internal_consistency(response.content):
            bias_indicators += 1
            
        # Check 5: Privacy violations
        if self._check_privacy_violations(response.content, context.privacy_level):
            bias_indicators += 1
            
        # Return bias score (0 = no bias, 1 = maximum bias)
        return bias_indicators / total_checks
    
    async def _assess_context_relevance(self, response: LLMResponse, context: ValidationContext) -> float:
        """Assess how relevant the response is to the original query"""
        # Simple relevance scoring based on keyword overlap and semantic similarity
        query_tokens = set(context.query.lower().split())
        response_tokens = set(response.content.lower().split())
        
        # Jaccard similarity as basic relevance measure
        intersection = len(query_tokens & response_tokens)
        union = len(query_tokens | response_tokens)
        
        if union == 0:
            return 0.0
            
        relevance = intersection / union
        
        # Boost for reasonable response length
        if 50 <= len(response.content) <= 2000:
            relevance += 0.1
            
        return min(relevance, 1.0)
    
    async def _validate_factual_claims(self, response: LLMResponse) -> float:
        """Validate factual claims through external verification"""
        # This would integrate with AxiomEngine for fact verification
        try:
            from ..axiom_integration import AxiomValidator
            axiom_validator = AxiomValidator()
            return await axiom_validator.verify_claims(response.content)
        except ImportError:
            # Fallback to basic fact checking
            return 0.7  # Neutral score when external validation unavailable
    
    async def _gather_peer_consensus(self, response: LLMResponse, context: ValidationContext) -> float:
        """Gather consensus from peer nodes about response quality"""
        if not context.validation_peers:
            return 0.5  # No peers available
            
        peer_scores = []
        
        for peer_id in context.validation_peers:
            try:
                # In real implementation, this would be network calls to peer nodes
                peer_score = await self._get_peer_validation(peer_id, response, context)
                peer_scores.append(peer_score)
            except Exception as e:
                logger.warning(f"Failed to get validation from peer {peer_id}: {e}")
                
        if not peer_scores:
            return 0.5
            
        # Return average peer consensus
        return sum(peer_scores) / len(peer_scores)
    
    async def _assess_historical_performance(self, model_name: str) -> float:
        """Assess model's historical performance in the mesh"""
        if model_name not in self.trust_history:
            return 0.6  # Neutral score for new models
            
        history = self.trust_history[model_name]
        
        if len(history) < 5:
            return 0.6  # Need more data
            
        # Average mesh confidence over recent history
        recent_scores = [tm.mesh_confidence for tm in history[-10:]]
        return sum(recent_scores) / len(recent_scores)
    
    def _calculate_mesh_confidence(self, metrics: TrustMetrics) -> float:
        """Calculate overall mesh confidence using weighted metrics"""
        weights = {
            'social_consensus': 0.25,     # Peer validation is important
            'factual_alignment': 0.20,    # Facts matter
            'bias_detection': -0.15,      # Bias reduces confidence (negative weight)
            'source_credibility': 0.15,   # Model reputation matters
            'historical_accuracy': 0.10,  # Track record important
            'context_relevance': 0.20,    # Relevance is key
        }
        
        confidence = 0.5  # Base confidence
        
        for metric, weight in weights.items():
            value = getattr(metrics, metric)
            confidence += weight * value
            
        return max(0.0, min(1.0, confidence))
    
    async def _update_trust_history(self, model_name: str, metrics: TrustMetrics):
        """Update trust history for the model"""
        if model_name not in self.trust_history:
            self.trust_history[model_name] = []
            
        self.trust_history[model_name].append(metrics)
        
        # Keep only last 100 entries per model
        if len(self.trust_history[model_name]) > 100:
            self.trust_history[model_name] = self.trust_history[model_name][-100:]
    
    def _generate_cache_key(self, response: LLMResponse, context: ValidationContext) -> str:
        """Generate cache key for validation result"""
        key_data = f"{response.response_hash}{context.query}{context.user_intent}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _check_intent_alignment(self, content: str, user_intent: str) -> bool:
        """Check if response aligns with user's true intent"""
        # Simplified intent alignment check
        intent_keywords = user_intent.lower().split()
        content_lower = content.lower()
        
        matches = sum(1 for keyword in intent_keywords if keyword in content_lower)
        return matches >= len(intent_keywords) * 0.5
    
    def _contains_extreme_language(self, content: str) -> bool:
        """Check for extreme or inflammatory language"""
        extreme_indicators = [
            'always', 'never', 'everyone', 'nobody', 'completely wrong',
            'absolutely must', 'totally false', 'only way', 'without exception'
        ]
        
        content_lower = content.lower()
        return sum(1 for indicator in extreme_indicators if indicator in content_lower) > 2
    
    def _check_internal_consistency(self, content: str) -> bool:
        """Check for internal contradictions in the response"""
        # Simplified consistency check - look for contradiction keywords
        contradictions = ['but', 'however', 'although', 'despite', 'contrary']
        return sum(1 for word in contradictions if word in content.lower()) > 3
    
    def _check_privacy_violations(self, content: str, privacy_level: str) -> bool:
        """Check for privacy violations based on user's privacy preferences"""
        if privacy_level == 'high':
            # Check for personal information sharing
            personal_indicators = ['my', 'your name', 'your address', 'personal']
            return any(indicator in content.lower() for indicator in personal_indicators)
        return False
    
    async def _get_peer_validation(self, peer_id: str, response: LLMResponse, context: ValidationContext) -> float:
        """Get validation score from a peer node"""
        # This would be implemented as network calls to peer mesh nodes
        # For now, return simulated peer validation
        return np.random.uniform(0.3, 0.9)
    
    def register_model(self, model_name: str, model_info: Dict[str, Any]):
        """Register a model in the trust system"""
        self.model_registry[model_name] = model_info
        logger.info(f"Registered model: {model_name}")
    
    def get_model_trust_score(self, model_name: str) -> float:
        """Get current trust score for a model"""
        if model_name not in self.trust_history:
            return 0.6  # Neutral for unknown models
            
        recent_metrics = self.trust_history[model_name][-10:]
        if not recent_metrics:
            return 0.6
            
        return sum(m.mesh_confidence for m in recent_metrics) / len(recent_metrics)
    
    def export_trust_report(self) -> Dict[str, Any]:
        """Export comprehensive trust report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_tracked': len(self.model_registry),
            'total_validations': sum(len(history) for history in self.trust_history.values()),
            'model_scores': {
                model: self.get_model_trust_score(model) 
                for model in self.model_registry.keys()
            }
        }
        return report