"""
Axiom-Mesh Bridge - Integration layer between AxiomEngine and The Mesh

Provides seamless integration between AxiomEngine's truth verification
capabilities and The Mesh's distributed consensus system, enabling
hybrid centralized-decentralized truth validation.

Key Features:
- Bidirectional data flow between AxiomEngine and Mesh
- Distributed consensus integration
- Truth synchronization across the network
- Hybrid verification workflows
- Conflict resolution between centralized and distributed truth
- Performance optimization and caching
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

try:
    from .axiom_processor import AxiomProcessor, AxiomVerificationResult
    from .truth_validator import TruthValidator, TruthValidationResult, TruthClaim
    from .knowledge_validator import KnowledgeValidator, ValidationResult
    from .confidence_scorer import ConfidenceScorer, ConfidenceScore
    from ..trust.trust_ledger import TrustLedger
    from ..network.mesh_protocol import MeshProtocol
    from ..network.message_router import MessageRouter
    from ..provenance.provenance_tracker import ProvenanceTracker
    from ..config_manager import ConfigurationManager as ConfigManager
except ImportError:
    # Mock classes for development
    class MockAxiomProcessor:
        async def verify_claim(self, claim: str, context: Optional[Dict] = None):
            return {"verification_status": "verified", "confidence_score": 0.8}
        
        async def submit_fact(self, submission):
            return {"success": True, "fact_id": "mock_fact_123"}
    
    class MockTruthValidator:
        async def validate_truth(self, claim, validation_mode):
            return type('MockResult', (), {
                'truth_status': 'corroborated',
                'overall_confidence': 0.8,
                'consensus_result': None
            })()
    
    class MockKnowledgeValidator:
        async def validate_knowledge_claim(self, claim, validation_level, context=None):
            return type('MockResult', (), {'confidence_score': 0.7})()
    
    class MockConfidenceScorer:
        async def calculate_confidence_score(self, claim, evidence, context=None):
            return type('MockScore', (), {'final_score': 0.75})()
    
    class MockTrustLedger:
        def calculate_trust_score(self, node_id: str) -> float:
            return 0.8
        
        def record_verification_result(self, node_id: str, result: Dict):
            pass
    
    class MockMeshProtocol:
        async def broadcast_truth_verification(self, verification_data: Dict) -> List[Dict]:
            return [{"node_id": "node1", "agreement": True, "confidence": 0.8}]
        
        async def query_network_consensus(self, query: Dict) -> Dict:
            return {"consensus_score": 0.75, "participating_nodes": 5}
    
    class MockMessageRouter:
        async def route_verification_request(self, request: Dict) -> Dict:
            return {"routed": True, "target_nodes": ["node1", "node2"]}
    
    class MockProvenanceTracker:
        def track_verification_chain(self, verification_id: str, chain_data: Dict) -> str:
            return f"trace_{verification_id}"
    
    class MockConfigManager:
        def get_verification_config(self) -> Dict:
            return {"hybrid_mode": True, "consensus_threshold": 0.7}
    
    # Use mocks
    AxiomProcessor = MockAxiomProcessor
    TruthValidator = MockTruthValidator
    KnowledgeValidator = MockKnowledgeValidator
    ConfidenceScorer = MockConfidenceScorer
    TrustLedger = MockTrustLedger
    MeshProtocol = MockMeshProtocol
    MessageRouter = MockMessageRouter
    ProvenanceTracker = MockProvenanceTracker
    ConfigManager = MockConfigManager
    
    # Mock result types
    AxiomVerificationResult = dict
    TruthValidationResult = type('TruthValidationResult', (), {})
    ValidationResult = type('ValidationResult', (), {})
    TruthClaim = type('TruthClaim', (), {})
    ConfidenceScore = type('ConfidenceScore', (), {})


class VerificationStrategy(Enum):
    """Verification strategy options"""
    AXIOM_ONLY = "axiom_only"                    # AxiomEngine verification only
    MESH_ONLY = "mesh_only"                      # Mesh consensus only
    HYBRID_PARALLEL = "hybrid_parallel"          # Both systems in parallel
    HYBRID_SEQUENTIAL = "hybrid_sequential"      # AxiomEngine first, then Mesh
    ADAPTIVE = "adaptive"                        # Choose strategy based on context


class ConflictResolution(Enum):
    """Conflict resolution strategies"""
    TRUST_WEIGHTED = "trust_weighted"            # Weight by source trust
    CONFIDENCE_WEIGHTED = "confidence_weighted"  # Weight by confidence scores
    MAJORITY_CONSENSUS = "majority_consensus"    # Simple majority rule
    EVIDENCE_BASED = "evidence_based"            # Resolution based on evidence quality
    HYBRID_SCORING = "hybrid_scoring"            # Combined scoring approach


@dataclass
class VerificationRequest:
    """Request for truth verification"""
    request_id: str
    claim: str
    strategy: VerificationStrategy
    context: Dict[str, Any] = field(default_factory=dict)
    priority: str = "normal"  # low, normal, high, critical
    source_node: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    timeout: int = 30  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "claim": self.claim,
            "strategy": self.strategy.value,
            "context": self.context,
            "priority": self.priority,
            "source_node": self.source_node,
            "timestamp": self.timestamp.isoformat(),
            "timeout": self.timeout
        }


@dataclass
class HybridVerificationResult:
    """Result of hybrid verification combining AxiomEngine and Mesh"""
    request_id: str
    claim: str
    strategy_used: VerificationStrategy
    
    # Individual results
    axiom_result: Optional[Union[AxiomVerificationResult, Dict]] = None
    mesh_result: Optional[TruthValidationResult] = None
    knowledge_result: Optional[ValidationResult] = None
    confidence_score: Optional[ConfidenceScore] = None
    
    # Combined assessment
    final_verdict: str = "unverified"
    final_confidence: float = 0.0
    consensus_agreement: float = 0.0
    
    # Conflict information
    conflicts_detected: List[Dict[str, Any]] = field(default_factory=list)
    resolution_strategy: Optional[ConflictResolution] = None
    
    # Performance metrics
    processing_time: float = 0.0
    axiom_time: float = 0.0
    mesh_time: float = 0.0
    
    # Metadata
    participating_nodes: List[str] = field(default_factory=list)
    verification_chain: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AxiomMeshBridge:
    """Bridge between AxiomEngine and The Mesh for hybrid truth verification"""
    
    def __init__(self):
        # Core components
        self.axiom_processor = AxiomProcessor()
        self.truth_validator = TruthValidator()
        self.knowledge_validator = KnowledgeValidator()
        self.confidence_scorer = ConfidenceScorer()
        
        # Mesh components
        self.trust_ledger = TrustLedger()
        self.mesh_protocol = MeshProtocol()
        self.message_router = MessageRouter()
        self.provenance_tracker = ProvenanceTracker()
        self.config_manager = ConfigManager()
        
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = self.config_manager.get_verification_config()
        self.default_strategy = VerificationStrategy.HYBRID_PARALLEL
        self.conflict_resolution = ConflictResolution.HYBRID_SCORING
        self.consensus_threshold = self.config.get("consensus_threshold", 0.7)
        
        # Performance settings
        self.parallel_timeout = 30
        self.cache_ttl = timedelta(minutes=15)
        self.verification_cache = {}
        
        # Statistics
        self.verifications_processed = 0
        self.conflicts_resolved = 0
        self.cache_hits = 0
        self.strategy_usage = defaultdict(int)
    
    async def verify_truth_hybrid(
        self,
        claim: str,
        strategy: Optional[VerificationStrategy] = None,
        context: Optional[Dict[str, Any]] = None,
        priority: str = "normal"
    ) -> HybridVerificationResult:
        """Perform hybrid truth verification using both AxiomEngine and Mesh"""
        request_id = f"hybrid_{int(time.time() * 1000000)}"
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting hybrid verification: {request_id}")
            
            # Create verification request
            request = VerificationRequest(
                request_id=request_id,
                claim=claim,
                strategy=strategy or self.default_strategy,
                context=context or {},
                priority=priority
            )
            
            # Check cache first
            cache_result = self._check_verification_cache(claim)
            if cache_result:
                self.cache_hits += 1
                self.logger.info(f"Cache hit for verification: {request_id}")
                return cache_result
            
            # Initialize result
            result = HybridVerificationResult(
                request_id=request_id,
                claim=claim,
                strategy_used=request.strategy
            )
            
            # Execute verification based on strategy
            if request.strategy == VerificationStrategy.AXIOM_ONLY:
                await self._axiom_only_verification(request, result)
            elif request.strategy == VerificationStrategy.MESH_ONLY:
                await self._mesh_only_verification(request, result)
            elif request.strategy == VerificationStrategy.HYBRID_PARALLEL:
                await self._hybrid_parallel_verification(request, result)
            elif request.strategy == VerificationStrategy.HYBRID_SEQUENTIAL:
                await self._hybrid_sequential_verification(request, result)
            elif request.strategy == VerificationStrategy.ADAPTIVE:
                await self._adaptive_verification(request, result)
            
            # Resolve conflicts if any
            await self._resolve_conflicts(result)
            
            # Calculate final verdict and confidence
            self._calculate_final_assessment(result)
            
            # Track provenance
            result.verification_chain = self.provenance_tracker.track_verification_chain(
                request_id,
                {
                    "strategy": request.strategy.value,
                    "claim": claim,
                    "results": self._serialize_results(result)
                }
            )
            
            # Record performance metrics
            result.processing_time = time.time() - start_time
            
            # Cache the result
            self._cache_verification_result(claim, result)
            
            # Update statistics
            self.verifications_processed += 1
            self.strategy_usage[request.strategy] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in hybrid verification {request_id}: {e}")
            return HybridVerificationResult(
                request_id=request_id,
                claim=claim,
                strategy_used=strategy or self.default_strategy,
                final_verdict="error",
                metadata={"error": str(e)},
                processing_time=time.time() - start_time
            )
    
    async def batch_verify_truths(
        self,
        claims: List[str],
        strategy: Optional[VerificationStrategy] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[HybridVerificationResult]:
        """Batch verification of multiple truth claims"""
        tasks = [
            self.verify_truth_hybrid(claim, strategy, context)
            for claim in claims
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def submit_truth_to_network(
        self,
        claim: str,
        evidence: Dict[str, Any],
        confidence: float,
        source_node: Optional[str] = None
    ) -> Dict[str, Any]:
        """Submit a verified truth to both AxiomEngine and Mesh network"""
        try:
            submission_id = f"submit_{int(time.time() * 1000000)}"
            
            # Create submission for AxiomEngine
            from .axiom_processor import AxiomFactSubmission
            axiom_submission = AxiomFactSubmission(
                content=claim,
                confidence=confidence,
                metadata=evidence
            )
            
            # Submit to AxiomEngine
            axiom_result = await self.axiom_processor.submit_fact(axiom_submission)
            
            # Broadcast to Mesh network
            mesh_broadcast = {
                "submission_id": submission_id,
                "claim": claim,
                "evidence": evidence,
                "confidence": confidence,
                "source_node": source_node,
                "axiom_fact_id": axiom_result.get("fact_id"),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            mesh_responses = await self.mesh_protocol.broadcast_truth_verification(
                mesh_broadcast
            )
            
            # Process network responses
            agreement_count = sum(1 for r in mesh_responses if r.get("agreement", False))
            total_responses = len(mesh_responses)
            network_consensus = agreement_count / max(total_responses, 1)
            
            return {
                "submission_id": submission_id,
                "axiom_result": axiom_result,
                "network_consensus": network_consensus,
                "network_responses": total_responses,
                "agreement_ratio": f"{agreement_count}/{total_responses}",
                "success": axiom_result.get("success", False) and network_consensus >= self.consensus_threshold
            }
            
        except Exception as e:
            self.logger.error(f"Error submitting truth to network: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def synchronize_truths(self) -> Dict[str, Any]:
        """Synchronize truths between AxiomEngine and Mesh network"""
        try:
            self.logger.info("Starting truth synchronization...")
            
            # Get recent verifications from Mesh
            mesh_truths = await self._get_recent_mesh_truths()
            
            # Get recent facts from AxiomEngine (would need API)
            # axiom_truths = await self._get_recent_axiom_truths()
            
            # Find discrepancies and sync
            sync_results = {
                "mesh_truths_processed": len(mesh_truths),
                "discrepancies_found": 0,
                "synced_successfully": 0,
                "errors": []
            }
            
            for mesh_truth in mesh_truths:
                try:
                    # Check if truth exists in AxiomEngine
                    # If not, submit it
                    # This would be implemented with actual AxiomEngine API
                    pass
                except Exception as e:
                    sync_results["errors"].append(str(e))
            
            return sync_results
            
        except Exception as e:
            self.logger.error(f"Error synchronizing truths: {e}")
            return {"success": False, "error": str(e)}
    
    async def _axiom_only_verification(
        self,
        request: VerificationRequest,
        result: HybridVerificationResult
    ):
        """Perform AxiomEngine-only verification"""
        try:
            axiom_start = time.time()
            result.axiom_result = await self.axiom_processor.verify_claim(
                request.claim,
                request.context
            )
            result.axiom_time = time.time() - axiom_start
            
        except Exception as e:
            result.metadata["axiom_error"] = str(e)
    
    async def _mesh_only_verification(
        self,
        request: VerificationRequest,
        result: HybridVerificationResult
    ):
        """Perform Mesh-only verification"""
        try:
            mesh_start = time.time()
            
            # Create truth claim
            truth_claim = TruthClaim(
                content=request.claim,
                source=request.source_node,
                context=request.context
            )
            
            result.mesh_result = await self.truth_validator.validate_truth(
                truth_claim,
                validation_mode="standard"
            )
            result.mesh_time = time.time() - mesh_start
            
        except Exception as e:
            result.metadata["mesh_error"] = str(e)
    
    async def _hybrid_parallel_verification(
        self,
        request: VerificationRequest,
        result: HybridVerificationResult
    ):
        """Perform parallel verification with both systems"""
        try:
            # Run both verifications in parallel
            axiom_task = self._axiom_only_verification(request, result)
            mesh_task = self._mesh_only_verification(request, result)
            
            await asyncio.gather(axiom_task, mesh_task, return_exceptions=True)
            
        except Exception as e:
            result.metadata["parallel_error"] = str(e)
    
    async def _hybrid_sequential_verification(
        self,
        request: VerificationRequest,
        result: HybridVerificationResult
    ):
        """Perform sequential verification: AxiomEngine first, then Mesh"""
        try:
            # AxiomEngine first
            await self._axiom_only_verification(request, result)
            
            # Use AxiomEngine result to enhance Mesh verification
            if result.axiom_result:
                enhanced_context = request.context.copy()
                enhanced_context["axiom_verification"] = result.axiom_result
                request.context = enhanced_context
            
            # Then Mesh verification
            await self._mesh_only_verification(request, result)
            
        except Exception as e:
            result.metadata["sequential_error"] = str(e)
    
    async def _adaptive_verification(
        self,
        request: VerificationRequest,
        result: HybridVerificationResult
    ):
        """Adaptively choose verification strategy based on context"""
        try:
            # Analyze context to choose strategy
            strategy = self._choose_adaptive_strategy(request)
            result.strategy_used = strategy
            
            # Execute chosen strategy
            if strategy == VerificationStrategy.AXIOM_ONLY:
                await self._axiom_only_verification(request, result)
            elif strategy == VerificationStrategy.MESH_ONLY:
                await self._mesh_only_verification(request, result)
            else:
                await self._hybrid_parallel_verification(request, result)
                
        except Exception as e:
            result.metadata["adaptive_error"] = str(e)
    
    def _choose_adaptive_strategy(self, request: VerificationRequest) -> VerificationStrategy:
        """Choose verification strategy based on request context"""
        try:
            # Analyze request characteristics
            claim_length = len(request.claim)
            priority = request.priority
            context_complexity = len(request.context)
            
            # Simple heuristics for strategy selection
            if priority == "critical" or claim_length > 500:
                return VerificationStrategy.HYBRID_PARALLEL
            elif context_complexity > 10:
                return VerificationStrategy.HYBRID_SEQUENTIAL
            elif priority == "low":
                return VerificationStrategy.AXIOM_ONLY
            else:
                return VerificationStrategy.HYBRID_PARALLEL
                
        except Exception:
            return VerificationStrategy.HYBRID_PARALLEL
    
    async def _resolve_conflicts(self, result: HybridVerificationResult):
        """Resolve conflicts between different verification results"""
        try:
            conflicts = self._detect_conflicts(result)
            result.conflicts_detected = conflicts
            
            if conflicts:
                self.conflicts_resolved += 1
                result.resolution_strategy = self.conflict_resolution
                
                # Apply resolution strategy
                if self.conflict_resolution == ConflictResolution.CONFIDENCE_WEIGHTED:
                    self._resolve_by_confidence_weighting(result)
                elif self.conflict_resolution == ConflictResolution.TRUST_WEIGHTED:
                    self._resolve_by_trust_weighting(result)
                elif self.conflict_resolution == ConflictResolution.EVIDENCE_BASED:
                    self._resolve_by_evidence_quality(result)
                else:
                    self._resolve_by_hybrid_scoring(result)
                    
        except Exception as e:
            result.metadata["conflict_resolution_error"] = str(e)
    
    def _detect_conflicts(self, result: HybridVerificationResult) -> List[Dict[str, Any]]:
        """Detect conflicts between verification results"""
        conflicts = []
        
        try:
            # Compare AxiomEngine and Mesh results
            if result.axiom_result and result.mesh_result:
                axiom_confidence = self._extract_axiom_confidence(result.axiom_result)
                mesh_confidence = getattr(result.mesh_result, 'overall_confidence', 0.5)
                
                # Significant confidence difference
                if abs(axiom_confidence - mesh_confidence) > 0.3:
                    conflicts.append({
                        "type": "confidence_divergence",
                        "axiom_confidence": axiom_confidence,
                        "mesh_confidence": mesh_confidence,
                        "difference": abs(axiom_confidence - mesh_confidence)
                    })
                
                # Verdict disagreement
                axiom_status = self._extract_axiom_status(result.axiom_result)
                mesh_status = getattr(result.mesh_result, 'truth_status', 'unknown')
                
                if self._are_statuses_conflicting(axiom_status, str(mesh_status)):
                    conflicts.append({
                        "type": "verdict_disagreement",
                        "axiom_status": axiom_status,
                        "mesh_status": str(mesh_status)
                    })
            
        except Exception as e:
            self.logger.error(f"Error detecting conflicts: {e}")
        
        return conflicts
    
    def _resolve_by_confidence_weighting(self, result: HybridVerificationResult):
        """Resolve conflicts by weighting based on confidence scores"""
        try:
            axiom_confidence = self._extract_axiom_confidence(result.axiom_result) if result.axiom_result else 0.0
            mesh_confidence = getattr(result.mesh_result, 'overall_confidence', 0.0) if result.mesh_result else 0.0
            
            # Weight final confidence by individual confidences
            total_weight = axiom_confidence + mesh_confidence
            if total_weight > 0:
                axiom_weight = axiom_confidence / total_weight
                mesh_weight = mesh_confidence / total_weight
                
                result.final_confidence = (axiom_confidence * axiom_weight + 
                                         mesh_confidence * mesh_weight)
            
        except Exception as e:
            self.logger.error(f"Error in confidence weighting: {e}")
    
    def _resolve_by_trust_weighting(self, result: HybridVerificationResult):
        """Resolve conflicts by weighting based on source trust"""
        try:
            # Get trust scores for different sources
            axiom_trust = 0.9  # AxiomEngine assumed high trust
            mesh_trust = 0.7   # Mesh consensus trust
            
            # Weight results by trust
            total_trust = axiom_trust + mesh_trust
            axiom_weight = axiom_trust / total_trust
            mesh_weight = mesh_trust / total_trust
            
            axiom_confidence = self._extract_axiom_confidence(result.axiom_result) if result.axiom_result else 0.0
            mesh_confidence = getattr(result.mesh_result, 'overall_confidence', 0.0) if result.mesh_result else 0.0
            
            result.final_confidence = (axiom_confidence * axiom_weight + 
                                     mesh_confidence * mesh_weight)
            
        except Exception as e:
            self.logger.error(f"Error in trust weighting: {e}")
    
    def _resolve_by_evidence_quality(self, result: HybridVerificationResult):
        """Resolve conflicts based on evidence quality"""
        try:
            axiom_evidence_count = 0
            mesh_evidence_count = 0
            
            # Count evidence from AxiomEngine
            if result.axiom_result:
                corroborations = result.axiom_result.get("corroborations", []) if isinstance(result.axiom_result, dict) else getattr(result.axiom_result, "corroborations", [])
                citations = result.axiom_result.get("citations", []) if isinstance(result.axiom_result, dict) else getattr(result.axiom_result, "citations", [])
                axiom_evidence_count = len(corroborations) + len(citations)
            
            # Count evidence from Mesh
            if result.mesh_result:
                supporting_evidence = getattr(result.mesh_result, 'supporting_evidence', [])
                mesh_evidence_count = len(supporting_evidence)
            
            # Weight by evidence quality
            total_evidence = axiom_evidence_count + mesh_evidence_count
            if total_evidence > 0:
                axiom_weight = axiom_evidence_count / total_evidence
                mesh_weight = mesh_evidence_count / total_evidence
                
                axiom_confidence = self._extract_axiom_confidence(result.axiom_result) if result.axiom_result else 0.0
                mesh_confidence = getattr(result.mesh_result, 'overall_confidence', 0.0) if result.mesh_result else 0.0
                
                result.final_confidence = (axiom_confidence * axiom_weight + 
                                         mesh_confidence * mesh_weight)
                
        except Exception as e:
            self.logger.error(f"Error in evidence-based resolution: {e}")
    
    def _resolve_by_hybrid_scoring(self, result: HybridVerificationResult):
        """Resolve conflicts using hybrid scoring approach"""
        try:
            # Combine confidence, trust, and evidence weighting
            self._resolve_by_confidence_weighting(result)
            confidence_score = result.final_confidence
            
            self._resolve_by_trust_weighting(result)
            trust_score = result.final_confidence
            
            self._resolve_by_evidence_quality(result)
            evidence_score = result.final_confidence
            
            # Final hybrid score
            result.final_confidence = (confidence_score * 0.4 + 
                                     trust_score * 0.3 + 
                                     evidence_score * 0.3)
            
        except Exception as e:
            self.logger.error(f"Error in hybrid scoring: {e}")
    
    def _calculate_final_assessment(self, result: HybridVerificationResult):
        """Calculate final verdict and confidence"""
        try:
            # If no conflicts were resolved, calculate from individual results
            if result.final_confidence == 0.0:
                confidences = []
                
                if result.axiom_result:
                    confidences.append(self._extract_axiom_confidence(result.axiom_result))
                
                if result.mesh_result:
                    confidences.append(getattr(result.mesh_result, 'overall_confidence', 0.5))
                
                if confidences:
                    result.final_confidence = sum(confidences) / len(confidences)
            
            # Determine final verdict based on confidence
            if result.final_confidence >= 0.9:
                result.final_verdict = "empirically_verified"
            elif result.final_confidence >= 0.8:
                result.final_verdict = "strongly_supported"
            elif result.final_confidence >= 0.7:
                result.final_verdict = "corroborated"
            elif result.final_confidence >= 0.6:
                result.final_verdict = "logically_consistent"
            elif result.final_confidence >= 0.4:
                result.final_verdict = "proposed"
            else:
                result.final_verdict = "unverified"
            
        except Exception as e:
            self.logger.error(f"Error calculating final assessment: {e}")
            result.final_verdict = "error"
    
    def _check_verification_cache(self, claim: str) -> Optional[HybridVerificationResult]:
        """Check if verification result is cached"""
        cache_key = f"verify_{hash(claim)}"
        if cache_key in self.verification_cache:
            cached_result, timestamp = self.verification_cache[cache_key]
            if datetime.utcnow() - timestamp < self.cache_ttl:
                return cached_result
            else:
                del self.verification_cache[cache_key]
        return None
    
    def _cache_verification_result(self, claim: str, result: HybridVerificationResult):
        """Cache verification result"""
        cache_key = f"verify_{hash(claim)}"
        self.verification_cache[cache_key] = (result, datetime.utcnow())
    
    def _extract_axiom_confidence(self, axiom_result) -> float:
        """Extract confidence score from AxiomEngine result"""
        if isinstance(axiom_result, dict):
            return axiom_result.get("confidence_score", 0.5)
        else:
            return getattr(axiom_result, "confidence_score", 0.5)
    
    def _extract_axiom_status(self, axiom_result) -> str:
        """Extract status from AxiomEngine result"""
        if isinstance(axiom_result, dict):
            return axiom_result.get("verification_status", "unknown")
        else:
            return getattr(axiom_result, "verification_status", "unknown")
    
    def _are_statuses_conflicting(self, status1: str, status2: str) -> bool:
        """Check if two verification statuses are conflicting"""
        positive_statuses = {"verified", "empirically_verified", "strongly_supported", "corroborated"}
        negative_statuses = {"contradicted", "disputed", "unverified"}
        
        status1_positive = status1.lower() in positive_statuses
        status1_negative = status1.lower() in negative_statuses
        status2_positive = status2.lower() in positive_statuses
        status2_negative = status2.lower() in negative_statuses
        
        return (status1_positive and status2_negative) or (status1_negative and status2_positive)
    
    def _serialize_results(self, result: HybridVerificationResult) -> Dict[str, Any]:
        """Serialize results for provenance tracking"""
        return {
            "final_verdict": result.final_verdict,
            "final_confidence": result.final_confidence,
            "strategy_used": result.strategy_used.value,
            "conflicts_count": len(result.conflicts_detected),
            "processing_time": result.processing_time
        }
    
    async def _get_recent_mesh_truths(self) -> List[Dict[str, Any]]:
        """Get recent truth verifications from Mesh network"""
        # This would query the mesh network for recent verifications
        # Mock implementation
        return []
    
    def get_bridge_statistics(self) -> Dict[str, Any]:
        """Get bridge performance statistics"""
        return {
            "total_verifications": self.verifications_processed,
            "conflicts_resolved": self.conflicts_resolved,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": f"{(self.cache_hits / max(self.verifications_processed, 1)) * 100:.1f}%",
            "strategy_usage": dict(self.strategy_usage),
            "cached_results": len(self.verification_cache)
        }
    
    def reset_statistics(self):
        """Reset bridge statistics"""
        self.verifications_processed = 0
        self.conflicts_resolved = 0
        self.cache_hits = 0
        self.strategy_usage.clear()
        self.verification_cache.clear()