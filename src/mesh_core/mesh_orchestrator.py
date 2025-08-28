"""
Mesh Orchestrator - Central coordination for all Mesh components
The brain that coordinates truth, empathy, intent monitoring, and personal AI
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from .truth_layer import TruthLayer
from .intent_monitor import IntentMonitor  
from .empathy_engine import EmpathyEngine
from .personal_agent import PersonalAgent
from .palm_slab import PalmSlabInterface


class MeshResponseType(Enum):
    OBJECTIVE_TRUTH = "objective_truth"
    PERSONAL_GUIDANCE = "personal_guidance"  
    SOCIAL_MEDIATION = "social_mediation"
    MANIPULATION_WARNING = "manipulation_warning"
    EMPATHETIC_SUPPORT = "empathetic_support"


@dataclass
class MeshQuery:
    """Unified query structure for all Mesh interactions"""
    user_id: str
    query: str
    context: Dict[str, Any]
    biometric_data: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    requires_authentication: bool = True


@dataclass 
class MeshResponse:
    """Unified response structure from Mesh"""
    response_type: MeshResponseType
    content: str
    confidence: float
    source_facts: List[Dict[str, Any]]
    empathy_score: Optional[float] = None
    manipulation_detected: bool = False
    truth_breadcrumbs: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class MeshOrchestrator:
    """
    Central coordinator for all Mesh operations
    
    The Mesh is the invisible nervous system behind every palm slab.
    It replaces central authority with cooperative trust, localized inference, 
    and adaptive synchronization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.truth_layer = TruthLayer(config.get('axiom_config', {}))
        self.intent_monitor = IntentMonitor(config.get('intent_config', {}))
        self.empathy_engine = EmpathyEngine(config.get('empathy_config', {}))
        self.personal_agent = PersonalAgent(config.get('personal_config', {}))
        self.palm_slab = PalmSlabInterface(config.get('slab_config', {}))
        
        # User-specific data (local-first)
        self.user_contexts: Dict[str, Dict[str, Any]] = {}
        self.trust_ledgers: Dict[str, Any] = {}
        self.intent_baselines: Dict[str, Any] = {}
        
        self.logger.info("Mesh Orchestrator initialized - Ready for decentralized truth")
    
    async def process_mesh_query(self, mesh_query: MeshQuery) -> MeshResponse:
        """
        Process a query through the complete Mesh pipeline
        
        Core Principles:
        - Decentralization by Default: Every slab is a full node
        - Data is Local First: Personal knowledge stays private
        - Consensus through Cross-Validation: No single source of truth
        - Adaptive Synapses: Weighted connections based on reliability
        """
        
        try:
            # Step 1: Authentication & Intent Baseline
            if mesh_query.requires_authentication:
                auth_result = await self._authenticate_user(mesh_query)
                if not auth_result.success:
                    return self._create_auth_failure_response(auth_result.reason)
            
            # Step 2: Intent Analysis & Manipulation Detection
            intent_analysis = await self.intent_monitor.analyze_query(
                mesh_query.query, 
                self._get_user_intent_baseline(mesh_query.user_id)
            )
            
            # Step 3: Truth Layer Processing  
            truth_response = await self.truth_layer.process_query(
                mesh_query.query,
                mesh_query.context
            )
            
            # Step 4: Personal Context Integration
            personal_response = await self.personal_agent.contextualize(
                mesh_query.query,
                mesh_query.context, 
                truth_response.facts,
                self._get_user_context(mesh_query.user_id)
            )
            
            # Step 5: Check for Intent Divergence (Manipulation Detection)
            manipulation_check = await self.intent_monitor.detect_manipulation(
                mesh_query.query,
                personal_response.content,
                intent_analysis.baseline_intent
            )
            
            # Step 6: Social Repair if Needed
            empathy_response = None
            if mesh_query.context.get('conflict_detected') or manipulation_check.requires_mediation:
                empathy_response = await self.empathy_engine.generate_mediation(
                    mesh_query.query,
                    mesh_query.context,
                    truth_response.facts
                )
            
            # Step 7: Synthesize Final Response
            return await self._synthesize_mesh_response(
                mesh_query,
                truth_response,
                personal_response, 
                intent_analysis,
                manipulation_check,
                empathy_response
            )
            
        except Exception as e:
            self.logger.error(f"Mesh processing error: {e}")
            return self._create_error_response(str(e))
    
    async def _authenticate_user(self, mesh_query: MeshQuery) -> Any:
        """Authenticate user via Palm Slab interface"""
        return await self.palm_slab.authenticate(
            mesh_query.user_id,
            mesh_query.biometric_data
        )
    
    def _get_user_intent_baseline(self, user_id: str) -> Dict[str, Any]:
        """Get user's historical intent patterns for manipulation detection"""
        if user_id not in self.intent_baselines:
            self.intent_baselines[user_id] = self.intent_monitor.build_baseline(user_id)
        return self.intent_baselines[user_id]
    
    def _get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get user's personal context (local-first storage)"""
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = self._load_user_context(user_id)
        return self.user_contexts[user_id]
    
    def _load_user_context(self, user_id: str) -> Dict[str, Any]:
        """Load user context from local storage"""
        # Implementation will load from local encrypted storage
        return {
            "preferences": {},
            "trust_network": {},
            "learning_history": {},
            "emotional_state": {}
        }
    
    async def _synthesize_mesh_response(
        self,
        query: MeshQuery,
        truth_response: Any,
        personal_response: Any,
        intent_analysis: Any, 
        manipulation_check: Any,
        empathy_response: Optional[Any] = None
    ) -> MeshResponse:
        """
        Synthesize all component responses into unified Mesh response
        
        The Mesh doesn't just collect facts; it understands relationships
        and provides context-aware guidance while protecting user autonomy.
        """
        
        # Determine primary response type
        if manipulation_check.manipulation_detected:
            response_type = MeshResponseType.MANIPULATION_WARNING
            content = self._generate_manipulation_warning(
                truth_response, manipulation_check
            )
        elif empathy_response:
            response_type = MeshResponseType.SOCIAL_MEDIATION  
            content = empathy_response.mediation_guidance
        elif personal_response.confidence > truth_response.confidence:
            response_type = MeshResponseType.PERSONAL_GUIDANCE
            content = personal_response.content
        else:
            response_type = MeshResponseType.OBJECTIVE_TRUTH
            content = truth_response.content
        
        # Create unified response
        return MeshResponse(
            response_type=response_type,
            content=content,
            confidence=max(truth_response.confidence, personal_response.confidence),
            source_facts=truth_response.facts,
            empathy_score=empathy_response.empathy_score if empathy_response else None,
            manipulation_detected=manipulation_check.manipulation_detected,
            truth_breadcrumbs=manipulation_check.breadcrumbs if manipulation_check.manipulation_detected else None,
            metadata={
                "intent_baseline": intent_analysis.baseline_intent,
                "truth_sources": truth_response.sources,
                "personal_context_used": personal_response.context_keys,
                "processing_time": personal_response.processing_time + truth_response.processing_time
            }
        )
    
    def _generate_manipulation_warning(self, truth_response: Any, manipulation_check: Any) -> str:
        """Generate warning when manipulation is detected"""
        return f"""
        ⚠️  Intent Divergence Detected
        
        Your original question was about: {manipulation_check.original_intent}
        But the response is leading toward: {manipulation_check.detected_intent}
        
        Here are the objective facts: {truth_response.summary}
        
        Truth breadcrumbs back to your original goal:
        {chr(10).join(f"• {breadcrumb}" for breadcrumb in manipulation_check.breadcrumbs)}
        """
    
    def _create_auth_failure_response(self, reason: str) -> MeshResponse:
        """Create response for authentication failure"""
        return MeshResponse(
            response_type=MeshResponseType.MANIPULATION_WARNING,
            content=f"Authentication failed: {reason}. Possible coercion detected.",
            confidence=1.0,
            source_facts=[],
            manipulation_detected=True
        )
    
    def _create_error_response(self, error: str) -> MeshResponse:
        """Create response for processing errors"""
        return MeshResponse(
            response_type=MeshResponseType.OBJECTIVE_TRUTH,
            content=f"Mesh processing error: {error}",
            confidence=0.0,
            source_facts=[]
        )
    
    async def update_user_trust_ledger(
        self, 
        user_id: str, 
        interaction_result: Dict[str, Any]
    ):
        """Update user's trust ledger based on interaction outcomes"""
        if user_id not in self.trust_ledgers:
            self.trust_ledgers[user_id] = {}
        
        # Update trust scores based on verification outcomes
        # This implements the "social checksum" concept
        await self._update_trust_scores(user_id, interaction_result)
    
    async def _update_trust_scores(self, user_id: str, result: Dict[str, Any]):
        """Update trust scores for sources based on verification results"""
        # Implementation for adaptive trust scoring
        # Sources gain trust when consistently accurate
        # Sources lose trust when contradicted by primary data
        pass


# Factory function for easy initialization
def create_mesh_orchestrator(config_path: Optional[str] = None) -> MeshOrchestrator:
    """Create and initialize Mesh Orchestrator with configuration"""
    if config_path:
        # Load configuration from file
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            "axiom_config": {"node_port": 8001},
            "intent_config": {"sensitivity_threshold": 0.7},
            "empathy_config": {"emotion_model": "focused-empathy"},
            "personal_config": {"character_system": "victoria_steel"},
            "slab_config": {"biometric_required": True}
        }
    
    return MeshOrchestrator(config)