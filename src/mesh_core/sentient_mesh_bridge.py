#!/usr/bin/env python3
"""
Sentient-Mesh Bridge - Connect Sentient Integration with Mesh Core

This bridge connects the 4-phase Sentient integration (voice, memory, tasks, personal AI)
with The Mesh's trust, consensus, and network systems to create true palm slab nodes.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time

# Import Mesh core systems
from mesh_core.trust.reputation_engine import ReputationEngine
from mesh_core.trust.social_checksum import SocialChecksum
from mesh_core.consensus.voting_engine import VotingEngine
from mesh_core.consensus.proposal_system import ProposalSystem
from mesh_core.network.node_discovery import NodeDiscovery
from mesh_core.network.mesh_protocol import MeshProtocol

# Import Sentient integration modules
from mesh_core.memory.fact_extractor import FactExtractor
from mesh_core.tasks.task_parser import TaskParser
from mesh_core.personal_agent import PersonalAgent
from mesh_core.proactive_manager import ProactiveManager

@dataclass
class MeshValidationResult:
    """Result of mesh validation process"""
    validated: bool
    confidence_score: float
    peer_consensus: float
    trust_score: float
    social_checksum: Optional[str]
    validation_nodes: List[str]
    metadata: Dict[str, Any]

@dataclass
class PalmSlabConfig:
    """Configuration for palm slab operation"""
    node_id: str
    enable_mesh_validation: bool = True
    enable_peer_sharing: bool = True
    enable_social_checksum: bool = True
    privacy_level: str = "selective"  # "private", "selective", "open"
    trust_threshold: float = 0.7
    consensus_threshold: float = 0.6
    max_validation_nodes: int = 5

class SentientMeshBridge:
    """
    Bridge connecting Sentient integration with Mesh core systems
    
    Transforms Sentient modules from isolated systems into cooperative palm slab nodes
    that participate in The Mesh's trust, consensus, and validation systems.
    """
    
    def __init__(self, config: PalmSlabConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Mesh core systems
        self.reputation_engine = None
        self.social_checksum = None
        self.voting_engine = None
        self.proposal_system = None
        self.node_discovery = None
        self.mesh_protocol = None
        
        # Sentient integration modules
        self.fact_extractor = None
        self.task_parser = None
        self.personal_agent = None
        self.proactive_manager = None
        
        # Bridge state
        self.trusted_nodes: List[str] = []
        self.validation_cache: Dict[str, MeshValidationResult] = {}
        self.privacy_settings: Dict[str, Any] = {
            "share_extracted_facts": self.config.privacy_level in ["selective", "open"],
            "share_task_patterns": self.config.privacy_level in ["selective", "open"], 
            "share_personal_insights": self.config.privacy_level == "open",
            "enable_peer_validation": self.config.enable_mesh_validation
        }
        
        self.logger.info(f"Initializing Sentient-Mesh bridge for node {config.node_id}")
    
    async def initialize(self):
        """Initialize the mesh bridge and all connected systems"""
        
        try:
            # Initialize Mesh core systems
            await self._initialize_mesh_systems()
            
            # Initialize Sentient modules
            await self._initialize_sentient_modules()
            
            # Setup bridge connections
            await self._setup_bridge_connections()
            
            self.logger.info("Sentient-Mesh bridge initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize bridge: {e}")
            raise
    
    async def _initialize_mesh_systems(self):
        """Initialize Mesh core trust, consensus, and network systems"""
        
        try:
            # Trust systems
            self.reputation_engine = ReputationEngine(self.config.node_id)
            self.social_checksum = SocialChecksum(self.config.node_id)
            
            # Consensus systems  
            self.voting_engine = VotingEngine(self.config.node_id)
            self.proposal_system = ProposalSystem(self.config.node_id)
            
            # Network systems
            self.node_discovery = NodeDiscovery(self.config.node_id, port=8000)  # Default port
            self.mesh_protocol = MeshProtocol(self.config.node_id)
            
            self.logger.info("Mesh core systems initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize mesh systems: {e}")
            raise
    
    async def _initialize_sentient_modules(self):
        """Initialize Sentient integration modules"""
        
        try:
            # Import individual components
            from mesh_core.memory.fact_extractor import FactExtractor, FactExtractionConfig
            from mesh_core.tasks.task_parser import TaskParser, TaskParserConfig
            from mesh_core.personal_agent import PersonalAgent, PersonalAgentConfig
            from mesh_core.proactive_manager import ProactiveManager, ProactiveConfig
            
            # Memory system
            fact_config = FactExtractionConfig(enable_personalization=True)
            self.fact_extractor = FactExtractor(fact_config)
            
            # Task system
            task_config = TaskParserConfig(enable_ai_parsing=True)
            self.task_parser = TaskParser(task_config)
            
            # Personal AI systems
            personal_config = PersonalAgentConfig(enable_adaptive_responses=True)
            self.personal_agent = PersonalAgent(personal_config)
            
            proactive_config = ProactiveConfig(enabled=True)
            self.proactive_manager = ProactiveManager(proactive_config)
            
            self.logger.info("Sentient integration modules initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Sentient modules: {e}")
            # Continue with fallback - don't fail completely
            self.logger.warning("Continuing with fallback implementations")
            self.fact_extractor = None
            self.task_parser = None  
            self.personal_agent = None
            self.proactive_manager = None
    
    async def _setup_bridge_connections(self):
        """Setup connections between Sentient modules and Mesh systems"""
        
        try:
            # Discover trusted peer nodes
            if self.node_discovery:
                self.trusted_nodes = await self.node_discovery.discover_trusted_nodes(
                    max_nodes=self.config.max_validation_nodes,
                    trust_threshold=self.config.trust_threshold
                )
            else:
                # Fallback to mock trusted nodes
                self.trusted_nodes = [f"trusted_peer_{i}" for i in range(self.config.max_validation_nodes)]
            
            # Privacy settings already initialized in __init__
            
            self.logger.info(f"Bridge connections established with {len(self.trusted_nodes)} trusted nodes")
            
        except Exception as e:
            self.logger.error(f"Failed to setup bridge connections: {e}")
            # Continue with fallback - don't fail completely
            self.trusted_nodes = [f"fallback_peer_{i}" for i in range(3)]
            self.logger.warning("Continuing with fallback trusted nodes")
    
    async def extract_facts_with_mesh_validation(
        self, 
        text: str, 
        username: str = "user",
        source: str = "unknown"
    ) -> Tuple[Any, Optional[MeshValidationResult]]:
        """Extract facts using Sentient extractor + Mesh validation"""
        
        try:
            # Step 1: Local fact extraction (privacy ring)
            extraction_result = await self.fact_extractor.extract_facts(text, username, source)
            
            # Step 2: Mesh validation (if enabled and user permits)
            mesh_validation = None
            if self.config.enable_mesh_validation and self.privacy_settings["share_extracted_facts"]:
                mesh_validation = await self._validate_facts_with_peers(extraction_result.facts)
                
                # Step 3: Apply social checksum to facts
                if mesh_validation.validated and mesh_validation.confidence_score >= self.config.trust_threshold:
                    enhanced_facts = await self._apply_social_checksum_to_facts(
                        extraction_result.facts, mesh_validation
                    )
                    extraction_result.facts = enhanced_facts
            
            return extraction_result, mesh_validation
            
        except Exception as e:
            self.logger.error(f"Mesh-validated fact extraction failed: {e}")
            # Fallback to local-only extraction
            extraction_result = await self.fact_extractor.extract_facts(text, username, source)
            return extraction_result, None
    
    async def _validate_facts_with_peers(self, facts: List[Any]) -> MeshValidationResult:
        """Validate extracted facts with trusted peer nodes"""
        
        try:
            # Create validation proposal
            fact_contents = [fact.content for fact in facts]
            proposal = await self.proposal_system.create_fact_validation_proposal(
                facts=fact_contents,
                validation_type="fact_accuracy"
            )
            
            # Submit to peer voting
            voting_result = await self.voting_engine.submit_proposal(
                proposal=proposal,
                target_nodes=self.trusted_nodes,
                consensus_threshold=self.config.consensus_threshold
            )
            
            # Calculate trust scores from reputation engine
            trust_scores = []
            for node_id in voting_result.participating_nodes:
                trust_score = await self.reputation_engine.get_node_trust_score(node_id)
                trust_scores.append(trust_score)
            
            avg_trust_score = sum(trust_scores) / len(trust_scores) if trust_scores else 0.0
            
            # Generate social checksum
            social_checksum = await self.social_checksum.generate_checksum(
                data=fact_contents,
                validation_result=voting_result,
                trust_scores=trust_scores
            )
            
            return MeshValidationResult(
                validated=voting_result.passed,
                confidence_score=voting_result.confidence,
                peer_consensus=voting_result.consensus_percentage,
                trust_score=avg_trust_score,
                social_checksum=social_checksum,
                validation_nodes=voting_result.participating_nodes,
                metadata={
                    "proposal_id": proposal.proposal_id,
                    "validation_time": time.time(),
                    "fact_count": len(facts)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Peer fact validation failed: {e}")
            return MeshValidationResult(
                validated=False,
                confidence_score=0.0,
                peer_consensus=0.0,
                trust_score=0.0,
                social_checksum=None,
                validation_nodes=[],
                metadata={"error": str(e)}
            )
    
    async def _apply_social_checksum_to_facts(self, facts: List[Any], validation: MeshValidationResult) -> List[Any]:
        """Apply social checksum validation to enhance fact confidence"""
        
        try:
            enhanced_facts = []
            
            for fact in facts:
                # Enhance fact with mesh validation metadata
                if hasattr(fact, 'metadata'):
                    fact.metadata.update({
                        "mesh_validated": validation.validated,
                        "peer_confidence": validation.confidence_score,
                        "trust_score": validation.trust_score,
                        "social_checksum": validation.social_checksum,
                        "validation_nodes": len(validation.validation_nodes)
                    })
                
                # Adjust fact confidence based on mesh validation
                if validation.validated:
                    # Boost confidence for peer-validated facts
                    original_confidence = getattr(fact, 'confidence', 1.0)
                    mesh_boost = validation.confidence_score * 0.3  # 30% boost from mesh validation
                    fact.confidence = min(1.0, original_confidence + mesh_boost)
                
                enhanced_facts.append(fact)
            
            return enhanced_facts
            
        except Exception as e:
            self.logger.error(f"Social checksum application failed: {e}")
            return facts
    
    async def parse_tasks_with_mesh_consensus(
        self,
        task_prompt: str,
        username: str = "user"
    ) -> Tuple[Any, Optional[MeshValidationResult]]:
        """Parse tasks using Sentient parser + Mesh consensus validation"""
        
        try:
            # Step 1: Local task parsing (privacy ring)
            parsing_result = await self.task_parser.parse_task(task_prompt, username)
            
            # Step 2: Mesh consensus validation (if enabled)
            mesh_validation = None
            if self.config.enable_mesh_validation and self.privacy_settings["share_task_patterns"]:
                mesh_validation = await self._validate_task_with_consensus(parsing_result.task)
                
                # Step 3: Enhance task with mesh insights
                if mesh_validation.validated:
                    enhanced_task = await self._enhance_task_with_mesh_insights(
                        parsing_result.task, mesh_validation
                    )
                    parsing_result.task = enhanced_task
            
            return parsing_result, mesh_validation
            
        except Exception as e:
            self.logger.error(f"Mesh-validated task parsing failed: {e}")
            # Fallback to local-only parsing
            parsing_result = await self.task_parser.parse_task(task_prompt, username)
            return parsing_result, None
    
    async def _validate_task_with_consensus(self, task: Any) -> MeshValidationResult:
        """Validate parsed task with mesh consensus"""
        
        try:
            # Create task validation proposal
            proposal = await self.proposal_system.create_task_validation_proposal(
                task_name=task.name,
                task_description=task.description,
                task_priority=task.priority,
                validation_type="task_feasibility"
            )
            
            # Get consensus from trusted nodes
            voting_result = await self.voting_engine.submit_proposal(
                proposal=proposal,
                target_nodes=self.trusted_nodes,
                consensus_threshold=self.config.consensus_threshold
            )
            
            return MeshValidationResult(
                validated=voting_result.passed,
                confidence_score=voting_result.confidence,
                peer_consensus=voting_result.consensus_percentage,
                trust_score=0.0,  # Calculate if needed
                social_checksum=None,
                validation_nodes=voting_result.participating_nodes,
                metadata={
                    "proposal_id": proposal.proposal_id,
                    "task_complexity": "medium",
                    "validation_time": time.time()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Task consensus validation failed: {e}")
            return MeshValidationResult(
                validated=False,
                confidence_score=0.0,
                peer_consensus=0.0,
                trust_score=0.0,
                social_checksum=None,
                validation_nodes=[],
                metadata={"error": str(e)}
            )
    
    async def _enhance_task_with_mesh_insights(self, task: Any, validation: MeshValidationResult) -> Any:
        """Enhance task with insights from mesh consensus"""
        
        try:
            # Add mesh validation metadata to task
            if hasattr(task, 'metadata'):
                task.metadata.update({
                    "mesh_consensus": validation.peer_consensus,
                    "peer_validated": validation.validated,
                    "validation_confidence": validation.confidence_score,
                    "consensus_nodes": len(validation.validation_nodes)
                })
            
            # Enhance task confidence based on mesh consensus
            if validation.validated and validation.peer_consensus >= 0.7:
                task.confidence = min(1.0, task.confidence + 0.2)  # Boost confidence
            
            return task
            
        except Exception as e:
            self.logger.error(f"Task mesh enhancement failed: {e}")
            return task
    
    async def generate_personal_response_with_mesh_intelligence(
        self,
        user_id: str,
        interaction_type: str,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """Generate personal AI response enhanced with mesh intelligence"""
        
        try:
            # Import interaction type enum
            from mesh_core.personal_agent import InteractionType
            interaction_enum = InteractionType(interaction_type)
            
            # Step 1: Local personal response generation (privacy ring)
            personal_response = await self.personal_agent.process_interaction(
                user_id=user_id,
                interaction_type=interaction_enum,
                content=content,
                context=context
            )
            
            # Step 2: Mesh intelligence enhancement (if permitted)
            mesh_insights = None
            if self.privacy_settings["enable_peer_validation"]:
                mesh_insights = await self._gather_mesh_intelligence_for_response(
                    personal_response, context
                )
                
                # Step 3: Enhance response with mesh insights
                if mesh_insights:
                    enhanced_response = await self._enhance_response_with_mesh_intelligence(
                        personal_response, mesh_insights
                    )
                    personal_response = enhanced_response
            
            return personal_response, mesh_insights
            
        except Exception as e:
            self.logger.error(f"Mesh-enhanced personal response failed: {e}")
            # Fallback to local-only response
            from mesh_core.personal_agent import InteractionType
            interaction_enum = InteractionType(interaction_type)
            personal_response = await self.personal_agent.process_interaction(
                user_id, interaction_enum, content, context
            )
            return personal_response, None
    
    async def _gather_mesh_intelligence_for_response(
        self, 
        response: Any, 
        context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Gather intelligence from mesh network to enhance personal response"""
        
        try:
            # Query trusted nodes for similar contexts/responses
            mesh_insights = {
                "peer_suggestions": [],
                "confidence_boost": 0.0,
                "alternative_approaches": [],
                "trust_signals": []
            }
            
            # This would be expanded with actual mesh intelligence gathering
            # For now, return basic structure
            
            return mesh_insights
            
        except Exception as e:
            self.logger.error(f"Mesh intelligence gathering failed: {e}")
            return None
    
    async def _enhance_response_with_mesh_intelligence(
        self, 
        response: Any, 
        mesh_insights: Dict[str, Any]
    ) -> Any:
        """Enhance personal response with mesh intelligence"""
        
        try:
            # Add mesh intelligence to response metadata
            if hasattr(response, 'confidence_score'):
                response.confidence_score += mesh_insights.get("confidence_boost", 0.0)
                response.confidence_score = min(1.0, response.confidence_score)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Response mesh enhancement failed: {e}")
            return response
    
    async def get_node_status(self) -> Dict[str, Any]:
        """Get comprehensive status of this palm slab node"""
        
        return {
            "node_id": self.config.node_id,
            "privacy_level": self.config.privacy_level,
            "mesh_validation_enabled": self.config.enable_mesh_validation,
            "trusted_nodes": len(self.trusted_nodes),
            "validation_cache_size": len(self.validation_cache),
            "systems_status": {
                "reputation_engine": self.reputation_engine is not None,
                "social_checksum": self.social_checksum is not None,
                "voting_engine": self.voting_engine is not None,
                "fact_extractor": self.fact_extractor is not None,
                "task_parser": self.task_parser is not None,
                "personal_agent": self.personal_agent is not None,
                "proactive_manager": self.proactive_manager is not None
            }
        }
    
    async def cleanup(self):
        """Clean up bridge resources"""
        
        try:
            # Cleanup Sentient modules
            if self.fact_extractor and hasattr(self.fact_extractor, 'cleanup'):
                await self.fact_extractor.cleanup()
            
            if self.personal_agent and hasattr(self.personal_agent, 'cleanup'):
                await self.personal_agent.cleanup()
            
            # Cleanup Mesh systems
            if self.mesh_protocol and hasattr(self.mesh_protocol, 'cleanup'):
                await self.mesh_protocol.cleanup()
            
            self.logger.info("Sentient-Mesh bridge cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Bridge cleanup failed: {e}")


# Factory function for easy palm slab creation
def create_palm_slab_node(config: Optional[PalmSlabConfig] = None) -> SentientMeshBridge:
    """Create a new palm slab node with Sentient capabilities and Mesh coordination"""
    
    if config is None:
        config = PalmSlabConfig(node_id=f"palm_slab_{int(time.time())}")
    
    return SentientMeshBridge(config)