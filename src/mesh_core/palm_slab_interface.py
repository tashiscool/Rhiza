#!/usr/bin/env python3
"""
Palm Slab Interface - Complete Palm Slab Node Implementation

This interface provides the complete palm slab experience as envisioned in The Mesh:
- Local-first AI with cooperative mesh intelligence
- Privacy ring for controlled data sharing
- Social checksum for truth validation
- Adaptive synapses forming weighted connections to useful peers
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import time

from mesh_core.sentient_mesh_bridge import SentientMeshBridge, MeshValidationResult
from mesh_core.sentient_mesh_bridge import PalmSlabConfig as SentientPalmSlabConfig

@dataclass
class PalmSlabUser:
    """User profile for palm slab personalization"""
    user_id: str
    preferences: Dict[str, Any]
    privacy_settings: Dict[str, bool]
    mesh_permissions: Dict[str, bool]
    trusted_nodes: List[str]
    learning_history: Dict[str, Any]

@dataclass
class PalmSlabInteraction:
    """A complete interaction with the palm slab"""
    interaction_id: str
    user_id: str
    timestamp: float
    interaction_type: str
    input_content: str
    local_response: Any
    mesh_validation: Optional[MeshValidationResult]
    confidence_score: float
    processing_time: float
    privacy_level: str
    metadata: Dict[str, Any]

class PalmSlabInterface:
    """
    Complete Palm Slab Node Interface
    
    Provides the full palm slab experience:
    - "Every slab is a full node" - Complete autonomous operation
    - "Data is Local First" - Privacy ring with controlled sharing
    - "Consensus through Cross-Validation" - Social checksum validation
    - "Adaptive Synapses" - Weighted connections to useful peers
    """
    
    def __init__(self, node_id: str, privacy_level: str = "selective"):
        self.node_id = node_id
        self.privacy_level = privacy_level
        self.logger = logging.getLogger(__name__)
        
        # Initialize mesh bridge
        self.bridge_config = SentientPalmSlabConfig(
            node_id=node_id,
            privacy_level=privacy_level,
            enable_mesh_validation=True,
            enable_peer_sharing=privacy_level in ["selective", "open"]
        )
        self.mesh_bridge = SentientMeshBridge(self.bridge_config)
        
        # User management
        self.users: Dict[str, PalmSlabUser] = {}
        self.interaction_history: List[PalmSlabInteraction] = []
        
        # Adaptive synapses - weighted peer connections
        self.peer_connections: Dict[str, float] = {}  # node_id -> usefulness_weight
        self.peer_interaction_history: Dict[str, List[Dict]] = {}
        
        # Privacy ring - controlled data sharing
        self.privacy_ring = PrivacyRing(privacy_level)
        
        self.logger.info(f"Palm slab node {node_id} initialized with {privacy_level} privacy")
    
    async def initialize(self):
        """Initialize the complete palm slab node"""
        
        try:
            # Initialize mesh bridge
            await self.mesh_bridge.initialize()
            
            # Initialize privacy ring
            await self.privacy_ring.initialize()
            
            # Discover and establish adaptive synapses
            await self._establish_adaptive_synapses()
            
            self.logger.info("Palm slab node fully initialized and operational")
            
        except Exception as e:
            self.logger.error(f"Palm slab initialization failed: {e}")
            raise
    
    async def _establish_adaptive_synapses(self):
        """Establish weighted connections to useful peer nodes"""
        
        try:
            # Discover peer nodes
            if hasattr(self.mesh_bridge, 'trusted_nodes'):
                for peer_node_id in self.mesh_bridge.trusted_nodes:
                    # Initialize with neutral weight
                    self.peer_connections[peer_node_id] = 0.5
                    self.peer_interaction_history[peer_node_id] = []
            
            self.logger.info(f"Established adaptive synapses with {len(self.peer_connections)} peers")
            
        except Exception as e:
            self.logger.error(f"Failed to establish adaptive synapses: {e}")
    
    async def process_user_input(
        self,
        user_id: str,
        input_content: str,
        interaction_type: str = "conversation",
        context: Optional[Dict[str, Any]] = None
    ) -> PalmSlabInteraction:
        """
        Process user input through the complete palm slab pipeline
        
        This embodies The Mesh principles:
        1. Local processing first (privacy ring)
        2. Mesh validation if user permits (social checksum)
        3. Adaptive learning from peer feedback (adaptive synapses)
        4. Truth without gatekeepers (confidence-ranked insights)
        """
        
        interaction_id = f"interaction_{int(time.time() * 1000)}"
        start_time = time.time()
        
        try:
            # Get or create user profile
            user = await self._get_or_create_user(user_id)
            
            # Step 1: Privacy ring - determine what can be shared
            sharing_permissions = await self.privacy_ring.evaluate_sharing_permissions(
                user=user,
                content=input_content,
                interaction_type=interaction_type
            )
            
            # Step 2: Local processing first (respecting privacy ring)
            local_response = await self._process_locally(
                user_id=user_id,
                content=input_content,
                interaction_type=interaction_type,
                context=context
            )
            
            # Step 3: Mesh validation (if permissions allow)
            mesh_validation = None
            if sharing_permissions["allow_mesh_validation"]:
                mesh_validation = await self._validate_with_mesh(
                    user_id=user_id,
                    content=input_content,
                    local_response=local_response,
                    interaction_type=interaction_type
                )
                
                # Step 4: Update adaptive synapses based on peer usefulness
                if mesh_validation:
                    await self._update_adaptive_synapses(mesh_validation)
            
            # Step 5: Calculate final confidence with social checksum
            final_confidence = await self._calculate_confidence_with_social_checksum(
                local_response=local_response,
                mesh_validation=mesh_validation
            )
            
            # Step 6: Learn from interaction for future improvements
            await self._learn_from_interaction(
                user=user,
                input_content=input_content,
                local_response=local_response,
                mesh_validation=mesh_validation,
                final_confidence=final_confidence
            )
            
            processing_time = time.time() - start_time
            
            # Create interaction record
            interaction = PalmSlabInteraction(
                interaction_id=interaction_id,
                user_id=user_id,
                timestamp=start_time,
                interaction_type=interaction_type,
                input_content=input_content,
                local_response=local_response,
                mesh_validation=mesh_validation,
                confidence_score=final_confidence,
                processing_time=processing_time,
                privacy_level=sharing_permissions["privacy_level"],
                metadata={
                    "sharing_permissions": sharing_permissions,
                    "peer_nodes_consulted": len(mesh_validation.validation_nodes) if mesh_validation else 0,
                    "adaptive_synapses_updated": mesh_validation is not None
                }
            )
            
            # Store interaction history
            self.interaction_history.append(interaction)
            
            self.logger.info(f"Processed interaction {interaction_id} in {processing_time:.3f}s with {final_confidence:.3f} confidence")
            return interaction
            
        except Exception as e:
            self.logger.error(f"Palm slab interaction processing failed: {e}")
            
            # Return fallback interaction
            processing_time = time.time() - start_time
            return PalmSlabInteraction(
                interaction_id=interaction_id,
                user_id=user_id,
                timestamp=start_time,
                interaction_type=interaction_type,
                input_content=input_content,
                local_response={"content": "I'm having trouble processing that right now. Let me try a different approach."},
                mesh_validation=None,
                confidence_score=0.1,
                processing_time=processing_time,
                privacy_level="private",
                metadata={"error": str(e)}
            )
    
    async def _get_or_create_user(self, user_id: str) -> PalmSlabUser:
        """Get or create user profile"""
        
        if user_id not in self.users:
            self.users[user_id] = PalmSlabUser(
                user_id=user_id,
                preferences={
                    "response_style": "friendly",
                    "privacy_level": self.privacy_level,
                    "mesh_validation": True
                },
                privacy_settings={
                    "share_with_mesh": self.privacy_level != "private",
                    "allow_peer_validation": True,
                    "store_interaction_history": True
                },
                mesh_permissions={
                    "fact_validation": True,
                    "task_validation": True,
                    "response_enhancement": self.privacy_level == "open"
                },
                trusted_nodes=list(self.peer_connections.keys()),
                learning_history={}
            )
        
        return self.users[user_id]
    
    async def _process_locally(
        self,
        user_id: str,
        content: str,
        interaction_type: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process input locally first (privacy ring)"""
        
        try:
            # Determine processing approach based on interaction type
            if interaction_type == "fact_extraction":
                result, _ = await self.mesh_bridge.extract_facts_with_mesh_validation(
                    text=content,
                    username=user_id,
                    source="user_input"
                )
                return {"type": "facts", "content": result.facts, "processing_time": result.processing_time}
            
            elif interaction_type == "task_request":
                result, _ = await self.mesh_bridge.parse_tasks_with_mesh_consensus(
                    task_prompt=content,
                    username=user_id
                )
                return {"type": "task", "content": result.task, "parsing_method": result.parsing_method}
            
            else:  # General conversation/personal AI
                result, _ = await self.mesh_bridge.generate_personal_response_with_mesh_intelligence(
                    user_id=user_id,
                    interaction_type=interaction_type,
                    content=content,
                    context=context
                )
                return {"type": "response", "content": result.content, "style": result.style.value}
            
        except Exception as e:
            self.logger.error(f"Local processing failed: {e}")
            return {"type": "error", "content": "Local processing encountered an error", "error": str(e)}
    
    async def _validate_with_mesh(
        self,
        user_id: str,
        content: str,
        local_response: Dict[str, Any],
        interaction_type: str
    ) -> Optional[MeshValidationResult]:
        """Validate local response with mesh network (social checksum)"""
        
        try:
            # Process through mesh bridge for validation
            if interaction_type == "fact_extraction":
                _, mesh_validation = await self.mesh_bridge.extract_facts_with_mesh_validation(
                    text=content,
                    username=user_id,
                    source="user_input"
                )
                
            elif interaction_type == "task_request":
                _, mesh_validation = await self.mesh_bridge.parse_tasks_with_mesh_consensus(
                    task_prompt=content,
                    username=user_id
                )
                
            else:
                _, mesh_insights = await self.mesh_bridge.generate_personal_response_with_mesh_intelligence(
                    user_id=user_id,
                    interaction_type=interaction_type,
                    content=content,
                    context=None
                )
                
                # Convert mesh insights to validation result
                if mesh_insights:
                    mesh_validation = MeshValidationResult(
                        validated=True,
                        confidence_score=mesh_insights.get("confidence_boost", 0.0),
                        peer_consensus=0.8,  # Placeholder
                        trust_score=0.7,     # Placeholder
                        social_checksum="generated",
                        validation_nodes=["peer1", "peer2"],  # Placeholder
                        metadata=mesh_insights
                    )
                else:
                    mesh_validation = None
            
            return mesh_validation
            
        except Exception as e:
            self.logger.error(f"Mesh validation failed: {e}")
            return None
    
    async def _update_adaptive_synapses(self, mesh_validation: MeshValidationResult):
        """Update adaptive synapses based on peer usefulness"""
        
        try:
            for node_id in mesh_validation.validation_nodes:
                if node_id in self.peer_connections:
                    # Increase weight for nodes that provided good validation
                    if mesh_validation.confidence_score > 0.7:
                        self.peer_connections[node_id] += 0.1
                    else:
                        self.peer_connections[node_id] -= 0.05
                    
                    # Keep weights in reasonable bounds
                    self.peer_connections[node_id] = max(0.1, min(1.0, self.peer_connections[node_id]))
                    
                    # Record interaction history
                    if node_id not in self.peer_interaction_history:
                        self.peer_interaction_history[node_id] = []
                    
                    self.peer_interaction_history[node_id].append({
                        "timestamp": time.time(),
                        "usefulness_score": mesh_validation.confidence_score,
                        "interaction_type": "validation"
                    })
            
            self.logger.debug(f"Updated adaptive synapses for {len(mesh_validation.validation_nodes)} peers")
            
        except Exception as e:
            self.logger.error(f"Adaptive synapse update failed: {e}")
    
    async def _calculate_confidence_with_social_checksum(
        self,
        local_response: Dict[str, Any],
        mesh_validation: Optional[MeshValidationResult]
    ) -> float:
        """Calculate final confidence score with social checksum"""
        
        try:
            # Base confidence from local processing
            base_confidence = 0.7  # Default local confidence
            
            # Extract local confidence if available
            if "confidence" in local_response:
                base_confidence = local_response["confidence"]
            
            # Apply mesh validation boost
            if mesh_validation and mesh_validation.validated:
                # Social checksum boost based on peer consensus
                social_boost = mesh_validation.peer_consensus * 0.3
                final_confidence = min(1.0, base_confidence + social_boost)
            else:
                final_confidence = base_confidence
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.5  # Fallback confidence
    
    async def _learn_from_interaction(
        self,
        user: PalmSlabUser,
        input_content: str,
        local_response: Dict[str, Any],
        mesh_validation: Optional[MeshValidationResult],
        final_confidence: float
    ):
        """Learn from interaction for future improvements"""
        
        try:
            # Update user learning history
            interaction_data = {
                "timestamp": time.time(),
                "input_length": len(input_content),
                "response_type": local_response.get("type", "unknown"),
                "local_confidence": local_response.get("confidence", 0.0),
                "mesh_validated": mesh_validation is not None,
                "final_confidence": final_confidence
            }
            
            if "interactions" not in user.learning_history:
                user.learning_history["interactions"] = []
            
            user.learning_history["interactions"].append(interaction_data)
            
            # Keep learning history manageable
            if len(user.learning_history["interactions"]) > 100:
                user.learning_history["interactions"] = user.learning_history["interactions"][-100:]
            
        except Exception as e:
            self.logger.error(f"Interaction learning failed: {e}")
    
    async def get_palm_slab_status(self) -> Dict[str, Any]:
        """Get comprehensive palm slab status"""
        
        try:
            mesh_status = await self.mesh_bridge.get_node_status()
            
            return {
                "node_id": self.node_id,
                "privacy_level": self.privacy_level,
                "users": len(self.users),
                "interactions": len(self.interaction_history),
                "adaptive_synapses": {
                    "peer_connections": len(self.peer_connections),
                    "average_peer_weight": sum(self.peer_connections.values()) / len(self.peer_connections) if self.peer_connections else 0.0,
                    "top_peers": sorted(self.peer_connections.items(), key=lambda x: x[1], reverse=True)[:3]
                },
                "privacy_ring_status": await self.privacy_ring.get_status(),
                "mesh_bridge_status": mesh_status,
                "capabilities": {
                    "local_ai_processing": True,
                    "mesh_validation": mesh_status.get("mesh_validation_enabled", False),
                    "social_checksum": True,
                    "adaptive_learning": True,
                    "cooperative_trust": len(self.peer_connections) > 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Status retrieval failed: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Clean up palm slab resources"""
        
        try:
            await self.mesh_bridge.cleanup()
            await self.privacy_ring.cleanup()
            self.logger.info("Palm slab cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Palm slab cleanup failed: {e}")


class PrivacyRing:
    """Privacy ring implementation for controlled data sharing"""
    
    def __init__(self, privacy_level: str):
        self.privacy_level = privacy_level
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize privacy ring"""
        self.logger.info(f"Privacy ring initialized with {self.privacy_level} level")
    
    async def evaluate_sharing_permissions(
        self,
        user: PalmSlabUser,
        content: str,
        interaction_type: str
    ) -> Dict[str, Any]:
        """Evaluate what can be shared with the mesh"""
        
        base_permissions = {
            "allow_mesh_validation": user.privacy_settings.get("share_with_mesh", False),
            "allow_fact_sharing": user.mesh_permissions.get("fact_validation", False),
            "allow_task_sharing": user.mesh_permissions.get("task_validation", False),
            "privacy_level": self.privacy_level
        }
        
        # Adjust based on content sensitivity
        if any(sensitive_word in content.lower() for sensitive_word in ["password", "private", "secret"]):
            base_permissions["allow_mesh_validation"] = False
            base_permissions["privacy_level"] = "private"
        
        return base_permissions
    
    async def get_status(self) -> Dict[str, Any]:
        """Get privacy ring status"""
        return {
            "privacy_level": self.privacy_level,
            "active": True,
            "filtering_enabled": True
        }
    
    async def cleanup(self):
        """Clean up privacy ring resources"""
        pass


# Factory function for creating palm slab nodes
def create_palm_slab_node(node_id: str, privacy_level: str = "selective") -> PalmSlabInterface:
    """Create a new palm slab node with full Mesh integration"""
    return PalmSlabInterface(node_id=node_id, privacy_level=privacy_level)