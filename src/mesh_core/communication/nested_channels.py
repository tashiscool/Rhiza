"""
Nested Communication Channels for The Mesh
==========================================

"from the intimacy of family, to the voice of the village, to the chorus of regions, 
to the global tide, to the smaller circles of shared passions."

Implements hierarchical communication layers that mirror human social organization:
- Family: intimate trusted circle (1-8 nodes)
- Village: local community (20-150 nodes) 
- Region: broader geographic/cultural area (500-5000 nodes)
- World: global mesh network (unlimited)
- Chosen Circles: affinity-based groups across all levels

Each layer has different privacy, trust, and consensus requirements.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

class CommunicationScope(Enum):
    """Nested communication scopes - from intimate to global"""
    FAMILY = "family"           # Intimate trusted circle (1-8 nodes)
    VILLAGE = "village"         # Local community (20-150 nodes)  
    REGION = "region"           # Geographic/cultural area (500-5000 nodes)
    WORLD = "world"             # Global mesh network (unlimited)
    CHOSEN = "chosen"           # Affinity circles across levels

class MessagePriority(Enum):
    """Message priority levels"""
    WHISPER = "whisper"         # Private, low priority
    VOICE = "voice"             # Normal communication
    CALL = "call"               # Important, needs attention
    CHORUS = "chorus"           # Broadcast, high visibility
    EMERGENCY = "emergency"     # Crisis, immediate response

class TrustLevel(Enum):
    """Trust levels for different communication scopes"""
    INTIMATE = "intimate"       # Family-level trust (>0.9)
    TRUSTED = "trusted"         # Village-level trust (>0.8)
    RESPECTED = "respected"     # Regional trust (>0.7)
    KNOWN = "known"             # World-level trust (>0.6)
    SPECIALIZED = "specialized" # Chosen circle trust (varies)

@dataclass
class CommunicationChannel:
    """A communication channel within a specific scope"""
    channel_id: str
    scope: CommunicationScope
    name: str
    description: str
    
    # Membership
    members: Set[str] = field(default_factory=set)
    max_members: Optional[int] = None
    min_trust_level: float = 0.6
    
    # Configuration
    privacy_level: str = "selective"  # private, selective, open
    requires_consensus: bool = False
    consensus_threshold: float = 0.6
    message_retention_days: int = 30
    
    # Moderation
    moderators: Set[str] = field(default_factory=set)
    auto_moderation: bool = True
    content_filters: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Set default member limits based on scope
        if self.max_members is None:
            scope_limits = {
                CommunicationScope.FAMILY: 8,
                CommunicationScope.VILLAGE: 150,
                CommunicationScope.REGION: 5000,
                CommunicationScope.WORLD: None,
                CommunicationScope.CHOSEN: 500
            }
            self.max_members = scope_limits[self.scope]

@dataclass
class NestedMessage:
    """A message that can flow through nested communication channels"""
    message_id: str
    content: str
    sender_id: str
    
    # Scope and routing
    origin_scope: CommunicationScope
    target_scopes: List[CommunicationScope]
    channel_ids: List[str]
    
    # Priority and delivery
    priority: MessagePriority
    requires_response: bool = False
    response_deadline: Optional[datetime] = None
    
    # Trust and verification
    trust_verified: bool = False
    mesh_confidence: float = 0.0
    peer_validations: Dict[str, float] = field(default_factory=dict)
    
    # Privacy and filtering
    privacy_level: str = "selective"
    content_filtered: bool = False
    anonymized: bool = False
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    message_type: str = "text"
    attachments: List[str] = field(default_factory=list)
    thread_id: Optional[str] = None
    
    # Status
    delivery_status: Dict[str, str] = field(default_factory=dict)  # node_id -> status
    read_receipts: Dict[str, datetime] = field(default_factory=dict)

class NestedChannelManager:
    """
    Manages hierarchical communication channels across all scopes
    
    Handles message routing, trust verification, and privacy controls
    across the nested communication layers of The Mesh.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.channels: Dict[str, CommunicationChannel] = {}
        self.message_history: Dict[str, NestedMessage] = {}
        self.routing_table: Dict[str, List[str]] = {}  # scope -> channel_ids
        self.trust_scores: Dict[str, float] = {}       # node_id -> trust_score
        
        # Initialize default channels
        self._initialize_default_channels()
    
    def _initialize_default_channels(self):
        """Initialize default channels for each communication scope"""
        
        # Family channel - intimate trusted circle
        family_channel = CommunicationChannel(
            channel_id=f"family_{self.node_id}",
            scope=CommunicationScope.FAMILY,
            name="Family Circle",
            description="Intimate trusted circle for closest relationships",
            privacy_level="private",
            min_trust_level=0.9,
            requires_consensus=False,
            created_by=self.node_id
        )
        
        # Village channel - local community
        village_channel = CommunicationChannel(
            channel_id=f"village_{self.node_id[:8]}",
            scope=CommunicationScope.VILLAGE,
            name="Local Village",
            description="Local community for geographic neighbors",
            privacy_level="selective", 
            min_trust_level=0.8,
            requires_consensus=True,
            consensus_threshold=0.7,
            created_by=self.node_id
        )
        
        # Region channel - broader geographic area
        region_channel = CommunicationChannel(
            channel_id=f"region_{self.node_id[:4]}",
            scope=CommunicationScope.REGION,
            name="Regional Council",
            description="Broader geographic and cultural community",
            privacy_level="selective",
            min_trust_level=0.7,
            requires_consensus=True,
            consensus_threshold=0.6,
            created_by=self.node_id
        )
        
        # World channel - global mesh
        world_channel = CommunicationChannel(
            channel_id="world_global",
            scope=CommunicationScope.WORLD,
            name="Global Chorus",
            description="Global mesh network communication",
            privacy_level="open",
            min_trust_level=0.6,
            requires_consensus=True,
            consensus_threshold=0.5,
            created_by="mesh_system"
        )
        
        # Register channels
        channels = [family_channel, village_channel, region_channel, world_channel]
        for channel in channels:
            self.channels[channel.channel_id] = channel
            
            # Update routing table
            scope_key = channel.scope.value
            if scope_key not in self.routing_table:
                self.routing_table[scope_key] = []
            self.routing_table[scope_key].append(channel.channel_id)
        
        logger.info(f"Initialized {len(channels)} default communication channels")
    
    async def create_chosen_circle(
        self, 
        name: str, 
        description: str, 
        specialization: str,
        initial_members: Optional[List[str]] = None
    ) -> CommunicationChannel:
        """
        Create a chosen circle - affinity-based group across all levels
        
        These are specialized communities based on shared interests, 
        skills, or passions that transcend geographic boundaries.
        """
        
        channel_id = f"chosen_{specialization}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        
        chosen_channel = CommunicationChannel(
            channel_id=channel_id,
            scope=CommunicationScope.CHOSEN,
            name=name,
            description=description,
            privacy_level="selective",
            min_trust_level=0.7,
            requires_consensus=True,
            consensus_threshold=0.6,
            tags=[specialization, "chosen_circle"],
            created_by=self.node_id
        )
        
        # Add initial members
        if initial_members:
            chosen_channel.members.update(initial_members)
        chosen_channel.members.add(self.node_id)  # Creator is always a member
        chosen_channel.moderators.add(self.node_id)  # Creator is initial moderator
        
        # Register channel
        self.channels[channel_id] = chosen_channel
        
        # Update routing table
        if "chosen" not in self.routing_table:
            self.routing_table["chosen"] = []
        self.routing_table["chosen"].append(channel_id)
        
        logger.info(f"Created chosen circle: {name} with specialization: {specialization}")
        return chosen_channel
    
    async def send_message(
        self,
        content: str,
        target_scopes: List[CommunicationScope],
        priority: MessagePriority = MessagePriority.VOICE,
        privacy_level: str = "selective",
        requires_response: bool = False
    ) -> NestedMessage:
        """
        Send a message through the nested communication channels
        
        The message will be routed to appropriate channels based on
        target scopes, trust levels, and privacy requirements.
        """
        
        message_id = f"msg_{self.node_id}_{int(datetime.now().timestamp())}"
        
        # Determine target channels
        target_channels = []
        for scope in target_scopes:
            scope_channels = self.routing_table.get(scope.value, [])
            target_channels.extend(scope_channels)
        
        # Create message
        message = NestedMessage(
            message_id=message_id,
            content=content,
            sender_id=self.node_id,
            origin_scope=target_scopes[0] if target_scopes else CommunicationScope.VILLAGE,
            target_scopes=target_scopes,
            channel_ids=target_channels,
            priority=priority,
            requires_response=requires_response,
            privacy_level=privacy_level
        )
        
        # Validate and route message
        await self._validate_message(message)
        await self._route_message(message)
        
        # Store in history
        self.message_history[message_id] = message
        
        logger.info(f"Sent message {message_id} to {len(target_channels)} channels")
        return message
    
    async def _validate_message(self, message: NestedMessage):
        """Validate message for trust, privacy, and content guidelines"""
        
        # Basic trust validation
        sender_trust = self.trust_scores.get(message.sender_id, 0.5)
        
        # Check if sender meets minimum trust for target scopes
        for scope in message.target_scopes:
            min_trust = self._get_min_trust_for_scope(scope)
            if sender_trust < min_trust:
                logger.warning(f"Sender trust {sender_trust:.2f} below minimum {min_trust:.2f} for {scope.value}")
                message.trust_verified = False
                return
        
        message.trust_verified = True
        message.mesh_confidence = sender_trust
    
    async def _route_message(self, message: NestedMessage):
        """Route message to appropriate channels based on scopes and trust"""
        
        for channel_id in message.channel_ids:
            if channel_id not in self.channels:
                continue
                
            channel = self.channels[channel_id]
            
            # Check if sender is member or meets trust requirements
            if not await self._can_send_to_channel(message.sender_id, channel):
                message.delivery_status[channel_id] = "access_denied"
                continue
            
            # Apply privacy and filtering
            filtered_content = await self._filter_content(message.content, channel)
            
            # Route to channel members
            await self._deliver_to_channel_members(message, channel, filtered_content)
            message.delivery_status[channel_id] = "delivered"
    
    async def _can_send_to_channel(self, sender_id: str, channel: CommunicationChannel) -> bool:
        """Check if sender can send messages to channel"""
        
        # Check membership
        if sender_id in channel.members:
            return True
        
        # Check trust level
        sender_trust = self.trust_scores.get(sender_id, 0.5)
        if sender_trust >= channel.min_trust_level:
            return True
        
        return False
    
    async def _filter_content(self, content: str, channel: CommunicationChannel) -> str:
        """Apply content filtering based on channel settings"""
        
        if not channel.auto_moderation:
            return content
        
        # Apply content filters (simplified)
        filtered_content = content
        for filter_rule in channel.content_filters:
            # In real implementation, would apply sophisticated content filtering
            pass
        
        return filtered_content
    
    async def _deliver_to_channel_members(
        self, 
        message: NestedMessage, 
        channel: CommunicationChannel,
        filtered_content: str
    ):
        """Deliver message to channel members"""
        
        for member_id in channel.members:
            if member_id == message.sender_id:
                continue  # Don't deliver to sender
                
            # In real implementation, would send via network
            logger.debug(f"Delivering message {message.message_id} to {member_id} in {channel.name}")
    
    def _get_min_trust_for_scope(self, scope: CommunicationScope) -> float:
        """Get minimum trust level required for communication scope"""
        
        trust_requirements = {
            CommunicationScope.FAMILY: 0.9,    # Intimate trust
            CommunicationScope.VILLAGE: 0.8,   # Community trust
            CommunicationScope.REGION: 0.7,    # Regional trust
            CommunicationScope.WORLD: 0.6,     # Basic trust
            CommunicationScope.CHOSEN: 0.7     # Specialized trust
        }
        
        return trust_requirements.get(scope, 0.6)
    
    async def join_channel(self, channel_id: str, node_id: Optional[str] = None) -> bool:
        """Join a communication channel"""
        
        if node_id is None:
            node_id = self.node_id
            
        if channel_id not in self.channels:
            logger.error(f"Channel {channel_id} not found")
            return False
        
        channel = self.channels[channel_id]
        
        # Check trust requirements
        node_trust = self.trust_scores.get(node_id, 0.5)
        if node_trust < channel.min_trust_level:
            logger.warning(f"Node {node_id} trust {node_trust:.2f} below required {channel.min_trust_level:.2f}")
            return False
        
        # Check member limit
        if channel.max_members and len(channel.members) >= channel.max_members:
            logger.warning(f"Channel {channel.name} is full ({channel.max_members} members)")
            return False
        
        # Add to channel
        channel.members.add(node_id)
        logger.info(f"Node {node_id} joined channel {channel.name}")
        return True
    
    async def leave_channel(self, channel_id: str, node_id: Optional[str] = None) -> bool:
        """Leave a communication channel"""
        
        if node_id is None:
            node_id = self.node_id
            
        if channel_id not in self.channels:
            return False
        
        channel = self.channels[channel_id]
        if node_id in channel.members:
            channel.members.remove(node_id)
            
            # Remove from moderators if applicable
            if node_id in channel.moderators:
                channel.moderators.discard(node_id)
            
            logger.info(f"Node {node_id} left channel {channel.name}")
            return True
        
        return False
    
    def get_channels_for_scope(self, scope: CommunicationScope) -> List[CommunicationChannel]:
        """Get all channels for a specific communication scope"""
        
        return [
            channel for channel in self.channels.values() 
            if channel.scope == scope
        ]
    
    def get_node_channels(self, node_id: Optional[str] = None) -> List[CommunicationChannel]:
        """Get all channels a node is a member of"""
        
        if node_id is None:
            node_id = self.node_id
            
        return [
            channel for channel in self.channels.values()
            if node_id in channel.members
        ]
    
    def update_trust_score(self, node_id: str, trust_score: float):
        """Update trust score for a node"""
        self.trust_scores[node_id] = max(0.0, min(1.0, trust_score))
        logger.debug(f"Updated trust score for {node_id}: {trust_score:.3f}")
    
    def get_communication_summary(self) -> Dict[str, Any]:
        """Get summary of all communication channels and activity"""
        
        summary = {
            'node_id': self.node_id,
            'timestamp': datetime.now().isoformat(),
            'total_channels': len(self.channels),
            'channels_by_scope': {},
            'total_messages': len(self.message_history),
            'active_conversations': 0,
            'trust_network_size': len(self.trust_scores)
        }
        
        # Count channels by scope
        for channel in self.channels.values():
            scope = channel.scope.value
            if scope not in summary['channels_by_scope']:
                summary['channels_by_scope'][scope] = {
                    'count': 0,
                    'total_members': 0,
                    'average_trust': 0.0
                }
            
            summary['channels_by_scope'][scope]['count'] += 1
            summary['channels_by_scope'][scope]['total_members'] += len(channel.members)
        
        # Calculate active conversations (messages in last 24 hours)
        recent_threshold = datetime.now() - timedelta(hours=24)
        summary['active_conversations'] = len([
            msg for msg in self.message_history.values()
            if msg.timestamp >= recent_threshold
        ])
        
        return summary
    
    def export_channel_configuration(self) -> Dict[str, Any]:
        """Export channel configuration for backup/sync"""
        
        return {
            'node_id': self.node_id,
            'channels': {
                channel_id: {
                    'scope': channel.scope.value,
                    'name': channel.name,
                    'description': channel.description,
                    'members': list(channel.members),
                    'moderators': list(channel.moderators),
                    'privacy_level': channel.privacy_level,
                    'min_trust_level': channel.min_trust_level,
                    'tags': channel.tags
                }
                for channel_id, channel in self.channels.items()
            },
            'routing_table': self.routing_table,
            'export_timestamp': datetime.now().isoformat()
        }