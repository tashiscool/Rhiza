"""
Nested Communication System for The Mesh
========================================

Implements hierarchical communication channels that mirror human social organization:

"from the intimacy of family, to the voice of the village, to the chorus of regions,
to the global tide, to the smaller circles of shared passions."

Communication Scopes:
- FAMILY: Intimate trusted circle (1-8 nodes) - "whispers at the hearth"
- VILLAGE: Local community (20-150 nodes) - "songs of the village"  
- REGION: Geographic/cultural area (500-5000 nodes) - "councils of regions"
- WORLD: Global mesh network (unlimited) - "the great chorus of the world"
- CHOSEN: Affinity circles across levels - "secret circles of shared obsession"

Each scope has different privacy, trust, and consensus requirements.
Messages can flow through the hierarchy with intelligent routing and escalation.
"""

from .nested_channels import (
    CommunicationScope,
    MessagePriority, 
    TrustLevel,
    CommunicationChannel,
    NestedMessage,
    NestedChannelManager
)

from .message_router import (
    RoutingStrategy,
    MessageFlow,
    RoutingRule,
    HierarchicalMessageRouter
)

__all__ = [
    # Core communication types
    'CommunicationScope',
    'MessagePriority',
    'TrustLevel',
    'CommunicationChannel', 
    'NestedMessage',
    'NestedChannelManager',
    
    # Message routing
    'RoutingStrategy',
    'MessageFlow', 
    'RoutingRule',
    'HierarchicalMessageRouter'
]