"""
Mesh Network Foundation - P2P Communication Layer

This module provides the foundational networking infrastructure for The Mesh,
enabling peer-to-peer communication, node discovery, and secure message routing
between mesh nodes.

Core Components:
- MeshProtocol: Core P2P protocol implementation
- NodeDiscovery: Automatic peer discovery and connection
- MessageRouter: Intelligent message routing between peers
- ConnectionManager: Manage peer connections and health
- NetworkHealth: Monitor network topology and performance
"""

from .mesh_protocol import MeshProtocol, MeshMessage, MessageType
from .node_discovery import NodeDiscovery, PeerInfo, DiscoveryMethod
from .message_router import MessageRouter, RoutingStrategy
from .connection_manager import ConnectionManager, ConnectionState
from .network_health import NetworkHealth, NetworkMetrics
from .tcp_transport import NetworkTransport, NetworkMessage, TransportType

__all__ = [
    'MeshProtocol',
    'MeshMessage', 
    'MessageType',
    'NodeDiscovery',
    'PeerInfo',
    'DiscoveryMethod',
    'MessageRouter',
    'RoutingStrategy', 
    'ConnectionManager',
    'ConnectionState',
    'NetworkHealth',
    'NetworkMetrics',
    'NetworkTransport',
    'NetworkMessage',
    'TransportType'
]

# Version info
__version__ = "1.0.0"
__author__ = "The Mesh Development Team"
__description__ = "Decentralized P2P networking for The Mesh"