"""
Distributed Data Synchronization Module

This module provides comprehensive distributed data synchronization capabilities
for The Mesh network, including data chunking, peer-to-peer relay protocols,
synchronization coordination, conflict resolution, and privacy protection.

Core Components:
- DataChunker: Anonymized data chunking with privacy protection
- PeerRelaySystem: Intelligent peer-to-peer data relay and routing
- SyncCoordinator: Multi-phase synchronization coordination
- ConflictResolver: Sophisticated conflict resolution mechanisms
- PrivacyFilterSystem: Multi-level privacy protection and filtering

Usage Example:
    from mesh_core.sync import (
        DataChunker, PeerRelaySystem, SyncCoordinator, 
        ConflictResolver, PrivacyFilterSystem
    )
    
    # Initialize components
    chunker = DataChunker()
    relay = PeerRelaySystem("local_peer_id")
    sync_coord = SyncCoordinator("local_peer_id", relay, trust_ledger, chunker)
    conflict_resolver = ConflictResolver(trust_ledger, "local_peer_id")
    privacy_filter = PrivacyFilterSystem("local_peer_id", trust_ledger)
    
    # Start systems
    await relay.start()
    await sync_coord.start()
    await conflict_resolver.start()
"""

from .data_chunker import (
    DataChunker,
    DataChunk,
    ChunkType,
    PrivacyLevel,
    ChunkMetadata,
    ChunkingStrategy,
    RabinChunking,
    PrivacyProtector
)

from .peer_relay import (
    PeerRelaySystem,
    PeerCapability,
    RelayStrategy,
    TransferRequest,
    TransferProgress,
    TransferStatus,
    RelayRoute,
    RelayHop,
    RelayMetrics
)

from .sync_manager import (
    SyncCoordinator,
    SyncSession,
    SyncPolicy,
    SyncStrategy,
    SyncItem,
    ConsistencyLevel,
    SyncPhase,
    VectorClock
)

from .conflict_resolver import (
    ConflictResolver,
    DataConflict,
    ConflictingData,
    ConflictType,
    ResolutionStrategy,
    ResolutionResult,
    ConflictSeverity,
    SemanticAnalyzer,
    TrustBasedResolver
)

from .privacy_filter import (
    PrivacyFilterSystem,
    PrivacyRule,
    PrivacyContext,
    FilterDecision,
    PrivacyPolicy,
    DataSensitivity,
    FilterAction,
    DataAnonymizer
)

__all__ = [
    # Data Chunker
    'DataChunker',
    'DataChunk', 
    'ChunkType',
    'PrivacyLevel',
    'ChunkMetadata',
    'ChunkingStrategy',
    'RabinChunking',
    'PrivacyProtector',
    
    # Peer Relay
    'PeerRelaySystem',
    'PeerCapability',
    'RelayStrategy', 
    'TransferRequest',
    'TransferProgress',
    'TransferStatus',
    'RelayRoute',
    'RelayHop',
    'RelayMetrics',
    
    # Sync Manager
    'SyncCoordinator',
    'SyncSession',
    'SyncPolicy',
    'SyncStrategy',
    'SyncItem',
    'ConsistencyLevel',
    'SyncPhase',
    'VectorClock',
    
    # Conflict Resolver
    'ConflictResolver',
    'DataConflict',
    'ConflictingData',
    'ConflictType',
    'ResolutionStrategy',
    'ResolutionResult',
    'ConflictSeverity',
    'SemanticAnalyzer',
    'TrustBasedResolver',
    
    # Privacy Filter
    'PrivacyFilterSystem',
    'PrivacyRule',
    'PrivacyContext',
    'FilterDecision',
    'PrivacyPolicy',
    'DataSensitivity', 
    'FilterAction',
    'DataAnonymizer'
]

# Version information
__version__ = "1.0.0"
__author__ = "The Mesh Development Team"
__description__ = "Distributed data synchronization with privacy protection for The Mesh"