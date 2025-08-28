"""
Sync Manager - Coordination of Data Synchronization

Orchestrates distributed data synchronization across The Mesh network,
managing consistency, conflict resolution, and privacy-aware data sharing
with intelligent coordination algorithms.

Key Features:
- Multi-phase synchronization protocol
- Vector clock-based consistency
- Privacy-aware synchronization policies  
- Bandwidth-adaptive sync strategies
- Conflict-aware merge coordination
"""

import asyncio
import time
import json
import secrets
import hashlib
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from enum import Enum
from collections import defaultdict, deque
import logging

try:
    from mesh_core.sync.data_chunker import DataChunk, DataChunker, ChunkType, PrivacyLevel
    from mesh_core.sync.peer_relay import PeerRelaySystem, TransferRequest, RelayStrategy
    from mesh_core.trust.trust_ledger import TrustLedger, TrustScore
except ImportError:
    # Fallback to relative imports
    try:
        from .data_chunker import DataChunk, DataChunker, ChunkType, PrivacyLevel
        from .peer_relay import PeerRelaySystem, TransferRequest, RelayStrategy
        from ..trust.trust_ledger import TrustLedger, TrustScore
    except ImportError:
        # Mock classes for testing
        class DataChunk:
            def __init__(self):
                pass
        
        class DataChunker:
            def __init__(self):
                pass
        
        class ChunkType:
            pass
        
        class PrivacyLevel:
            pass
        
        class PeerRelaySystem:
            def __init__(self, node_id):
                pass
        
        class TransferRequest:
            pass
        
        class RelayStrategy:
            pass
        
        class TrustLedger:
            def __init__(self, node_id):
                pass
        
        class TrustScore:
            pass


class SyncPhase(Enum):
    """Phases of synchronization protocol"""
    DISCOVERY = "discovery"        # Peer discovery and capability exchange
    NEGOTIATION = "negotiation"    # Sync policy negotiation
    PREPARATION = "preparation"    # Data preparation and chunking
    TRANSFER = "transfer"         # Actual data transfer
    VERIFICATION = "verification"  # Integrity and consistency verification
    FINALIZATION = "finalization" # Commit and cleanup


class SyncStrategy(Enum):
    """Synchronization strategies"""
    FULL_SYNC = "full_sync"              # Complete data synchronization
    INCREMENTAL = "incremental"          # Only changed data
    SELECTIVE = "selective"              # User-specified data subsets
    PRIORITY_BASED = "priority_based"    # High-priority data first
    BANDWIDTH_ADAPTIVE = "bandwidth_adaptive"  # Adapt to available bandwidth


class ConsistencyLevel(Enum):
    """Data consistency requirements"""
    EVENTUAL = "eventual"        # Eventually consistent
    STRONG = "strong"           # Strong consistency required
    SESSION = "session"         # Session consistency
    MONOTONIC = "monotonic"     # Monotonic consistency


@dataclass
class VectorClock:
    """Vector clock for distributed consistency"""
    node_id: str
    clock: Dict[str, int] = field(default_factory=dict)
    
    def increment(self, node_id: Optional[str] = None):
        """Increment clock for specified node"""
        target_node = node_id or self.node_id
        self.clock[target_node] = self.clock.get(target_node, 0) + 1
        
    def update(self, other_clock: 'VectorClock'):
        """Update clock with another vector clock"""
        for node, timestamp in other_clock.clock.items():
            self.clock[node] = max(self.clock.get(node, 0), timestamp)
        self.increment()  # Increment local clock
        
    def compare(self, other_clock: 'VectorClock') -> str:
        """Compare with another vector clock"""
        self_greater = False
        other_greater = False
        
        all_nodes = set(self.clock.keys()) | set(other_clock.clock.keys())
        
        for node in all_nodes:
            self_val = self.clock.get(node, 0)
            other_val = other_clock.clock.get(node, 0)
            
            if self_val > other_val:
                self_greater = True
            elif other_val > self_val:
                other_greater = True
                
        if self_greater and not other_greater:
            return "after"
        elif other_greater and not self_greater:
            return "before"
        elif not self_greater and not other_greater:
            return "equal"
        else:
            return "concurrent"


@dataclass
class SyncItem:
    """Item to be synchronized"""
    item_id: str
    data_type: str
    chunk_ids: List[str]
    vector_clock: VectorClock
    privacy_level: PrivacyLevel
    priority: int = 5
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncPolicy:
    """Synchronization policy configuration"""
    sync_strategy: SyncStrategy
    consistency_level: ConsistencyLevel
    max_chunk_size: int = 1024 * 1024  # 1MB
    bandwidth_limit: Optional[int] = None  # Bytes per second
    privacy_filters: List[PrivacyLevel] = field(default_factory=lambda: [PrivacyLevel.ANONYMOUS])
    conflict_resolution: str = "merge"  # merge, overwrite, manual
    max_sync_time: float = 3600  # 1 hour timeout
    retry_attempts: int = 3


@dataclass
class SyncSession:
    """Active synchronization session"""
    session_id: str
    initiator_peer: str
    participant_peers: List[str]
    sync_policy: SyncPolicy
    current_phase: SyncPhase
    items_to_sync: Dict[str, SyncItem]
    sync_progress: Dict[str, float] = field(default_factory=dict)
    vector_clock: VectorClock = field(default_factory=lambda: VectorClock(""))
    started_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    errors: List[str] = field(default_factory=list)


class SyncCoordinator:
    """Coordinates synchronization between multiple peers"""
    
    def __init__(self, local_peer_id: str, 
                 peer_relay: PeerRelaySystem,
                 trust_ledger: TrustLedger,
                 data_chunker: DataChunker):
        self.local_peer_id = local_peer_id
        self.peer_relay = peer_relay
        self.trust_ledger = trust_ledger
        self.data_chunker = data_chunker
        
        self.active_sessions: Dict[str, SyncSession] = {}
        self.sync_policies: Dict[str, SyncPolicy] = {}
        self.sync_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
        
        # Callbacks for sync events
        self.on_sync_started: List[Callable] = []
        self.on_sync_completed: List[Callable] = []
        self.on_conflict_detected: List[Callable] = []
        
        # Configuration
        self.max_concurrent_sessions = 5
        self.session_timeout = 3600  # 1 hour
        self.heartbeat_interval = 30  # 30 seconds
        
    async def start(self):
        """Start the sync coordinator"""
        self.logger.info(f"Starting sync coordinator for {self.local_peer_id}")
        
        # Start background tasks
        asyncio.create_task(self._session_monitor())
        asyncio.create_task(self._sync_processor())
        
    def create_sync_policy(self, policy_name: str, **kwargs) -> SyncPolicy:
        """Create and register a synchronization policy"""
        policy = SyncPolicy(**kwargs)
        self.sync_policies[policy_name] = policy
        return policy
        
    async def initiate_sync(self, 
                          target_peers: List[str],
                          items: List[SyncItem],
                          policy_name: str = "default") -> str:
        """Initiate synchronization with target peers"""
        
        # Check concurrent session limit
        if len(self.active_sessions) >= self.max_concurrent_sessions:
            raise RuntimeError("Maximum concurrent sync sessions reached")
            
        # Get sync policy
        policy = self.sync_policies.get(policy_name)
        if not policy:
            policy = SyncPolicy(
                sync_strategy=SyncStrategy.INCREMENTAL,
                consistency_level=ConsistencyLevel.EVENTUAL
            )
            
        # Create sync session
        session_id = secrets.token_hex(16)
        session = SyncSession(
            session_id=session_id,
            initiator_peer=self.local_peer_id,
            participant_peers=target_peers,
            sync_policy=policy,
            current_phase=SyncPhase.DISCOVERY,
            items_to_sync={item.item_id: item for item in items},
            vector_clock=VectorClock(self.local_peer_id)
        )
        
        self.active_sessions[session_id] = session
        
        # Start sync process
        asyncio.create_task(self._execute_sync_session(session))
        
        # Notify callbacks
        for callback in self.on_sync_started:
            try:
                await callback(session)
            except Exception as e:
                self.logger.error(f"Sync started callback error: {e}")
                
        self.logger.info(f"Initiated sync session {session_id} with {len(target_peers)} peers")
        return session_id
        
    async def join_sync_session(self, session_info: Dict[str, Any]) -> bool:
        """Join an existing sync session as participant"""
        try:
            session_id = session_info['session_id']
            initiator = session_info['initiator_peer']
            
            # Create local session representation
            session = SyncSession(
                session_id=session_id,
                initiator_peer=initiator,
                participant_peers=[self.local_peer_id],
                sync_policy=SyncPolicy(**session_info.get('policy', {})),
                current_phase=SyncPhase.DISCOVERY,
                items_to_sync={},
                vector_clock=VectorClock(self.local_peer_id)
            )
            
            self.active_sessions[session_id] = session
            
            # Start participant process
            asyncio.create_task(self._participate_in_sync(session))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to join sync session: {e}")
            return False
            
    def get_sync_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a sync session"""
        session = self.active_sessions.get(session_id)
        if not session:
            return None
            
        total_items = len(session.items_to_sync)
        completed_items = sum(1 for progress in session.sync_progress.values() if progress >= 1.0)
        
        return {
            'session_id': session_id,
            'phase': session.current_phase.value,
            'progress': completed_items / max(total_items, 1),
            'items_total': total_items,
            'items_completed': completed_items,
            'participants': len(session.participant_peers),
            'started_at': session.started_at,
            'last_activity': session.last_activity,
            'errors': session.errors
        }
        
    async def cancel_sync_session(self, session_id: str) -> bool:
        """Cancel an active sync session"""
        if session_id not in self.active_sessions:
            return False
            
        session = self.active_sessions[session_id]
        
        # Cancel any active transfers
        for item_id in session.items_to_sync:
            # This would cancel transfers in the peer relay system
            pass
            
        # Remove session
        del self.active_sessions[session_id]
        
        self.logger.info(f"Cancelled sync session {session_id}")
        return True
        
    async def _execute_sync_session(self, session: SyncSession):
        """Execute complete sync session as initiator"""
        try:
            session.last_activity = time.time()
            
            # Phase 1: Discovery
            session.current_phase = SyncPhase.DISCOVERY
            await self._phase_discovery(session)
            
            # Phase 2: Negotiation
            session.current_phase = SyncPhase.NEGOTIATION
            await self._phase_negotiation(session)
            
            # Phase 3: Preparation
            session.current_phase = SyncPhase.PREPARATION
            await self._phase_preparation(session)
            
            # Phase 4: Transfer
            session.current_phase = SyncPhase.TRANSFER
            await self._phase_transfer(session)
            
            # Phase 5: Verification
            session.current_phase = SyncPhase.VERIFICATION
            await self._phase_verification(session)
            
            # Phase 6: Finalization
            session.current_phase = SyncPhase.FINALIZATION
            await self._phase_finalization(session)
            
            # Record successful completion
            self._record_sync_completion(session, True)
            
            # Notify callbacks
            for callback in self.on_sync_completed:
                try:
                    await callback(session)
                except Exception as e:
                    self.logger.error(f"Sync completed callback error: {e}")
                    
        except Exception as e:
            self.logger.error(f"Sync session {session.session_id} failed: {e}")
            session.errors.append(str(e))
            self._record_sync_completion(session, False)
            
        finally:
            # Clean up session
            if session.session_id in self.active_sessions:
                del self.active_sessions[session.session_id]
                
    async def _participate_in_sync(self, session: SyncSession):
        """Participate in sync session as non-initiator"""
        try:
            # Wait for instructions from initiator
            # This would implement the participant side of each phase
            await self._wait_for_phase_instructions(session)
            
        except Exception as e:
            self.logger.error(f"Sync participation failed: {e}")
            session.errors.append(str(e))
            
    async def _phase_discovery(self, session: SyncSession):
        """Discovery phase: Find available peers and capabilities"""
        self.logger.info(f"Starting discovery phase for session {session.session_id}")
        
        # Query peer capabilities
        available_peers = []
        for peer_id in session.participant_peers:
            if peer_id in self.peer_relay.peer_capabilities:
                capability = self.peer_relay.peer_capabilities[peer_id]
                
                # Check if peer meets sync requirements
                if self._peer_meets_sync_requirements(capability, session.sync_policy):
                    available_peers.append(peer_id)
                    
        # Update participant list with available peers
        session.participant_peers = available_peers
        
        if not available_peers:
            raise RuntimeError("No suitable peers available for synchronization")
            
        self.logger.info(f"Discovered {len(available_peers)} suitable peers")
        
    async def _phase_negotiation(self, session: SyncSession):
        """Negotiation phase: Agree on sync parameters"""
        self.logger.info(f"Starting negotiation phase for session {session.session_id}")
        
        # This would implement policy negotiation with participants
        # For now, assume all participants accept the initiator's policy
        
        # Exchange vector clocks with participants
        for peer_id in session.participant_peers:
            # This would exchange vector clocks
            pass
            
    async def _phase_preparation(self, session: SyncSession):
        """Preparation phase: Prepare data for synchronization"""
        self.logger.info(f"Starting preparation phase for session {session.session_id}")
        
        # Chunk data items that need synchronization
        for item_id, sync_item in session.items_to_sync.items():
            try:
                # This would prepare the actual data for chunking
                # For now, simulate preparation
                session.sync_progress[item_id] = 0.1  # 10% prepared
                
            except Exception as e:
                self.logger.error(f"Failed to prepare item {item_id}: {e}")
                session.errors.append(f"Preparation failed for {item_id}: {e}")
                
    async def _phase_transfer(self, session: SyncSession):
        """Transfer phase: Actual data transfer"""
        self.logger.info(f"Starting transfer phase for session {session.session_id}")
        
        # Create transfer requests for each sync item
        transfer_tasks = []
        
        for item_id, sync_item in session.items_to_sync.items():
            # Create transfer request for each chunk
            for chunk_id in sync_item.chunk_ids:
                request = TransferRequest(
                    request_id=f"{session.session_id}:{chunk_id}",
                    chunk_id=chunk_id,
                    source_peer=self.local_peer_id,
                    target_peers=session.participant_peers,
                    priority=sync_item.priority,
                    privacy_level=sync_item.privacy_level,
                    timeout=session.sync_policy.max_sync_time / len(sync_item.chunk_ids)
                )
                
                task = asyncio.create_task(self._transfer_chunk(request, session))
                transfer_tasks.append(task)
                
        # Wait for all transfers to complete
        if transfer_tasks:
            results = await asyncio.gather(*transfer_tasks, return_exceptions=True)
            
            # Update progress based on results
            successful_transfers = sum(1 for result in results if result is True)
            total_transfers = len(transfer_tasks)
            
            for item_id in session.items_to_sync:
                session.sync_progress[item_id] = 0.8  # 80% transferred
                
    async def _phase_verification(self, session: SyncSession):
        """Verification phase: Verify integrity and consistency"""
        self.logger.info(f"Starting verification phase for session {session.session_id}")
        
        # Verify data integrity on all participants
        verification_tasks = []
        
        for peer_id in session.participant_peers:
            task = asyncio.create_task(self._verify_peer_data(peer_id, session))
            verification_tasks.append(task)
            
        if verification_tasks:
            results = await asyncio.gather(*verification_tasks, return_exceptions=True)
            
            # Check if all verifications passed
            failed_verifications = [r for r in results if r is not True]
            if failed_verifications:
                session.errors.extend([str(f) for f in failed_verifications])
                
        # Update progress
        for item_id in session.items_to_sync:
            session.sync_progress[item_id] = 0.95  # 95% verified
            
    async def _phase_finalization(self, session: SyncSession):
        """Finalization phase: Commit changes and cleanup"""
        self.logger.info(f"Starting finalization phase for session {session.session_id}")
        
        # Commit synchronized data
        for item_id, sync_item in session.items_to_sync.items():
            try:
                # This would commit the synchronized data
                session.sync_progress[item_id] = 1.0  # 100% complete
                
            except Exception as e:
                self.logger.error(f"Failed to finalize item {item_id}: {e}")
                session.errors.append(f"Finalization failed for {item_id}: {e}")
                
        self.logger.info(f"Sync session {session.session_id} completed successfully")
        
    async def _transfer_chunk(self, request: TransferRequest, session: SyncSession) -> bool:
        """Transfer a single chunk"""
        try:
            # Use peer relay system to transfer chunk
            request_id = await self.peer_relay.request_transfer(request)
            
            # Wait for transfer completion
            while True:
                status = self.peer_relay.get_transfer_status(request_id)
                if not status:
                    return False
                    
                if status.status.value in ['completed', 'failed', 'cancelled']:
                    return status.status.value == 'completed'
                    
                await asyncio.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Chunk transfer failed: {e}")
            return False
            
    async def _verify_peer_data(self, peer_id: str, session: SyncSession) -> bool:
        """Verify data integrity on a peer"""
        try:
            # This would implement data verification protocol
            # For now, simulate verification
            await asyncio.sleep(0.1)
            return True
            
        except Exception as e:
            self.logger.error(f"Peer {peer_id} verification failed: {e}")
            return False
            
    def _peer_meets_sync_requirements(self, capability, policy: SyncPolicy) -> bool:
        """Check if peer meets synchronization requirements"""
        # Check bandwidth requirements
        if policy.bandwidth_limit and capability.bandwidth < policy.bandwidth_limit:
            return False
            
        # Check privacy level support
        if not any(level in capability.supported_privacy_levels for level in policy.privacy_filters):
            return False
            
        # Check trust requirements
        trust_score = self.trust_ledger.get_trust_score(capability.peer_id)
        if trust_score and trust_score.composite_score < 0.5:  # Minimum trust threshold
            return False
            
        return True
        
    async def _wait_for_phase_instructions(self, session: SyncSession):
        """Wait for and execute phase instructions as participant"""
        # This would implement the participant side of the protocol
        pass
        
    def _record_sync_completion(self, session: SyncSession, success: bool):
        """Record sync session completion"""
        completion_record = {
            'session_id': session.session_id,
            'success': success,
            'duration': time.time() - session.started_at,
            'participants': len(session.participant_peers),
            'items_synced': len(session.items_to_sync),
            'errors': session.errors,
            'completed_at': time.time()
        }
        
        self.sync_history.append(completion_record)
        
        # Keep only recent history
        if len(self.sync_history) > 1000:
            self.sync_history = self.sync_history[-1000:]
            
    async def _session_monitor(self):
        """Monitor active sessions for timeouts and health"""
        while True:
            try:
                current_time = time.time()
                
                # Check for timed out sessions
                for session_id, session in list(self.active_sessions.items()):
                    if current_time - session.last_activity > self.session_timeout:
                        self.logger.warning(f"Session {session_id} timed out")
                        session.errors.append("Session timeout")
                        await self.cancel_sync_session(session_id)
                        
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Error in session monitor: {e}")
                await asyncio.sleep(self.heartbeat_interval)
                
    async def _sync_processor(self):
        """Background processor for sync operations"""
        while True:
            try:
                # Process any queued sync operations
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in sync processor: {e}")
                await asyncio.sleep(1)
                
    def get_sync_statistics(self) -> Dict[str, Any]:
        """Get comprehensive sync statistics"""
        if not self.sync_history:
            return {'total_sessions': 0}
            
        recent_sessions = [s for s in self.sync_history if time.time() - s['completed_at'] < 86400]  # Last 24h
        successful_sessions = [s for s in recent_sessions if s['success']]
        
        return {
            'total_sessions': len(self.sync_history),
            'active_sessions': len(self.active_sessions),
            'recent_sessions_24h': len(recent_sessions),
            'success_rate_24h': len(successful_sessions) / max(len(recent_sessions), 1),
            'avg_session_duration': sum(s['duration'] for s in recent_sessions) / max(len(recent_sessions), 1),
            'avg_participants': sum(s['participants'] for s in recent_sessions) / max(len(recent_sessions), 1),
            'total_items_synced': sum(s['items_synced'] for s in self.sync_history),
            'common_errors': self._get_common_errors()
        }
        
    def _get_common_errors(self) -> Dict[str, int]:
        """Get most common sync errors"""
        error_counts = defaultdict(int)
        
        for session in self.sync_history[-100:]:  # Last 100 sessions
            for error in session['errors']:
                error_counts[error] += 1
                
        return dict(sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10])