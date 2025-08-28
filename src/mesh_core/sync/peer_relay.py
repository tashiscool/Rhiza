"""
Peer Relay - Peer-to-Peer Data Sharing Protocols

Implements efficient peer-to-peer data synchronization with intelligent
relay selection, bandwidth optimization, and trust-based routing for
The Mesh network.

Key Features:
- Multi-hop relay routing with trust weighting
- Bandwidth-aware chunk distribution
- Redundant transmission with error recovery
- Load balancing across peer network
- Privacy-preserving relay protocols
"""

import asyncio
import time
import secrets
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, AsyncGenerator
from enum import Enum
import heapq
from collections import defaultdict, deque
import logging

from .data_chunker import DataChunk, ChunkType, PrivacyLevel


class RelayStrategy(Enum):
    """Strategies for peer relay selection"""
    DIRECT = "direct"              # Direct peer-to-peer
    MULTI_HOP = "multi_hop"        # Multi-hop routing
    BROADCAST = "broadcast"         # Broadcast to multiple peers
    TRUST_WEIGHTED = "trust_weighted"  # Trust-based routing
    BANDWIDTH_OPTIMAL = "bandwidth_optimal"  # Bandwidth optimization


class TransferStatus(Enum):
    """Status of data transfer"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PeerCapability:
    """Capability information for a peer"""
    peer_id: str
    bandwidth: int          # Bytes per second
    latency: float         # Round-trip time in seconds
    reliability: float     # Success rate (0.0-1.0)
    trust_score: float     # Trust score from trust system
    storage_available: int  # Available storage in bytes
    cpu_load: float        # CPU utilization (0.0-1.0)
    online_since: float    # Timestamp when peer came online
    supported_privacy_levels: Set[PrivacyLevel]


@dataclass 
class RelayHop:
    """Single hop in a relay chain"""
    peer_id: str
    expected_latency: float
    expected_reliability: float
    bandwidth_allocation: int
    trust_weight: float


@dataclass
class RelayRoute:
    """Complete relay route for data transfer"""
    route_id: str
    source_peer: str
    target_peer: str
    hops: List[RelayHop]
    total_latency: float
    total_reliability: float
    min_bandwidth: int
    route_cost: float       # Combined cost metric
    privacy_compatible: bool


@dataclass
class TransferRequest:
    """Request for data transfer between peers"""
    request_id: str
    chunk_id: str
    source_peer: str
    target_peers: List[str]
    priority: int = 5       # 1=highest, 10=lowest
    max_hops: int = 3
    min_reliability: float = 0.8
    privacy_level: PrivacyLevel = PrivacyLevel.ANONYMOUS
    timeout: float = 30.0
    created_at: float = field(default_factory=time.time)


@dataclass
class TransferProgress:
    """Progress tracking for data transfer"""
    request_id: str
    chunk_id: str
    status: TransferStatus
    bytes_transferred: int
    total_bytes: int
    transfer_rate: float    # Bytes per second
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    active_routes: List[str] = field(default_factory=list)


class RelayMetrics:
    """Metrics collection for relay performance"""
    
    def __init__(self):
        self.transfer_stats = defaultdict(list)
        self.route_performance = defaultdict(list)
        self.peer_reliability = defaultdict(list)
        self.bandwidth_utilization = defaultdict(list)
        
    def record_transfer(self, request_id: str, progress: TransferProgress):
        """Record transfer completion metrics"""
        if progress.status == TransferStatus.COMPLETED and progress.started_at:
            duration = progress.completed_at - progress.started_at
            rate = progress.bytes_transferred / max(duration, 0.001)
            
            self.transfer_stats['duration'].append(duration)
            self.transfer_stats['rate'].append(rate)
            self.transfer_stats['size'].append(progress.bytes_transferred)
            
    def record_route_performance(self, route_id: str, actual_latency: float, success: bool):
        """Record route performance metrics"""
        self.route_performance[route_id].append({
            'latency': actual_latency,
            'success': success,
            'timestamp': time.time()
        })
        
    def get_peer_reliability(self, peer_id: str) -> float:
        """Get reliability score for a peer"""
        if peer_id not in self.peer_reliability:
            return 0.5  # Default neutral reliability
            
        recent_results = self.peer_reliability[peer_id][-10:]  # Last 10 interactions
        if not recent_results:
            return 0.5
            
        return sum(recent_results) / len(recent_results)


class PeerRelaySystem:
    """Main peer relay coordination system"""
    
    def __init__(self, local_peer_id: str):
        self.local_peer_id = local_peer_id
        self.peer_capabilities: Dict[str, PeerCapability] = {}
        self.active_transfers: Dict[str, TransferProgress] = {}
        self.route_cache: Dict[Tuple[str, str], List[RelayRoute]] = {}
        self.metrics = RelayMetrics()
        self.transfer_queue = []  # Priority queue for transfers
        self.bandwidth_allocations: Dict[str, int] = defaultdict(int)
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.max_concurrent_transfers = 10
        self.route_cache_ttl = 300  # 5 minutes
        self.heartbeat_interval = 30  # 30 seconds
        self.max_relay_hops = 5
        
    async def start(self):
        """Start the peer relay system"""
        self.logger.info(f"Starting peer relay system for {self.local_peer_id}")
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._transfer_processor())
        asyncio.create_task(self._metrics_collector())
        
    def register_peer(self, capability: PeerCapability):
        """Register a peer with their capabilities"""
        self.peer_capabilities[capability.peer_id] = capability
        self.logger.info(f"Registered peer {capability.peer_id} with {capability.bandwidth} bps")
        
    def unregister_peer(self, peer_id: str):
        """Remove a peer from the relay system"""
        if peer_id in self.peer_capabilities:
            del self.peer_capabilities[peer_id]
            
        # Cancel any active transfers involving this peer
        for request_id, progress in list(self.active_transfers.items()):
            if any(peer_id in route for route in progress.active_routes):
                progress.status = TransferStatus.FAILED
                progress.error_message = f"Peer {peer_id} disconnected"
                
        self.logger.info(f"Unregistered peer {peer_id}")
        
    async def request_transfer(self, request: TransferRequest) -> str:
        """Request transfer of a data chunk to target peers"""
        request_id = request.request_id
        
        # Initialize transfer progress
        progress = TransferProgress(
            request_id=request_id,
            chunk_id=request.chunk_id,
            status=TransferStatus.PENDING,
            bytes_transferred=0,
            total_bytes=0,  # Will be set when chunk is found
            transfer_rate=0.0
        )
        
        self.active_transfers[request_id] = progress
        
        # Add to transfer queue with priority
        heapq.heappush(self.transfer_queue, (request.priority, time.time(), request))
        
        self.logger.info(f"Queued transfer request {request_id} for chunk {request.chunk_id}")
        return request_id
        
    def get_transfer_status(self, request_id: str) -> Optional[TransferProgress]:
        """Get status of a transfer request"""
        return self.active_transfers.get(request_id)
        
    def cancel_transfer(self, request_id: str) -> bool:
        """Cancel an active transfer"""
        if request_id in self.active_transfers:
            self.active_transfers[request_id].status = TransferStatus.CANCELLED
            return True
        return False
        
    def calculate_relay_routes(self, 
                             source: str, 
                             targets: List[str],
                             strategy: RelayStrategy = RelayStrategy.TRUST_WEIGHTED,
                             constraints: Optional[Dict[str, Any]] = None) -> Dict[str, List[RelayRoute]]:
        """Calculate optimal relay routes to target peers"""
        constraints = constraints or {}
        routes_by_target = {}
        
        for target in targets:
            # Check cache first
            cache_key = (source, target)
            if cache_key in self.route_cache:
                cached_routes = self.route_cache[cache_key]
                if self._routes_still_valid(cached_routes):
                    routes_by_target[target] = cached_routes
                    continue
                    
            # Calculate new routes
            routes = self._calculate_routes_for_target(source, target, strategy, constraints)
            routes_by_target[target] = routes
            
            # Cache the routes
            self.route_cache[cache_key] = routes
            
        return routes_by_target
        
    def _calculate_routes_for_target(self,
                                   source: str,
                                   target: str,
                                   strategy: RelayStrategy,
                                   constraints: Dict[str, Any]) -> List[RelayRoute]:
        """Calculate routes for a specific target"""
        if strategy == RelayStrategy.DIRECT:
            return self._calculate_direct_routes(source, target, constraints)
        elif strategy == RelayStrategy.MULTI_HOP:
            return self._calculate_multihop_routes(source, target, constraints)
        elif strategy == RelayStrategy.TRUST_WEIGHTED:
            return self._calculate_trust_weighted_routes(source, target, constraints)
        elif strategy == RelayStrategy.BANDWIDTH_OPTIMAL:
            return self._calculate_bandwidth_optimal_routes(source, target, constraints)
        else:
            return self._calculate_broadcast_routes(source, target, constraints)
            
    def _calculate_direct_routes(self, source: str, target: str, constraints: Dict[str, Any]) -> List[RelayRoute]:
        """Calculate direct peer-to-peer routes"""
        routes = []
        
        if target in self.peer_capabilities:
            peer_cap = self.peer_capabilities[target]
            
            # Check if peer meets constraints
            if self._meets_constraints(peer_cap, constraints):
                route = RelayRoute(
                    route_id=secrets.token_hex(8),
                    source_peer=source,
                    target_peer=target,
                    hops=[RelayHop(
                        peer_id=target,
                        expected_latency=peer_cap.latency,
                        expected_reliability=peer_cap.reliability,
                        bandwidth_allocation=peer_cap.bandwidth,
                        trust_weight=peer_cap.trust_score
                    )],
                    total_latency=peer_cap.latency,
                    total_reliability=peer_cap.reliability,
                    min_bandwidth=peer_cap.bandwidth,
                    route_cost=peer_cap.latency / max(peer_cap.trust_score, 0.1),
                    privacy_compatible=self._check_privacy_compatibility(peer_cap, constraints)
                )
                routes.append(route)
                
        return routes
        
    def _calculate_multihop_routes(self, source: str, target: str, constraints: Dict[str, Any]) -> List[RelayRoute]:
        """Calculate multi-hop routes using Dijkstra's algorithm"""
        # Build graph of peer connections
        graph = self._build_peer_graph(constraints)
        
        # Find shortest paths
        distances = {peer: float('inf') for peer in graph}
        distances[source] = 0
        previous = {}
        unvisited = set(graph.keys())
        
        while unvisited:
            current = min(unvisited, key=lambda x: distances[x])
            if distances[current] == float('inf'):
                break
                
            unvisited.remove(current)
            
            if current == target:
                break
                
            for neighbor, cost in graph[current].items():
                if neighbor in unvisited:
                    alt_distance = distances[current] + cost
                    if alt_distance < distances[neighbor]:
                        distances[neighbor] = alt_distance
                        previous[neighbor] = current
                        
        # Reconstruct path
        if target not in previous and target != source:
            return []  # No path found
            
        path = []
        current = target
        while current != source:
            path.insert(0, current)
            current = previous[current]
            
        # Convert path to route
        return self._path_to_routes(source, path, constraints)
        
    def _calculate_trust_weighted_routes(self, source: str, target: str, constraints: Dict[str, Any]) -> List[RelayRoute]:
        """Calculate routes weighted by trust scores"""
        # Use trust scores to modify edge weights in graph
        trust_constraints = constraints.copy()
        trust_constraints['weight_by_trust'] = True
        
        return self._calculate_multihop_routes(source, target, trust_constraints)
        
    def _calculate_bandwidth_optimal_routes(self, source: str, target: str, constraints: Dict[str, Any]) -> List[RelayRoute]:
        """Calculate routes optimized for bandwidth"""
        bandwidth_constraints = constraints.copy()
        bandwidth_constraints['optimize_bandwidth'] = True
        
        return self._calculate_multihop_routes(source, target, bandwidth_constraints)
        
    def _calculate_broadcast_routes(self, source: str, target: str, constraints: Dict[str, Any]) -> List[RelayRoute]:
        """Calculate multiple redundant routes for broadcast"""
        # Get multiple independent paths
        routes = []
        
        # Try different strategies and combine results
        direct_routes = self._calculate_direct_routes(source, target, constraints)
        multihop_routes = self._calculate_multihop_routes(source, target, constraints)
        
        routes.extend(direct_routes)
        routes.extend(multihop_routes)
        
        # Remove duplicate routes and sort by quality
        unique_routes = []
        seen_paths = set()
        
        for route in routes:
            path_signature = tuple(hop.peer_id for hop in route.hops)
            if path_signature not in seen_paths:
                unique_routes.append(route)
                seen_paths.add(path_signature)
                
        return sorted(unique_routes, key=lambda r: r.route_cost)[:3]  # Top 3 routes
        
    def _build_peer_graph(self, constraints: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Build graph representation of peer network"""
        graph = defaultdict(dict)
        
        for peer_id, capability in self.peer_capabilities.items():
            if not self._meets_constraints(capability, constraints):
                continue
                
            # Add connections to all other suitable peers
            for other_id, other_capability in self.peer_capabilities.items():
                if other_id != peer_id and self._meets_constraints(other_capability, constraints):
                    
                    # Calculate edge weight based on constraints
                    if constraints.get('weight_by_trust'):
                        weight = capability.latency / max(capability.trust_score, 0.1)
                    elif constraints.get('optimize_bandwidth'):
                        weight = 1.0 / max(capability.bandwidth, 1)
                    else:
                        weight = capability.latency
                        
                    graph[peer_id][other_id] = weight
                    
        return graph
        
    def _meets_constraints(self, capability: PeerCapability, constraints: Dict[str, Any]) -> bool:
        """Check if peer capability meets transfer constraints"""
        min_bandwidth = constraints.get('min_bandwidth', 0)
        max_latency = constraints.get('max_latency', float('inf'))
        min_reliability = constraints.get('min_reliability', 0.0)
        min_trust = constraints.get('min_trust', 0.0)
        required_privacy = constraints.get('privacy_level')
        
        if capability.bandwidth < min_bandwidth:
            return False
        if capability.latency > max_latency:
            return False
        if capability.reliability < min_reliability:
            return False
        if capability.trust_score < min_trust:
            return False
        if required_privacy and required_privacy not in capability.supported_privacy_levels:
            return False
            
        return True
        
    def _check_privacy_compatibility(self, capability: PeerCapability, constraints: Dict[str, Any]) -> bool:
        """Check if peer supports required privacy level"""
        required_privacy = constraints.get('privacy_level')
        if not required_privacy:
            return True
            
        return required_privacy in capability.supported_privacy_levels
        
    def _path_to_routes(self, source: str, path: List[str], constraints: Dict[str, Any]) -> List[RelayRoute]:
        """Convert peer path to route objects"""
        if not path:
            return []
            
        hops = []
        total_latency = 0
        total_reliability = 1.0
        min_bandwidth = float('inf')
        
        for peer_id in path:
            capability = self.peer_capabilities[peer_id]
            
            hop = RelayHop(
                peer_id=peer_id,
                expected_latency=capability.latency,
                expected_reliability=capability.reliability,
                bandwidth_allocation=capability.bandwidth // len(path),  # Distribute bandwidth
                trust_weight=capability.trust_score
            )
            
            hops.append(hop)
            total_latency += capability.latency
            total_reliability *= capability.reliability
            min_bandwidth = min(min_bandwidth, capability.bandwidth)
            
        route_cost = total_latency / max(total_reliability, 0.1)
        
        route = RelayRoute(
            route_id=secrets.token_hex(8),
            source_peer=source,
            target_peer=path[-1],
            hops=hops,
            total_latency=total_latency,
            total_reliability=total_reliability,
            min_bandwidth=min_bandwidth,
            route_cost=route_cost,
            privacy_compatible=all(
                self._check_privacy_compatibility(self.peer_capabilities[hop.peer_id], constraints)
                for hop in hops
            )
        )
        
        return [route]
        
    def _routes_still_valid(self, routes: List[RelayRoute]) -> bool:
        """Check if cached routes are still valid"""
        current_time = time.time()
        
        for route in routes:
            # Check if all peers in route are still available
            for hop in route.hops:
                if hop.peer_id not in self.peer_capabilities:
                    return False
                    
                peer_cap = self.peer_capabilities[hop.peer_id]
                # Check if peer has been offline recently
                if current_time - peer_cap.online_since > self.route_cache_ttl:
                    return False
                    
        return True
        
    async def _transfer_processor(self):
        """Background task to process transfer queue"""
        while True:
            try:
                if self.transfer_queue and len([p for p in self.active_transfers.values() 
                                              if p.status == TransferStatus.IN_PROGRESS]) < self.max_concurrent_transfers:
                    
                    # Get next transfer request
                    priority, queued_at, request = heapq.heappop(self.transfer_queue)
                    
                    # Check if request hasn't timed out
                    if time.time() - queued_at > request.timeout:
                        self.active_transfers[request.request_id].status = TransferStatus.FAILED
                        self.active_transfers[request.request_id].error_message = "Request timeout"
                        continue
                        
                    # Process the transfer
                    await self._execute_transfer(request)
                    
                await asyncio.sleep(0.1)  # Small delay to prevent busy loop
                
            except Exception as e:
                self.logger.error(f"Error in transfer processor: {e}")
                await asyncio.sleep(1)
                
    async def _execute_transfer(self, request: TransferRequest):
        """Execute a data transfer request"""
        progress = self.active_transfers[request.request_id]
        progress.status = TransferStatus.IN_PROGRESS
        progress.started_at = time.time()
        
        try:
            # Calculate routes to all target peers
            routes_by_target = self.calculate_relay_routes(
                request.source_peer,
                request.target_peers,
                RelayStrategy.TRUST_WEIGHTED,
                {
                    'privacy_level': request.privacy_level,
                    'min_reliability': request.min_reliability,
                    'max_hops': request.max_hops
                }
            )
            
            # Execute transfers in parallel
            transfer_tasks = []
            for target, routes in routes_by_target.items():
                if routes:
                    best_route = routes[0]  # Use best route
                    task = asyncio.create_task(self._transfer_via_route(request, best_route))
                    transfer_tasks.append(task)
                    progress.active_routes.append(best_route.route_id)
                    
            if not transfer_tasks:
                progress.status = TransferStatus.FAILED
                progress.error_message = "No valid routes found"
                return
                
            # Wait for all transfers to complete
            results = await asyncio.gather(*transfer_tasks, return_exceptions=True)
            
            # Check results
            successful_transfers = sum(1 for result in results if result is True)
            if successful_transfers > 0:
                progress.status = TransferStatus.COMPLETED
                progress.completed_at = time.time()
                self.metrics.record_transfer(request.request_id, progress)
            else:
                progress.status = TransferStatus.FAILED
                progress.error_message = "All transfer routes failed"
                
        except Exception as e:
            progress.status = TransferStatus.FAILED
            progress.error_message = str(e)
            self.logger.error(f"Transfer execution failed: {e}")
            
    async def _transfer_via_route(self, request: TransferRequest, route: RelayRoute) -> bool:
        """Transfer data via a specific route"""
        try:
            # This would implement the actual data transfer protocol
            # For now, simulate transfer with delay
            transfer_delay = route.total_latency * 2  # Simulate network delay
            await asyncio.sleep(transfer_delay)
            
            # Simulate transfer success/failure based on route reliability
            import random
            success = random.random() < route.total_reliability
            
            if success:
                self.metrics.record_route_performance(route.route_id, route.total_latency, True)
                return True
            else:
                self.metrics.record_route_performance(route.route_id, route.total_latency, False)
                return False
                
        except Exception as e:
            self.logger.error(f"Route transfer failed: {e}")
            self.metrics.record_route_performance(route.route_id, route.total_latency, False)
            return False
            
    async def _heartbeat_loop(self):
        """Background heartbeat to monitor peer health"""
        while True:
            try:
                current_time = time.time()
                
                # Check peer health and update capabilities
                for peer_id, capability in list(self.peer_capabilities.items()):
                    # Simulate peer health check
                    if current_time - capability.online_since > 3600:  # 1 hour offline
                        self.logger.warning(f"Peer {peer_id} appears to be offline")
                        
                # Clear old route cache entries
                for cache_key in list(self.route_cache.keys()):
                    routes = self.route_cache[cache_key]
                    if not self._routes_still_valid(routes):
                        del self.route_cache[cache_key]
                        
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(self.heartbeat_interval)
                
    async def _metrics_collector(self):
        """Background task to collect and analyze metrics"""
        while True:
            try:
                # Collect metrics and update peer reliability scores
                for peer_id in self.peer_capabilities:
                    reliability = self.metrics.get_peer_reliability(peer_id)
                    self.peer_capabilities[peer_id].reliability = reliability
                    
                await asyncio.sleep(60)  # Collect metrics every minute
                
            except Exception as e:
                self.logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(60)
                
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        active_count = len([p for p in self.active_transfers.values() 
                           if p.status == TransferStatus.IN_PROGRESS])
        
        return {
            'peer_count': len(self.peer_capabilities),
            'active_transfers': active_count,
            'queued_transfers': len(self.transfer_queue),
            'total_transfers': len(self.active_transfers),
            'route_cache_size': len(self.route_cache),
            'bandwidth_allocations': dict(self.bandwidth_allocations),
            'avg_peer_trust': sum(p.trust_score for p in self.peer_capabilities.values()) / 
                             max(len(self.peer_capabilities), 1),
            'avg_peer_reliability': sum(p.reliability for p in self.peer_capabilities.values()) / 
                                  max(len(self.peer_capabilities), 1)
        }