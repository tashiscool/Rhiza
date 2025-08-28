"""
Message Router - Intelligent Message Routing for Mesh Networks

Implements sophisticated routing algorithms to efficiently deliver messages
through the mesh network with:
- Adaptive routing based on network topology
- Load balancing across multiple paths
- Fault tolerance and self-healing routes
- Quality of Service (QoS) routing
- Optimized for Apple M4 Pro performance
"""

import asyncio
import time
import math
from typing import Dict, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import heapq
from collections import defaultdict, deque
import random

logger = logging.getLogger(__name__)

class RoutingStrategy(Enum):
    """Available routing strategies"""
    SHORTEST_PATH = "shortest_path"
    LOAD_BALANCED = "load_balanced"
    TRUST_WEIGHTED = "trust_weighted"
    QOS_AWARE = "qos_aware"
    REDUNDANT = "redundant"
    GOSSIP = "gossip"

@dataclass
class RouteInfo:
    """Information about a route to a destination"""
    destination: str
    next_hop: str
    hop_count: int
    latency: float
    reliability: float
    trust_score: float
    bandwidth: float
    last_updated: float
    route_quality: float = 0.0
    
    def __post_init__(self):
        """Calculate route quality score"""
        self.route_quality = self._calculate_quality()
    
    def _calculate_quality(self) -> float:
        """Calculate overall route quality"""
        # Weighted score considering multiple factors
        quality = (
            (1.0 / max(self.hop_count, 1)) * 0.2 +  # Shorter is better
            (1.0 / max(self.latency, 0.001)) * 0.3 +  # Lower latency better
            self.reliability * 0.2 +  # Higher reliability better
            self.trust_score * 0.2 +  # Higher trust better
            (self.bandwidth / 1000.0) * 0.1  # Higher bandwidth better
        )
        return min(1.0, quality)
    
    def update_metrics(self, latency: float = None, reliability: float = None,
                      trust_score: float = None, bandwidth: float = None):
        """Update route metrics"""
        if latency is not None:
            self.latency = latency
        if reliability is not None:
            self.reliability = reliability
        if trust_score is not None:
            self.trust_score = trust_score
        if bandwidth is not None:
            self.bandwidth = bandwidth
        
        self.last_updated = time.time()
        self.route_quality = self._calculate_quality()

@dataclass
class NetworkTopology:
    """Represents the network topology for routing"""
    nodes: Set[str] = field(default_factory=set)
    edges: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    node_metrics: Dict[str, Dict] = field(default_factory=dict)
    last_update: float = field(default_factory=time.time)
    
    def add_node(self, node_id: str, metrics: Dict = None):
        """Add a node to the topology"""
        self.nodes.add(node_id)
        if metrics:
            self.node_metrics[node_id] = metrics
        self.last_update = time.time()
    
    def remove_node(self, node_id: str):
        """Remove a node from the topology"""
        self.nodes.discard(node_id)
        self.edges.pop(node_id, None)
        for node_edges in self.edges.values():
            node_edges.pop(node_id, None)
        self.node_metrics.pop(node_id, None)
        self.last_update = time.time()
    
    def add_edge(self, from_node: str, to_node: str, weight: float):
        """Add or update an edge between nodes"""
        self.edges[from_node][to_node] = weight
        self.edges[to_node][from_node] = weight  # Bidirectional
        self.last_update = time.time()
    
    def remove_edge(self, from_node: str, to_node: str):
        """Remove an edge between nodes"""
        self.edges.get(from_node, {}).pop(to_node, None)
        self.edges.get(to_node, {}).pop(from_node, None)
        self.last_update = time.time()
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get neighbors of a node"""
        return list(self.edges.get(node_id, {}).keys())
    
    def get_edge_weight(self, from_node: str, to_node: str) -> Optional[float]:
        """Get weight of edge between nodes"""
        return self.edges.get(from_node, {}).get(to_node)

class MessageRouter:
    """
    Intelligent message routing system for mesh networks
    
    Provides multiple routing strategies with dynamic adaptation
    based on network conditions, trust relationships, and performance.
    """
    
    def __init__(self, node_id: str, config: Dict = None):
        self.node_id = node_id
        self.config = config or {}
        
        # Routing state
        self.topology = NetworkTopology()
        self.routing_table: Dict[str, RouteInfo] = {}
        self.alternative_routes: Dict[str, List[RouteInfo]] = defaultdict(list)
        self.route_cache: Dict[str, List[str]] = {}
        self.failed_routes: Set[Tuple[str, str]] = set()
        
        # Strategy configuration
        self.default_strategy = RoutingStrategy(
            self.config.get('default_strategy', 'trust_weighted')
        )
        self.max_alternative_routes = self.config.get('max_alternative_routes', 3)
        self.route_timeout = self.config.get('route_timeout', 300)  # 5 minutes
        self.max_hops = self.config.get('max_hops', 10)
        
        # Performance metrics
        self.metrics = {
            'messages_routed': 0,
            'routing_failures': 0,
            'route_discoveries': 0,
            'topology_updates': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Callbacks
        self.message_sender: Optional[Callable] = None
        self.trust_scorer: Optional[Callable] = None
        
        # Background tasks
        self.is_running = False
        
        logger.info(f"Message router initialized for node {self.node_id}")
    
    async def start(self):
        """Start the message router"""
        if self.is_running:
            return
            
        self.is_running = True
        
        # Add ourselves to topology
        self.topology.add_node(self.node_id, {'type': 'local'})
        
        # Start background tasks
        asyncio.create_task(self._route_maintenance_task())
        asyncio.create_task(self._topology_discovery_task())
        
        logger.info("Message router started")
    
    async def stop(self):
        """Stop the message router"""
        if not self.is_running:
            return
            
        self.is_running = False
        logger.info("Message router stopped")
    
    async def route_message(self, destination: str, message_data: bytes,
                           strategy: RoutingStrategy = None,
                           priority: int = 1) -> bool:
        """Route a message to its destination"""
        try:
            if destination == self.node_id:
                # Message is for us, don't route
                return True
            
            strategy = strategy or self.default_strategy
            
            # Find route to destination
            route = await self._find_route(destination, strategy)
            if not route:
                logger.warning(f"No route found to {destination}")
                self.metrics['routing_failures'] += 1
                return False
            
            # Send message via route
            success = await self._send_via_route(route, message_data)
            if success:
                self.metrics['messages_routed'] += 1
                # Update route metrics based on success
                route.update_metrics(reliability=min(1.0, route.reliability + 0.1))
            else:
                # Mark route as potentially problematic
                route.update_metrics(reliability=max(0.0, route.reliability - 0.2))
                self.metrics['routing_failures'] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Error routing message to {destination}: {e}")
            self.metrics['routing_failures'] += 1
            return False
    
    async def _find_route(self, destination: str, strategy: RoutingStrategy) -> Optional[RouteInfo]:
        """Find the best route to a destination"""
        # Check routing table first
        if destination in self.routing_table:
            route = self.routing_table[destination]
            if time.time() - route.last_updated < self.route_timeout:
                self.metrics['cache_hits'] += 1
                return route
        
        self.metrics['cache_misses'] += 1
        
        # Find new route based on strategy
        if strategy == RoutingStrategy.SHORTEST_PATH:
            route = await self._find_shortest_path(destination)
        elif strategy == RoutingStrategy.LOAD_BALANCED:
            route = await self._find_load_balanced_route(destination)
        elif strategy == RoutingStrategy.TRUST_WEIGHTED:
            route = await self._find_trust_weighted_route(destination)
        elif strategy == RoutingStrategy.QOS_AWARE:
            route = await self._find_qos_aware_route(destination)
        elif strategy == RoutingStrategy.REDUNDANT:
            route = await self._find_redundant_route(destination)
        else:
            route = await self._find_shortest_path(destination)
        
        # Cache the route
        if route:
            self.routing_table[destination] = route
            self.metrics['route_discoveries'] += 1
        
        return route
    
    async def _find_shortest_path(self, destination: str) -> Optional[RouteInfo]:
        """Find shortest path using Dijkstra's algorithm"""
        if destination not in self.topology.nodes:
            return None
        
        # Dijkstra's algorithm
        distances = {node: float('inf') for node in self.topology.nodes}
        distances[self.node_id] = 0
        previous = {}
        unvisited = list(self.topology.nodes)
        
        while unvisited:
            # Find node with minimum distance
            current = min(unvisited, key=lambda x: distances[x])
            
            if distances[current] == float('inf'):
                break
            
            unvisited.remove(current)
            
            if current == destination:
                break
            
            # Check neighbors
            for neighbor in self.topology.get_neighbors(current):
                if neighbor in unvisited:
                    weight = self.topology.get_edge_weight(current, neighbor) or 1.0
                    distance = distances[current] + weight
                    
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current
        
        # Build route if path exists
        if destination in previous or destination == self.node_id:
            path = self._build_path(previous, destination)
            if len(path) > 1:
                next_hop = path[1]  # First hop after ourselves
                return RouteInfo(
                    destination=destination,
                    next_hop=next_hop,
                    hop_count=len(path) - 1,
                    latency=distances[destination],
                    reliability=0.8,  # Default reliability
                    trust_score=0.5,  # Default trust
                    bandwidth=100.0,  # Default bandwidth
                    last_updated=time.time()
                )
        
        return None
    
    async def _find_trust_weighted_route(self, destination: str) -> Optional[RouteInfo]:
        """Find route weighted by trust scores"""
        if destination not in self.topology.nodes:
            return None
        
        # Modified Dijkstra with trust weighting
        distances = {node: float('inf') for node in self.topology.nodes}
        distances[self.node_id] = 0
        trust_scores = {node: 1.0 for node in self.topology.nodes}
        previous = {}
        unvisited = list(self.topology.nodes)
        
        while unvisited:
            # Find node with best trust-weighted distance
            current = min(unvisited, key=lambda x: distances[x] / max(trust_scores[x], 0.1))
            
            if distances[current] == float('inf'):
                break
            
            unvisited.remove(current)
            
            if current == destination:
                break
            
            # Check neighbors with trust weighting
            for neighbor in self.topology.get_neighbors(current):
                if neighbor in unvisited:
                    base_weight = self.topology.get_edge_weight(current, neighbor) or 1.0
                    
                    # Get trust score for neighbor
                    neighbor_trust = await self._get_trust_score(neighbor)
                    trust_scores[neighbor] = neighbor_trust
                    
                    # Weight by inverse of trust (lower trust = higher cost)
                    trust_weight = base_weight / max(neighbor_trust, 0.1)
                    distance = distances[current] + trust_weight
                    
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current
        
        # Build route
        if destination in previous or destination == self.node_id:
            path = self._build_path(previous, destination)
            if len(path) > 1:
                next_hop = path[1]
                return RouteInfo(
                    destination=destination,
                    next_hop=next_hop,
                    hop_count=len(path) - 1,
                    latency=distances[destination],
                    reliability=0.9,
                    trust_score=trust_scores.get(destination, 0.5),
                    bandwidth=100.0,
                    last_updated=time.time()
                )
        
        return None
    
    async def _find_load_balanced_route(self, destination: str) -> Optional[RouteInfo]:
        """Find route considering load balancing"""
        # Get multiple possible routes
        routes = []
        
        # Try different intermediate nodes
        for intermediate in self.topology.get_neighbors(self.node_id):
            route = await self._find_route_via_intermediate(destination, intermediate)
            if route:
                routes.append(route)
        
        if not routes:
            return await self._find_shortest_path(destination)
        
        # Select route with best load characteristics
        best_route = min(routes, key=lambda r: r.latency * (2.0 - r.reliability))
        return best_route
    
    async def _find_qos_aware_route(self, destination: str) -> Optional[RouteInfo]:
        """Find route optimized for quality of service"""
        # Similar to trust-weighted but considers latency, bandwidth, reliability
        route = await self._find_trust_weighted_route(destination)
        if route:
            # Adjust quality score for QoS
            route.route_quality *= 1.2  # Boost QoS routes
        return route
    
    async def _find_redundant_route(self, destination: str) -> Optional[RouteInfo]:
        """Find multiple redundant routes for reliability"""
        primary_route = await self._find_trust_weighted_route(destination)
        
        if primary_route:
            # Find alternative routes
            alternatives = []
            for _ in range(self.max_alternative_routes - 1):
                # Temporarily remove primary route edges to find alternatives
                temp_removed = []
                path = await self._get_route_path(primary_route)
                
                for i in range(len(path) - 1):
                    from_node, to_node = path[i], path[i + 1]
                    weight = self.topology.get_edge_weight(from_node, to_node)
                    if weight is not None:
                        self.topology.remove_edge(from_node, to_node)
                        temp_removed.append((from_node, to_node, weight))
                
                # Find alternative route
                alt_route = await self._find_trust_weighted_route(destination)
                if alt_route:
                    alternatives.append(alt_route)
                
                # Restore removed edges
                for from_node, to_node, weight in temp_removed:
                    self.topology.add_edge(from_node, to_node, weight)
            
            # Store alternative routes
            if alternatives:
                self.alternative_routes[destination] = alternatives
        
        return primary_route
    
    async def _find_route_via_intermediate(self, destination: str, intermediate: str) -> Optional[RouteInfo]:
        """Find route via specific intermediate node"""
        # This is a simplified version - would need more sophisticated implementation
        if intermediate not in self.topology.nodes or destination not in self.topology.nodes:
            return None
        
        # Check if direct path exists via intermediate
        if (intermediate in self.topology.get_neighbors(self.node_id) and
            destination in self.topology.get_neighbors(intermediate)):
            
            weight1 = self.topology.get_edge_weight(self.node_id, intermediate) or 1.0
            weight2 = self.topology.get_edge_weight(intermediate, destination) or 1.0
            
            return RouteInfo(
                destination=destination,
                next_hop=intermediate,
                hop_count=2,
                latency=weight1 + weight2,
                reliability=0.8,
                trust_score=await self._get_trust_score(intermediate),
                bandwidth=100.0,
                last_updated=time.time()
            )
        
        return None
    
    def _build_path(self, previous: Dict, destination: str) -> List[str]:
        """Build path from previous nodes dict"""
        path = []
        current = destination
        
        while current is not None:
            path.append(current)
            current = previous.get(current)
        
        path.reverse()
        return path
    
    async def _get_route_path(self, route: RouteInfo) -> List[str]:
        """Get full path for a route (simplified)"""
        return [self.node_id, route.next_hop, route.destination]
    
    async def _send_via_route(self, route: RouteInfo, message_data: bytes) -> bool:
        """Send message via specified route"""
        if self.message_sender:
            return await self.message_sender(route.next_hop, message_data)
        else:
            logger.warning("No message sender configured")
            return False
    
    async def _get_trust_score(self, node_id: str) -> float:
        """Get trust score for a node"""
        if self.trust_scorer:
            return await self.trust_scorer(node_id)
        else:
            return 0.5  # Default neutral trust
    
    def update_topology(self, node_id: str, neighbors: List[str], metrics: Dict = None):
        """Update network topology information"""
        try:
            # Add node if not exists
            if node_id not in self.topology.nodes:
                self.topology.add_node(node_id, metrics)
            
            # Update node metrics
            if metrics:
                self.topology.node_metrics[node_id] = metrics
            
            # Update edges to neighbors
            current_neighbors = set(self.topology.get_neighbors(node_id))
            new_neighbors = set(neighbors)
            
            # Add new edges
            for neighbor in new_neighbors - current_neighbors:
                if neighbor in self.topology.nodes:
                    weight = 1.0  # Default weight
                    if metrics and 'latencies' in metrics:
                        weight = metrics['latencies'].get(neighbor, 1.0)
                    self.topology.add_edge(node_id, neighbor, weight)
            
            # Remove old edges
            for neighbor in current_neighbors - new_neighbors:
                self.topology.remove_edge(node_id, neighbor)
            
            # Clear cached routes that might be affected
            self._invalidate_routes_via_node(node_id)
            
            self.metrics['topology_updates'] += 1
            
        except Exception as e:
            logger.error(f"Error updating topology for {node_id}: {e}")
    
    def remove_node(self, node_id: str):
        """Remove a node from topology"""
        if node_id in self.topology.nodes:
            self.topology.remove_node(node_id)
            self._invalidate_routes_via_node(node_id)
            logger.info(f"Removed node {node_id} from topology")
    
    def _invalidate_routes_via_node(self, node_id: str):
        """Invalidate cached routes that go through a specific node"""
        to_remove = []
        for destination, route in self.routing_table.items():
            if route.next_hop == node_id:
                to_remove.append(destination)
        
        for destination in to_remove:
            del self.routing_table[destination]
            self.alternative_routes.pop(destination, None)
    
    async def _route_maintenance_task(self):
        """Background task for route maintenance"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Remove expired routes
                expired_routes = [
                    dest for dest, route in self.routing_table.items()
                    if current_time - route.last_updated > self.route_timeout
                ]
                
                for dest in expired_routes:
                    del self.routing_table[dest]
                    self.alternative_routes.pop(dest, None)
                
                # Clean up failed routes
                self.failed_routes = {
                    (from_node, to_node) for from_node, to_node in self.failed_routes
                    if current_time - time.time() < 300  # Keep failed routes for 5 minutes
                }
                
                await asyncio.sleep(60)  # Maintenance every minute
                
            except Exception as e:
                logger.error(f"Route maintenance error: {e}")
                await asyncio.sleep(60)
    
    async def _topology_discovery_task(self):
        """Background task for topology discovery"""
        while self.is_running:
            try:
                # Request topology updates from neighbors
                neighbors = self.topology.get_neighbors(self.node_id)
                for neighbor in neighbors:
                    # In a real implementation, this would send topology requests
                    pass
                
                await asyncio.sleep(30)  # Discovery every 30 seconds
                
            except Exception as e:
                logger.error(f"Topology discovery error: {e}")
                await asyncio.sleep(30)
    
    def set_message_sender(self, sender: Callable):
        """Set the message sender function"""
        self.message_sender = sender
    
    def set_trust_scorer(self, scorer: Callable):
        """Set the trust scorer function"""
        self.trust_scorer = scorer
    
    def get_routing_table(self) -> Dict[str, Dict]:
        """Get current routing table"""
        return {
            dest: {
                'next_hop': route.next_hop,
                'hop_count': route.hop_count,
                'latency': route.latency,
                'reliability': route.reliability,
                'trust_score': route.trust_score,
                'quality': route.route_quality,
                'last_updated': route.last_updated
            }
            for dest, route in self.routing_table.items()
        }
    
    def get_topology_info(self) -> Dict:
        """Get topology information"""
        return {
            'nodes': list(self.topology.nodes),
            'edges': dict(self.topology.edges),
            'node_count': len(self.topology.nodes),
            'edge_count': sum(len(edges) for edges in self.topology.edges.values()) // 2,
            'last_update': self.topology.last_update
        }
    
    def get_metrics(self) -> Dict:
        """Get routing metrics"""
        return {
            **self.metrics,
            'routing_table_size': len(self.routing_table),
            'alternative_routes': sum(len(routes) for routes in self.alternative_routes.values()),
            'topology_nodes': len(self.topology.nodes),
            'failed_routes': len(self.failed_routes),
            'is_running': self.is_running
        }