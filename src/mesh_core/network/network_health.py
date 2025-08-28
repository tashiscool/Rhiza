"""
Network Health - Monitor Network Topology and Performance

Comprehensive network health monitoring system that tracks:
- Network topology changes and stability
- Performance metrics and trends
- Connection quality and reliability
- Network partitions and healing
- Predictive failure detection
"""

import asyncio
import time
import statistics
from typing import Dict, List, Optional, Set, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Network health status levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class NetworkMetrics:
    """Comprehensive network metrics"""
    # Connectivity metrics
    total_nodes: int = 0
    connected_nodes: int = 0
    edge_count: int = 0
    connectivity_ratio: float = 0.0
    
    # Performance metrics
    average_latency: float = 0.0
    median_latency: float = 0.0
    latency_variance: float = 0.0
    packet_loss_rate: float = 0.0
    throughput: float = 0.0
    
    # Reliability metrics
    uptime_percentage: float = 100.0
    connection_stability: float = 1.0
    message_delivery_rate: float = 1.0
    
    # Network topology metrics
    clustering_coefficient: float = 0.0
    average_path_length: float = 0.0
    network_diameter: int = 0
    
    # Health indicators
    overall_health: HealthStatus = HealthStatus.UNKNOWN
    health_score: float = 0.0
    
    # Temporal data
    timestamp: float = field(default_factory=time.time)
    
    def calculate_health_score(self) -> float:
        """Calculate overall network health score (0-1)"""
        scores = []
        
        # Connectivity score
        if self.total_nodes > 0:
            connectivity_score = self.connected_nodes / self.total_nodes
            scores.append(connectivity_score * 0.25)
        
        # Performance score (inverse of latency, normalized)
        if self.average_latency > 0:
            # Lower latency is better, normalize to 0-1 range
            latency_score = max(0, 1 - (self.average_latency / 1000))  # Assume 1s is very poor
            scores.append(latency_score * 0.20)
        
        # Reliability score
        reliability_score = (
            self.uptime_percentage / 100 * 0.4 +
            self.connection_stability * 0.3 +
            self.message_delivery_rate * 0.3
        )
        scores.append(reliability_score * 0.25)
        
        # Packet loss score (lower is better)
        packet_loss_score = max(0, 1 - self.packet_loss_rate)
        scores.append(packet_loss_score * 0.15)
        
        # Topology score (clustering and path length)
        if self.average_path_length > 0:
            path_score = max(0, 1 - (self.average_path_length / 10))  # Assume 10 hops is very poor
            topology_score = (self.clustering_coefficient * 0.5 + path_score * 0.5)
            scores.append(topology_score * 0.15)
        
        self.health_score = sum(scores) if scores else 0.0
        
        # Determine health status
        if self.health_score >= 0.9:
            self.overall_health = HealthStatus.EXCELLENT
        elif self.health_score >= 0.75:
            self.overall_health = HealthStatus.GOOD
        elif self.health_score >= 0.6:
            self.overall_health = HealthStatus.FAIR
        elif self.health_score >= 0.4:
            self.overall_health = HealthStatus.POOR
        else:
            self.overall_health = HealthStatus.CRITICAL
        
        return self.health_score

@dataclass
class HealthAlert:
    """Network health alert"""
    alert_id: str
    level: AlertLevel
    message: str
    component: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolution_timestamp: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class NodeHealth:
    """Health information for a single node"""
    node_id: str
    is_online: bool = True
    last_seen: float = field(default_factory=time.time)
    connection_count: int = 0
    latency: float = 0.0
    packet_loss: float = 0.0
    throughput: float = 0.0
    uptime: float = 100.0
    error_rate: float = 0.0
    trust_score: float = 0.5
    
    @property
    def health_score(self) -> float:
        """Calculate node health score"""
        if not self.is_online:
            return 0.0
        
        return (
            (1 - self.packet_loss) * 0.3 +
            (1 - min(self.latency / 1000, 1)) * 0.2 +
            min(self.throughput / 1000, 1) * 0.2 +
            (self.uptime / 100) * 0.15 +
            (1 - self.error_rate) * 0.1 +
            self.trust_score * 0.05
        )

class NetworkHealth:
    """
    Comprehensive network health monitoring system
    
    Tracks network topology, performance, and reliability metrics
    to provide insights into mesh network health and predict issues.
    """
    
    def __init__(self, node_id: str, config: Dict = None):
        self.node_id = node_id
        self.config = config or {}
        
        # Health monitoring state
        self.is_monitoring = False
        self.node_health: Dict[str, NodeHealth] = {}
        self.current_metrics = NetworkMetrics()
        self.metrics_history: deque = deque(maxlen=1000)  # Keep last 1000 measurements
        
        # Alert system
        self.active_alerts: Dict[str, HealthAlert] = {}
        self.alert_history: deque = deque(maxlen=500)  # Keep last 500 alerts
        self.alert_callbacks: List[Callable] = []
        
        # Configuration
        self.monitoring_interval = self.config.get('monitoring_interval', 30)
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'latency_warning': 200,  # ms
            'latency_critical': 1000,  # ms
            'packet_loss_warning': 0.05,  # 5%
            'packet_loss_critical': 0.15,  # 15%
            'connectivity_warning': 0.8,  # 80%
            'connectivity_critical': 0.6,  # 60%
        })
        
        # Network topology tracking
        self.topology_graph: Dict[str, Set[str]] = defaultdict(set)
        self.topology_history: deque = deque(maxlen=100)
        
        # Performance tracking
        self.latency_measurements: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.throughput_measurements: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # Predictive analysis
        self.failure_predictions: Dict[str, float] = {}
        self.trend_analysis_enabled = self.config.get('trend_analysis', True)
        
        logger.info(f"Network health monitor initialized for node {self.node_id}")
    
    async def start_monitoring(self):
        """Start network health monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        
        # Start monitoring tasks
        asyncio.create_task(self._metrics_collection_task())
        asyncio.create_task(self._health_analysis_task())
        asyncio.create_task(self._alert_management_task())
        
        if self.trend_analysis_enabled:
            asyncio.create_task(self._trend_analysis_task())
        
        logger.info("Network health monitoring started")
    
    async def stop_monitoring(self):
        """Stop network health monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        logger.info("Network health monitoring stopped")
    
    def update_node_status(self, node_id: str, is_online: bool, 
                          connection_count: int = 0, latency: float = 0.0,
                          throughput: float = 0.0, error_count: int = 0):
        """Update status for a network node"""
        if node_id not in self.node_health:
            self.node_health[node_id] = NodeHealth(node_id=node_id)
        
        node = self.node_health[node_id]
        node.is_online = is_online
        node.connection_count = connection_count
        node.latency = latency
        node.throughput = throughput
        node.last_seen = time.time()
        
        # Update error rate
        if error_count > 0:
            self.error_counts[node_id] += error_count
            total_interactions = len(self.latency_measurements[node_id]) + 1
            node.error_rate = min(1.0, self.error_counts[node_id] / total_interactions)
        
        # Store measurements
        if latency > 0:
            self.latency_measurements[node_id].append(latency)
        if throughput > 0:
            self.throughput_measurements[node_id].append(throughput)
        
        logger.debug(f"Updated status for node {node_id}: online={is_online}, "
                    f"latency={latency}ms, throughput={throughput}")
    
    def update_topology(self, node_id: str, neighbors: List[str]):
        """Update network topology information"""
        # Clear old connections for this node
        old_neighbors = self.topology_graph[node_id].copy()
        self.topology_graph[node_id].clear()
        
        # Add new connections
        for neighbor in neighbors:
            self.topology_graph[node_id].add(neighbor)
            self.topology_graph[neighbor].add(node_id)  # Bidirectional
        
        # Remove old connections
        for old_neighbor in old_neighbors:
            if old_neighbor not in neighbors:
                self.topology_graph[old_neighbor].discard(node_id)
        
        # Store topology snapshot
        topology_snapshot = {
            'timestamp': time.time(),
            'nodes': len(self.topology_graph),
            'edges': sum(len(neighbors) for neighbors in self.topology_graph.values()) // 2,
            'node_id': node_id,
            'change': 'topology_update'
        }
        self.topology_history.append(topology_snapshot)
        
        logger.debug(f"Updated topology for node {node_id}: {len(neighbors)} neighbors")
    
    def record_message_metrics(self, from_node: str, to_node: str,
                             latency: float, success: bool, size: int):
        """Record message delivery metrics"""
        # Update latency
        if success and latency > 0:
            self.latency_measurements[to_node].append(latency)
        
        # Update delivery success/failure
        if not success:
            self.error_counts[to_node] += 1
        
        # Update throughput (simplified)
        if success and latency > 0:
            throughput = size / (latency / 1000)  # bytes per second
            self.throughput_measurements[to_node].append(throughput)
    
    def create_alert(self, level: AlertLevel, message: str, 
                    component: str, metadata: Dict = None) -> str:
        """Create a new health alert"""
        alert_id = f"{component}_{level.value}_{int(time.time())}"
        
        alert = HealthAlert(
            alert_id=alert_id,
            level=level,
            message=message,
            component=component,
            metadata=metadata or {}
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        logger.warning(f"Health alert [{level.value.upper()}]: {message}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                asyncio.create_task(callback(alert))
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        return alert_id
    
    def resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts.pop(alert_id)
            alert.resolved = True
            alert.resolution_timestamp = time.time()
            logger.info(f"Resolved alert: {alert.message}")
    
    async def _metrics_collection_task(self):
        """Background task to collect network metrics"""
        while self.is_monitoring:
            try:
                # Collect current metrics
                await self._collect_network_metrics()
                
                # Store in history
                self.metrics_history.append(self.current_metrics)
                
                # Check for alerts
                await self._check_thresholds()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_network_metrics(self):
        """Collect comprehensive network metrics"""
        current_time = time.time()
        
        # Basic connectivity metrics
        online_nodes = [node for node in self.node_health.values() if node.is_online]
        self.current_metrics.total_nodes = len(self.node_health)
        self.current_metrics.connected_nodes = len(online_nodes)
        
        if self.current_metrics.total_nodes > 0:
            self.current_metrics.connectivity_ratio = (
                self.current_metrics.connected_nodes / self.current_metrics.total_nodes
            )
        
        # Edge count from topology
        self.current_metrics.edge_count = sum(
            len(neighbors) for neighbors in self.topology_graph.values()
        ) // 2
        
        # Performance metrics
        all_latencies = []
        all_throughputs = []
        
        for node in online_nodes:
            if node.node_id in self.latency_measurements:
                recent_latencies = list(self.latency_measurements[node.node_id])
                all_latencies.extend(recent_latencies)
            
            if node.node_id in self.throughput_measurements:
                recent_throughputs = list(self.throughput_measurements[node.node_id])
                all_throughputs.extend(recent_throughputs)
        
        if all_latencies:
            self.current_metrics.average_latency = statistics.mean(all_latencies)
            self.current_metrics.median_latency = statistics.median(all_latencies)
            if len(all_latencies) > 1:
                self.current_metrics.latency_variance = statistics.variance(all_latencies)
        
        if all_throughputs:
            self.current_metrics.throughput = statistics.mean(all_throughputs)
        
        # Calculate packet loss rate
        total_errors = sum(self.error_counts.values())
        total_messages = sum(len(measurements) for measurements in self.latency_measurements.values())
        if total_messages > 0:
            self.current_metrics.packet_loss_rate = total_errors / total_messages
        
        # Topology metrics
        self._calculate_topology_metrics()
        
        # Calculate overall health
        self.current_metrics.calculate_health_score()
        
        # Update timestamp
        self.current_metrics.timestamp = current_time
    
    def _calculate_topology_metrics(self):
        """Calculate network topology metrics"""
        if not self.topology_graph:
            return
        
        nodes = list(self.topology_graph.keys())
        n = len(nodes)
        
        if n < 2:
            return
        
        # Calculate clustering coefficient
        total_clustering = 0
        for node in nodes:
            neighbors = self.topology_graph[node]
            if len(neighbors) < 2:
                continue
            
            # Count triangles
            triangles = 0
            possible_triangles = len(neighbors) * (len(neighbors) - 1) / 2
            
            for neighbor1 in neighbors:
                for neighbor2 in neighbors:
                    if neighbor1 != neighbor2 and neighbor2 in self.topology_graph[neighbor1]:
                        triangles += 1
            
            triangles = triangles // 2  # Each triangle counted twice
            
            if possible_triangles > 0:
                clustering = triangles / possible_triangles
                total_clustering += clustering
        
        self.current_metrics.clustering_coefficient = total_clustering / n
        
        # Calculate average path length using BFS
        total_path_length = 0
        total_paths = 0
        max_distance = 0
        
        for start_node in nodes:
            distances = self._bfs_distances(start_node)
            for end_node, distance in distances.items():
                if distance > 0 and distance != float('inf'):
                    total_path_length += distance
                    total_paths += 1
                    max_distance = max(max_distance, distance)
        
        if total_paths > 0:
            self.current_metrics.average_path_length = total_path_length / total_paths
        
        self.current_metrics.network_diameter = max_distance
    
    def _bfs_distances(self, start_node: str) -> Dict[str, int]:
        """Calculate shortest path distances from start node using BFS"""
        distances = {node: float('inf') for node in self.topology_graph}
        distances[start_node] = 0
        
        queue = deque([start_node])
        
        while queue:
            current = queue.popleft()
            current_distance = distances[current]
            
            for neighbor in self.topology_graph[current]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = current_distance + 1
                    queue.append(neighbor)
        
        return distances
    
    async def _check_thresholds(self):
        """Check metrics against alert thresholds"""
        metrics = self.current_metrics
        
        # Latency alerts
        if metrics.average_latency > self.alert_thresholds['latency_critical']:
            self.create_alert(
                AlertLevel.CRITICAL,
                f"Critical network latency: {metrics.average_latency:.1f}ms",
                "latency",
                {'value': metrics.average_latency, 'threshold': self.alert_thresholds['latency_critical']}
            )
        elif metrics.average_latency > self.alert_thresholds['latency_warning']:
            self.create_alert(
                AlertLevel.WARNING,
                f"High network latency: {metrics.average_latency:.1f}ms",
                "latency",
                {'value': metrics.average_latency, 'threshold': self.alert_thresholds['latency_warning']}
            )
        
        # Packet loss alerts
        if metrics.packet_loss_rate > self.alert_thresholds['packet_loss_critical']:
            self.create_alert(
                AlertLevel.CRITICAL,
                f"Critical packet loss: {metrics.packet_loss_rate:.1%}",
                "packet_loss",
                {'value': metrics.packet_loss_rate, 'threshold': self.alert_thresholds['packet_loss_critical']}
            )
        elif metrics.packet_loss_rate > self.alert_thresholds['packet_loss_warning']:
            self.create_alert(
                AlertLevel.WARNING,
                f"High packet loss: {metrics.packet_loss_rate:.1%}",
                "packet_loss",
                {'value': metrics.packet_loss_rate, 'threshold': self.alert_thresholds['packet_loss_warning']}
            )
        
        # Connectivity alerts
        if metrics.connectivity_ratio < self.alert_thresholds['connectivity_critical']:
            self.create_alert(
                AlertLevel.CRITICAL,
                f"Critical connectivity loss: {metrics.connectivity_ratio:.1%}",
                "connectivity",
                {'value': metrics.connectivity_ratio, 'threshold': self.alert_thresholds['connectivity_critical']}
            )
        elif metrics.connectivity_ratio < self.alert_thresholds['connectivity_warning']:
            self.create_alert(
                AlertLevel.WARNING,
                f"Low network connectivity: {metrics.connectivity_ratio:.1%}",
                "connectivity",
                {'value': metrics.connectivity_ratio, 'threshold': self.alert_thresholds['connectivity_warning']}
            )
    
    async def _health_analysis_task(self):
        """Background task for health analysis"""
        while self.is_monitoring:
            try:
                # Analyze individual node health
                for node_id, node in self.node_health.items():
                    health_score = node.health_score
                    
                    if health_score < 0.3:
                        self.create_alert(
                            AlertLevel.WARNING,
                            f"Poor node health: {node_id} (score: {health_score:.2f})",
                            "node_health",
                            {'node_id': node_id, 'health_score': health_score}
                        )
                
                await asyncio.sleep(60)  # Analyze every minute
                
            except Exception as e:
                logger.error(f"Health analysis error: {e}")
                await asyncio.sleep(60)
    
    async def _alert_management_task(self):
        """Background task for alert management"""
        while self.is_monitoring:
            try:
                current_time = time.time()
                
                # Auto-resolve alerts that are no longer relevant
                to_resolve = []
                for alert_id, alert in self.active_alerts.items():
                    # Auto-resolve old alerts (24 hours)
                    if current_time - alert.timestamp > 86400:
                        to_resolve.append(alert_id)
                    
                    # Check if conditions have improved
                    elif alert.component == "latency" and self.current_metrics.average_latency < self.alert_thresholds['latency_warning']:
                        to_resolve.append(alert_id)
                    elif alert.component == "packet_loss" and self.current_metrics.packet_loss_rate < self.alert_thresholds['packet_loss_warning']:
                        to_resolve.append(alert_id)
                    elif alert.component == "connectivity" and self.current_metrics.connectivity_ratio > self.alert_thresholds['connectivity_warning']:
                        to_resolve.append(alert_id)
                
                for alert_id in to_resolve:
                    self.resolve_alert(alert_id)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Alert management error: {e}")
                await asyncio.sleep(300)
    
    async def _trend_analysis_task(self):
        """Background task for trend analysis and predictions"""
        while self.is_monitoring:
            try:
                # Perform trend analysis if we have enough historical data
                if len(self.metrics_history) > 10:
                    await self._analyze_trends()
                
                await asyncio.sleep(300)  # Analyze trends every 5 minutes
                
            except Exception as e:
                logger.error(f"Trend analysis error: {e}")
                await asyncio.sleep(300)
    
    async def _analyze_trends(self):
        """Analyze trends in network metrics"""
        try:
            # Get recent metrics
            recent_metrics = list(self.metrics_history)[-20:]  # Last 20 measurements
            
            # Analyze latency trend
            latencies = [m.average_latency for m in recent_metrics if m.average_latency > 0]
            if len(latencies) > 5:
                trend = self._calculate_trend(latencies)
                if trend > 0.1:  # Increasing latency
                    self.create_alert(
                        AlertLevel.INFO,
                        f"Latency trend increasing: {trend:.2f}ms per measurement",
                        "trend_analysis",
                        {'trend': 'latency_increasing', 'value': trend}
                    )
            
            # Analyze connectivity trend
            connectivity_ratios = [m.connectivity_ratio for m in recent_metrics]
            if len(connectivity_ratios) > 5:
                trend = self._calculate_trend(connectivity_ratios)
                if trend < -0.01:  # Decreasing connectivity
                    self.create_alert(
                        AlertLevel.WARNING,
                        f"Connectivity trend decreasing: {trend:.2%} per measurement",
                        "trend_analysis",
                        {'trend': 'connectivity_decreasing', 'value': trend}
                    )
            
        except Exception as e:
            logger.error(f"Trend analysis calculation error: {e}")
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend in values"""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x_squared_sum = sum(i * i for i in range(n))
        
        # Linear regression slope
        slope = (n * xy_sum - x_sum * y_sum) / (n * x_squared_sum - x_sum * x_sum)
        return slope
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> NetworkMetrics:
        """Get current network metrics"""
        return self.current_metrics
    
    def get_metrics_history(self, count: int = 100) -> List[NetworkMetrics]:
        """Get historical metrics"""
        return list(self.metrics_history)[-count:]
    
    def get_active_alerts(self) -> List[HealthAlert]:
        """Get active alerts"""
        return list(self.active_alerts.values())
    
    def get_node_health(self, node_id: str) -> Optional[NodeHealth]:
        """Get health information for a specific node"""
        return self.node_health.get(node_id)
    
    def get_all_node_health(self) -> Dict[str, NodeHealth]:
        """Get health information for all nodes"""
        return self.node_health.copy()
    
    def get_network_summary(self) -> Dict:
        """Get network health summary"""
        return {
            'overall_health': self.current_metrics.overall_health.value,
            'health_score': self.current_metrics.health_score,
            'total_nodes': self.current_metrics.total_nodes,
            'connected_nodes': self.current_metrics.connected_nodes,
            'connectivity_ratio': self.current_metrics.connectivity_ratio,
            'average_latency': self.current_metrics.average_latency,
            'packet_loss_rate': self.current_metrics.packet_loss_rate,
            'throughput': self.current_metrics.throughput,
            'active_alerts': len(self.active_alerts),
            'critical_alerts': len([a for a in self.active_alerts.values() if a.level == AlertLevel.CRITICAL]),
            'network_diameter': self.current_metrics.network_diameter,
            'clustering_coefficient': self.current_metrics.clustering_coefficient,
            'timestamp': self.current_metrics.timestamp,
            'monitoring_active': self.is_monitoring
        }
    
    def export_metrics(self, filename: str):
        """Export metrics to file"""
        try:
            data = {
                'current_metrics': {
                    'overall_health': self.current_metrics.overall_health.value,
                    'health_score': self.current_metrics.health_score,
                    'total_nodes': self.current_metrics.total_nodes,
                    'connected_nodes': self.current_metrics.connected_nodes,
                    'connectivity_ratio': self.current_metrics.connectivity_ratio,
                    'average_latency': self.current_metrics.average_latency,
                    'packet_loss_rate': self.current_metrics.packet_loss_rate,
                    'throughput': self.current_metrics.throughput,
                    'timestamp': self.current_metrics.timestamp
                },
                'metrics_history': [
                    {
                        'health_score': m.health_score,
                        'connectivity_ratio': m.connectivity_ratio,
                        'average_latency': m.average_latency,
                        'packet_loss_rate': m.packet_loss_rate,
                        'timestamp': m.timestamp
                    }
                    for m in list(self.metrics_history)
                ],
                'active_alerts': [
                    {
                        'level': a.level.value,
                        'message': a.message,
                        'component': a.component,
                        'timestamp': a.timestamp
                    }
                    for a in self.active_alerts.values()
                ],
                'node_health': {
                    node_id: {
                        'is_online': node.is_online,
                        'health_score': node.health_score,
                        'latency': node.latency,
                        'throughput': node.throughput,
                        'error_rate': node.error_rate,
                        'last_seen': node.last_seen
                    }
                    for node_id, node in self.node_health.items()
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Exported metrics to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
    
    def get_health_report(self) -> Dict:
        """Generate comprehensive health report"""
        return {
            'summary': self.get_network_summary(),
            'detailed_metrics': {
                'connectivity': {
                    'total_nodes': self.current_metrics.total_nodes,
                    'connected_nodes': self.current_metrics.connected_nodes,
                    'connectivity_ratio': self.current_metrics.connectivity_ratio,
                    'edge_count': self.current_metrics.edge_count
                },
                'performance': {
                    'average_latency': self.current_metrics.average_latency,
                    'median_latency': self.current_metrics.median_latency,
                    'latency_variance': self.current_metrics.latency_variance,
                    'throughput': self.current_metrics.throughput,
                    'packet_loss_rate': self.current_metrics.packet_loss_rate
                },
                'topology': {
                    'clustering_coefficient': self.current_metrics.clustering_coefficient,
                    'average_path_length': self.current_metrics.average_path_length,
                    'network_diameter': self.current_metrics.network_diameter
                }
            },
            'alerts': {
                'active_count': len(self.active_alerts),
                'by_level': {
                    level.value: len([a for a in self.active_alerts.values() if a.level == level])
                    for level in AlertLevel
                },
                'recent_alerts': [
                    {
                        'level': a.level.value,
                        'message': a.message,
                        'component': a.component,
                        'age_seconds': time.time() - a.timestamp
                    }
                    for a in list(self.alert_history)[-10:]
                ]
            },
            'trends': {
                'metrics_available': len(self.metrics_history),
                'monitoring_duration': time.time() - self.metrics_history[0].timestamp if self.metrics_history else 0
            },
            'timestamp': time.time()
        }