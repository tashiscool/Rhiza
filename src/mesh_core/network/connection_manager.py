"""
Connection Manager - Manage Peer Connections and Health

Handles the lifecycle of peer connections with:
- Automatic connection establishment and maintenance
- Health monitoring and recovery
- Connection pooling and load balancing
- Graceful degradation under network stress
- M4 Pro optimized connection handling
"""

import asyncio
import time
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import random

logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    """Connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    HANDSHAKING = "handshaking"
    ESTABLISHED = "established"
    DEGRADED = "degraded"
    FAILED = "failed"
    CLOSING = "closing"

@dataclass
class ConnectionMetrics:
    """Metrics for a connection"""
    bytes_sent: int = 0
    bytes_received: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    connection_time: float = 0.0
    last_activity: float = field(default_factory=time.time)
    latency: float = 0.0
    packet_loss: float = 0.0
    bandwidth: float = 0.0
    errors: int = 0
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()
    
    def record_message_sent(self, size: int):
        """Record a sent message"""
        self.messages_sent += 1
        self.bytes_sent += size
        self.update_activity()
    
    def record_message_received(self, size: int):
        """Record a received message"""
        self.messages_received += 1
        self.bytes_received += size
        self.update_activity()
    
    def record_error(self):
        """Record an error"""
        self.errors += 1
    
    def get_throughput(self) -> float:
        """Calculate throughput in bytes per second"""
        if self.connection_time > 0:
            return (self.bytes_sent + self.bytes_received) / self.connection_time
        return 0.0

@dataclass
class ManagedConnection:
    """Represents a managed peer connection"""
    peer_id: str
    host: str
    port: int
    state: ConnectionState = ConnectionState.DISCONNECTED
    connection_obj: Any = None
    last_seen: float = field(default_factory=time.time)
    connection_attempts: int = 0
    max_attempts: int = 5
    retry_delay: float = 30.0
    priority: int = 1  # Higher = more important
    persistent: bool = True
    metrics: ConnectionMetrics = field(default_factory=ConnectionMetrics)
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.metrics.connection_time = time.time()
    
    @property
    def is_connected(self) -> bool:
        """Check if connection is established"""
        return self.state == ConnectionState.ESTABLISHED
    
    @property
    def can_retry(self) -> bool:
        """Check if connection can be retried"""
        return (self.connection_attempts < self.max_attempts and
                time.time() - self.last_seen > self.retry_delay)
    
    def reset_attempts(self):
        """Reset connection attempt counter"""
        self.connection_attempts = 0
    
    def record_attempt(self):
        """Record a connection attempt"""
        self.connection_attempts += 1
        self.last_seen = time.time()

class ConnectionManager:
    """
    Manages all peer connections for a mesh node
    
    Provides intelligent connection management with health monitoring,
    automatic recovery, and performance optimization.
    """
    
    def __init__(self, node_id: str, config: Dict = None):
        self.node_id = node_id
        self.config = config or {}
        
        # Connection management
        self.connections: Dict[str, ManagedConnection] = {}
        self.connection_pool: Dict[str, List[ManagedConnection]] = {}
        self.is_running = False
        
        # Configuration
        self.max_connections = self.config.get('max_connections', 50)
        self.min_connections = self.config.get('min_connections', 3)
        self.health_check_interval = self.config.get('health_check_interval', 30)
        self.connection_timeout = self.config.get('connection_timeout', 10)
        self.idle_timeout = self.config.get('idle_timeout', 300)
        
        # Health monitoring
        self.health_checks_enabled = self.config.get('health_checks_enabled', True)
        self.auto_reconnect = self.config.get('auto_reconnect', True)
        self.load_balancing = self.config.get('load_balancing', True)
        
        # Callbacks
        self.connection_established_callback: Optional[Callable] = None
        self.connection_lost_callback: Optional[Callable] = None
        self.message_handler: Optional[Callable] = None
        
        # Metrics
        self.global_metrics = {
            'total_connections': 0,
            'active_connections': 0,
            'failed_connections': 0,
            'reconnections': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'average_latency': 0.0
        }
        
        logger.info(f"Connection manager initialized for node {self.node_id}")
    
    async def start(self):
        """Start the connection manager"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background tasks
        if self.health_checks_enabled:
            asyncio.create_task(self._health_check_task())
        
        if self.auto_reconnect:
            asyncio.create_task(self._reconnection_task())
        
        asyncio.create_task(self._metrics_update_task())
        
        logger.info("Connection manager started")
    
    async def stop(self):
        """Stop the connection manager"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Close all connections gracefully
        close_tasks = []
        for connection in list(self.connections.values()):
            if connection.is_connected:
                close_tasks.append(self._close_connection(connection))
        
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        self.connections.clear()
        self.connection_pool.clear()
        
        logger.info("Connection manager stopped")
    
    async def add_peer(self, peer_id: str, host: str, port: int, 
                      priority: int = 1, persistent: bool = True,
                      metadata: Dict = None) -> bool:
        """Add a peer for connection management"""
        try:
            if peer_id == self.node_id:
                logger.warning(f"Cannot add self as peer: {peer_id}")
                return False
            
            if peer_id in self.connections:
                logger.info(f"Peer {peer_id} already managed, updating info")
                connection = self.connections[peer_id]
                connection.host = host
                connection.port = port
                connection.priority = priority
                connection.persistent = persistent
                if metadata:
                    connection.metadata.update(metadata)
                return True
            
            # Check connection limits
            if len(self.connections) >= self.max_connections:
                if not await self._make_room_for_connection(priority):
                    logger.warning(f"Cannot add peer {peer_id}: connection limit reached")
                    return False
            
            # Create managed connection
            connection = ManagedConnection(
                peer_id=peer_id,
                host=host,
                port=port,
                priority=priority,
                persistent=persistent,
                metadata=metadata or {}
            )
            
            self.connections[peer_id] = connection
            logger.info(f"Added peer {peer_id} for management")
            
            # Attempt immediate connection if we need more connections
            if len([c for c in self.connections.values() if c.is_connected]) < self.min_connections:
                asyncio.create_task(self._connect_to_peer(connection))
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding peer {peer_id}: {e}")
            return False
    
    async def remove_peer(self, peer_id: str) -> bool:
        """Remove a peer from management"""
        try:
            if peer_id not in self.connections:
                return False
            
            connection = self.connections.pop(peer_id)
            
            # Close connection if active
            if connection.is_connected:
                await self._close_connection(connection)
            
            logger.info(f"Removed peer {peer_id} from management")
            return True
            
        except Exception as e:
            logger.error(f"Error removing peer {peer_id}: {e}")
            return False
    
    async def connect_to_peer(self, peer_id: str) -> bool:
        """Manually connect to a specific peer"""
        if peer_id not in self.connections:
            logger.error(f"Peer {peer_id} not managed")
            return False
        
        connection = self.connections[peer_id]
        return await self._connect_to_peer(connection)
    
    async def disconnect_from_peer(self, peer_id: str) -> bool:
        """Manually disconnect from a specific peer"""
        if peer_id not in self.connections:
            return False
        
        connection = self.connections[peer_id]
        if connection.is_connected:
            await self._close_connection(connection)
            return True
        
        return False
    
    async def send_message(self, peer_id: str, message_data: bytes) -> bool:
        """Send message to a specific peer"""
        try:
            if peer_id not in self.connections:
                logger.warning(f"Peer {peer_id} not managed")
                return False
            
            connection = self.connections[peer_id]
            
            # Ensure connection is established
            if not connection.is_connected:
                if not await self._connect_to_peer(connection):
                    logger.warning(f"Failed to establish connection to {peer_id}")
                    return False
            
            # Send message through the connection
            success = await self._send_via_connection(connection, message_data)
            
            if success:
                connection.metrics.record_message_sent(len(message_data))
                self.global_metrics['bytes_sent'] += len(message_data)
            else:
                connection.metrics.record_error()
                # Mark connection as degraded for potential recovery
                connection.state = ConnectionState.DEGRADED
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending message to {peer_id}: {e}")
            return False
    
    async def broadcast_message(self, message_data: bytes, 
                              exclude: Set[str] = None) -> int:
        """Broadcast message to all connected peers"""
        exclude = exclude or set()
        success_count = 0
        
        connected_peers = [
            conn for conn in self.connections.values()
            if conn.is_connected and conn.peer_id not in exclude
        ]
        
        # Send to all connected peers
        tasks = []
        for connection in connected_peers:
            tasks.append(self._send_via_connection(connection, message_data))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if result is True:
                    success_count += 1
                    connected_peers[i].metrics.record_message_sent(len(message_data))
                    self.global_metrics['bytes_sent'] += len(message_data)
                elif isinstance(result, Exception):
                    connected_peers[i].metrics.record_error()
                    logger.error(f"Broadcast error to {connected_peers[i].peer_id}: {result}")
        
        return success_count
    
    async def _connect_to_peer(self, connection: ManagedConnection) -> bool:
        """Establish connection to a peer"""
        try:
            if connection.state == ConnectionState.CONNECTING:
                return False  # Already connecting
            
            connection.state = ConnectionState.CONNECTING
            connection.record_attempt()
            
            logger.debug(f"Connecting to peer {connection.peer_id} at {connection.host}:{connection.port}")
            
            # Attempt connection with timeout
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(connection.host, connection.port),
                    timeout=self.connection_timeout
                )
                
                connection.connection_obj = (reader, writer)
                connection.state = ConnectionState.CONNECTED
                
                # Perform handshake (simplified)
                if await self._perform_handshake(connection):
                    connection.state = ConnectionState.ESTABLISHED
                    connection.reset_attempts()
                    connection.metrics.connection_time = time.time() - connection.metrics.connection_time
                    
                    self.global_metrics['total_connections'] += 1
                    self.global_metrics['active_connections'] += 1
                    
                    logger.info(f"Connected to peer {connection.peer_id}")
                    
                    # Start listening for messages
                    asyncio.create_task(self._listen_to_peer(connection))
                    
                    # Notify callback
                    if self.connection_established_callback:
                        asyncio.create_task(
                            self.connection_established_callback(connection.peer_id)
                        )
                    
                    return True
                else:
                    await self._close_connection(connection)
                    connection.state = ConnectionState.FAILED
                    return False
                
            except asyncio.TimeoutError:
                logger.warning(f"Connection timeout to {connection.peer_id}")
                connection.state = ConnectionState.FAILED
                return False
            
        except Exception as e:
            logger.error(f"Error connecting to peer {connection.peer_id}: {e}")
            connection.state = ConnectionState.FAILED
            connection.metrics.record_error()
            return False
    
    async def _perform_handshake(self, connection: ManagedConnection) -> bool:
        """Perform handshake with peer (simplified)"""
        try:
            connection.state = ConnectionState.HANDSHAKING
            
            # In a real implementation, this would exchange protocol information
            # For now, we just simulate a successful handshake
            await asyncio.sleep(0.1)  # Simulate handshake delay
            
            return True
            
        except Exception as e:
            logger.error(f"Handshake failed with {connection.peer_id}: {e}")
            return False
    
    async def _listen_to_peer(self, connection: ManagedConnection):
        """Listen for messages from a peer"""
        try:
            reader, writer = connection.connection_obj
            
            while (connection.is_connected and 
                   self.is_running and 
                   not reader.at_eof()):
                
                try:
                    # Read message (simplified protocol)
                    data = await asyncio.wait_for(reader.read(4096), timeout=1.0)
                    
                    if not data:
                        break
                    
                    connection.metrics.record_message_received(len(data))
                    self.global_metrics['bytes_received'] += len(data)
                    
                    # Handle message
                    if self.message_handler:
                        asyncio.create_task(
                            self.message_handler(connection.peer_id, data)
                        )
                    
                except asyncio.TimeoutError:
                    # No data received, continue listening
                    continue
                except Exception as e:
                    logger.error(f"Error reading from {connection.peer_id}: {e}")
                    break
            
        except Exception as e:
            logger.error(f"Listen error for {connection.peer_id}: {e}")
        finally:
            # Connection lost
            if connection.is_connected:
                await self._handle_connection_lost(connection)
    
    async def _send_via_connection(self, connection: ManagedConnection, 
                                  data: bytes) -> bool:
        """Send data via a connection"""
        try:
            if not connection.is_connected or not connection.connection_obj:
                return False
            
            reader, writer = connection.connection_obj
            
            writer.write(data)
            await writer.drain()
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending to {connection.peer_id}: {e}")
            connection.metrics.record_error()
            return False
    
    async def _close_connection(self, connection: ManagedConnection):
        """Close a connection gracefully"""
        try:
            if connection.connection_obj:
                reader, writer = connection.connection_obj
                
                if not writer.is_closing():
                    writer.close()
                    await writer.wait_closed()
                
                connection.connection_obj = None
            
            if connection.state == ConnectionState.ESTABLISHED:
                self.global_metrics['active_connections'] -= 1
            
            connection.state = ConnectionState.DISCONNECTED
            
        except Exception as e:
            logger.error(f"Error closing connection to {connection.peer_id}: {e}")
    
    async def _handle_connection_lost(self, connection: ManagedConnection):
        """Handle lost connection"""
        await self._close_connection(connection)
        
        logger.warning(f"Connection lost to {connection.peer_id}")
        
        # Notify callback
        if self.connection_lost_callback:
            asyncio.create_task(
                self.connection_lost_callback(connection.peer_id)
            )
        
        # Schedule reconnection if persistent
        if connection.persistent and self.auto_reconnect:
            connection.retry_delay = min(300, connection.retry_delay * 1.5)  # Exponential backoff
    
    async def _make_room_for_connection(self, priority: int) -> bool:
        """Make room for a new connection by closing lower priority ones"""
        if not self.connections:
            return True
        
        # Find lowest priority connected peer
        connected = [c for c in self.connections.values() if c.is_connected]
        if not connected:
            return True
        
        lowest_priority_conn = min(connected, key=lambda c: c.priority)
        
        if lowest_priority_conn.priority < priority:
            await self._close_connection(lowest_priority_conn)
            logger.info(f"Closed connection to {lowest_priority_conn.peer_id} to make room")
            return True
        
        return False
    
    async def _health_check_task(self):
        """Background health check task"""
        while self.is_running:
            try:
                current_time = time.time()
                
                for connection in list(self.connections.values()):
                    if connection.is_connected:
                        # Check for idle connections
                        idle_time = current_time - connection.metrics.last_activity
                        if idle_time > self.idle_timeout:
                            logger.info(f"Closing idle connection to {connection.peer_id}")
                            await self._close_connection(connection)
                        
                        # Check for degraded connections
                        if connection.state == ConnectionState.DEGRADED:
                            # Attempt recovery
                            await self._close_connection(connection)
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _reconnection_task(self):
        """Background reconnection task"""
        while self.is_running:
            try:
                # Find peers that need reconnection
                to_reconnect = [
                    conn for conn in self.connections.values()
                    if (conn.persistent and 
                        not conn.is_connected and 
                        conn.can_retry and
                        conn.state not in [ConnectionState.CONNECTING, ConnectionState.HANDSHAKING])
                ]
                
                # Sort by priority
                to_reconnect.sort(key=lambda c: c.priority, reverse=True)
                
                # Limit concurrent reconnections
                max_concurrent = min(3, len(to_reconnect))
                reconnection_tasks = []
                
                for connection in to_reconnect[:max_concurrent]:
                    reconnection_tasks.append(self._connect_to_peer(connection))
                
                if reconnection_tasks:
                    results = await asyncio.gather(*reconnection_tasks, return_exceptions=True)
                    success_count = sum(1 for r in results if r is True)
                    self.global_metrics['reconnections'] += success_count
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Reconnection task error: {e}")
                await asyncio.sleep(10)
    
    async def _metrics_update_task(self):
        """Background metrics update task"""
        while self.is_running:
            try:
                # Update global metrics
                active_connections = [c for c in self.connections.values() if c.is_connected]
                self.global_metrics['active_connections'] = len(active_connections)
                
                # Calculate average latency
                if active_connections:
                    total_latency = sum(c.metrics.latency for c in active_connections)
                    self.global_metrics['average_latency'] = total_latency / len(active_connections)
                else:
                    self.global_metrics['average_latency'] = 0.0
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Metrics update error: {e}")
                await asyncio.sleep(30)
    
    def get_connection_info(self, peer_id: str) -> Optional[Dict]:
        """Get information about a specific connection"""
        if peer_id not in self.connections:
            return None
        
        connection = self.connections[peer_id]
        return {
            'peer_id': connection.peer_id,
            'host': connection.host,
            'port': connection.port,
            'state': connection.state.value,
            'priority': connection.priority,
            'persistent': connection.persistent,
            'connection_attempts': connection.connection_attempts,
            'last_seen': connection.last_seen,
            'metrics': {
                'bytes_sent': connection.metrics.bytes_sent,
                'bytes_received': connection.metrics.bytes_received,
                'messages_sent': connection.metrics.messages_sent,
                'messages_received': connection.metrics.messages_received,
                'latency': connection.metrics.latency,
                'throughput': connection.metrics.get_throughput(),
                'errors': connection.metrics.errors,
                'connection_time': connection.metrics.connection_time,
                'last_activity': connection.metrics.last_activity
            },
            'metadata': connection.metadata
        }
    
    def get_all_connections(self) -> Dict[str, Dict]:
        """Get information about all connections"""
        return {
            peer_id: self.get_connection_info(peer_id)
            for peer_id in self.connections
        }
    
    def get_connected_peers(self) -> List[str]:
        """Get list of connected peer IDs"""
        return [
            conn.peer_id for conn in self.connections.values()
            if conn.is_connected
        ]
    
    def set_connection_established_callback(self, callback: Callable):
        """Set callback for connection established events"""
        self.connection_established_callback = callback
    
    def set_connection_lost_callback(self, callback: Callable):
        """Set callback for connection lost events"""
        self.connection_lost_callback = callback
    
    def set_message_handler(self, handler: Callable):
        """Set message handler"""
        self.message_handler = handler
    
    def get_metrics(self) -> Dict:
        """Get connection manager metrics"""
        return {
            **self.global_metrics,
            'managed_peers': len(self.connections),
            'connection_limit': self.max_connections,
            'is_running': self.is_running
        }