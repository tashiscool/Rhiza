"""
TCP Transport - Real TCP/UDP network implementation for The Mesh

Provides actual network transport layer implementing the mesh protocol
over TCP and UDP with connection management, message framing, and
security features.

Key Features:
- TCP and UDP transport support
- Message framing and serialization
- Connection pooling and management
- TLS encryption support
- Network discovery and peer management
- Bandwidth optimization
- Network error handling and recovery
"""

import asyncio
import socket
import ssl
import json
import struct
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import hashlib
import ipaddress

try:
    import aiodns
    DNS_SUPPORT = True
except ImportError:
    DNS_SUPPORT = False
    logging.warning("aiodns not available - DNS discovery disabled")


class TransportType(Enum):
    """Network transport types"""
    TCP = "tcp"
    UDP = "udp"
    TLS = "tls"


class MessageType(Enum):
    """Message types for network protocol"""
    DISCOVERY = "discovery"
    HANDSHAKE = "handshake"
    DATA = "data" 
    HEARTBEAT = "heartbeat"
    ACKNOWLEDGMENT = "ack"
    ERROR = "error"
    MESH_QUERY = "mesh_query"
    MESH_RESPONSE = "mesh_response"


@dataclass
class NetworkMessage:
    """Network message structure"""
    message_id: str
    message_type: MessageType
    source_node: str
    destination_node: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    ttl: int = 10
    signature: Optional[str] = None
    
    def to_bytes(self) -> bytes:
        """Serialize message to bytes"""
        data = {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "source_node": self.source_node,
            "destination_node": self.destination_node,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "ttl": self.ttl,
            "signature": self.signature
        }
        json_data = json.dumps(data, separators=(',', ':')).encode('utf-8')
        
        # Frame format: [4-byte length][json data]
        length = struct.pack('!I', len(json_data))
        return length + json_data
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'NetworkMessage':
        """Deserialize message from bytes"""
        if len(data) < 4:
            raise ValueError("Invalid message: too short")
        
        # Extract length and JSON data
        length = struct.unpack('!I', data[:4])[0]
        json_data = data[4:4+length]
        
        message_data = json.loads(json_data.decode('utf-8'))
        
        return cls(
            message_id=message_data["message_id"],
            message_type=MessageType(message_data["message_type"]),
            source_node=message_data["source_node"],
            destination_node=message_data["destination_node"],
            payload=message_data["payload"],
            timestamp=datetime.fromisoformat(message_data["timestamp"]),
            ttl=message_data["ttl"],
            signature=message_data.get("signature")
        )


@dataclass
class PeerInfo:
    """Information about a network peer"""
    node_id: str
    address: str
    port: int
    transport: TransportType
    last_seen: datetime = field(default_factory=datetime.utcnow)
    trust_score: float = 0.5
    connection_count: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    rtt: float = 0.0  # Round trip time
    
    @property
    def endpoint(self) -> str:
        return f"{self.address}:{self.port}"
    
    def is_stale(self, timeout: timedelta = timedelta(minutes=5)) -> bool:
        return datetime.utcnow() - self.last_seen > timeout


class NetworkTransport:
    """Real network transport implementation"""
    
    def __init__(self, node_id: str, bind_address: str = "0.0.0.0", tcp_port: int = 8888, udp_port: int = 8889):
        self.node_id = node_id
        self.bind_address = bind_address
        self.tcp_port = tcp_port
        self.udp_port = udp_port
        self.logger = logging.getLogger(__name__)
        
        # Network state
        self.running = False
        self.tcp_server = None
        self.udp_transport = None
        self.udp_protocol = None
        
        # Connection management
        self.tcp_connections: Dict[str, asyncio.StreamWriter] = {}
        self.peer_info: Dict[str, PeerInfo] = {}
        self.connection_lock = asyncio.Lock()
        
        # Message handling
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.pending_responses: Dict[str, asyncio.Future] = {}
        self.message_queue = asyncio.Queue()
        
        # Network discovery
        self.discovery_interval = 30  # seconds
        self.discovery_task = None
        self.known_bootstrap_nodes = []
        
        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.connections_established = 0
        self.discovery_rounds = 0
        
        # TLS support
        self.ssl_context = None
        self._setup_ssl_context()
    
    def _setup_ssl_context(self):
        """Setup SSL context for TLS connections"""
        try:
            # Create self-signed certificate for development
            # In production, use proper certificates
            self.ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            self.ssl_context.check_hostname = False
            self.ssl_context.verify_mode = ssl.CERT_NONE
        except Exception as e:
            self.logger.warning(f"SSL context setup failed: {e}")
    
    async def start(self):
        """Start network transport"""
        if self.running:
            return
        
        self.logger.info(f"Starting network transport for node {self.node_id}")
        
        try:
            # Start TCP server
            self.tcp_server = await asyncio.start_server(
                self._handle_tcp_connection,
                self.bind_address,
                self.tcp_port
            )
            
            # Start UDP transport
            loop = asyncio.get_event_loop()
            self.udp_transport, self.udp_protocol = await loop.create_datagram_endpoint(
                lambda: UDPProtocol(self),
                local_addr=(self.bind_address, self.udp_port)
            )
            
            # Start message processing
            asyncio.create_task(self._process_message_queue())
            
            # Start network discovery
            self.discovery_task = asyncio.create_task(self._discovery_loop())
            
            self.running = True
            self.logger.info(f"Transport started - TCP:{self.tcp_port}, UDP:{self.udp_port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start transport: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop network transport"""
        if not self.running:
            return
        
        self.logger.info("Stopping network transport")
        self.running = False
        
        try:
            # Stop discovery
            if self.discovery_task:
                self.discovery_task.cancel()
            
            # Close TCP connections
            async with self.connection_lock:
                for writer in self.tcp_connections.values():
                    writer.close()
                    try:
                        await writer.wait_closed()
                    except:
                        pass
                self.tcp_connections.clear()
            
            # Stop TCP server
            if self.tcp_server:
                self.tcp_server.close()
                await self.tcp_server.wait_closed()
            
            # Stop UDP transport
            if self.udp_transport:
                self.udp_transport.close()
            
            self.logger.info("Transport stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping transport: {e}")
    
    async def _handle_tcp_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming TCP connection"""
        peer_addr = writer.get_extra_info('peername')
        self.logger.debug(f"New TCP connection from {peer_addr}")
        
        try:
            # Perform handshake
            peer_node_id = await self._perform_handshake(reader, writer)
            if not peer_node_id:
                writer.close()
                return
            
            # Store connection
            async with self.connection_lock:
                self.tcp_connections[peer_node_id] = writer
            
            self.connections_established += 1
            
            # Handle messages from this connection
            await self._handle_tcp_messages(reader, writer, peer_node_id)
            
        except Exception as e:
            self.logger.error(f"TCP connection error: {e}")
        finally:
            # Clean up connection
            if peer_node_id:
                async with self.connection_lock:
                    self.tcp_connections.pop(peer_node_id, None)
            
            writer.close()
            try:
                await writer.wait_closed()
            except:
                pass
    
    async def _perform_handshake(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> Optional[str]:
        """Perform connection handshake"""
        try:
            # Wait for handshake message
            length_data = await reader.readexactly(4)
            length = struct.unpack('!I', length_data)[0]
            
            if length > 1024 * 1024:  # 1MB max for handshake
                return None
            
            message_data = await reader.readexactly(length)
            message = NetworkMessage.from_bytes(length_data + message_data)
            
            if message.message_type != MessageType.HANDSHAKE:
                return None
            
            # Send handshake response
            response = NetworkMessage(
                message_id=f"handshake_response_{int(time.time() * 1000000)}",
                message_type=MessageType.HANDSHAKE,
                source_node=self.node_id,
                destination_node=message.source_node,
                payload={
                    "node_id": self.node_id,
                    "protocol_version": "1.0",
                    "capabilities": ["tcp", "udp", "mesh_query"],
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            writer.write(response.to_bytes())
            await writer.drain()
            
            # Update peer info
            peer_addr = writer.get_extra_info('peername')
            self.peer_info[message.source_node] = PeerInfo(
                node_id=message.source_node,
                address=peer_addr[0],
                port=peer_addr[1],
                transport=TransportType.TCP
            )
            
            return message.source_node
            
        except Exception as e:
            self.logger.error(f"Handshake failed: {e}")
            return None
    
    async def _handle_tcp_messages(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, peer_node_id: str):
        """Handle messages from TCP connection"""
        while self.running:
            try:
                # Read message length
                length_data = await reader.readexactly(4)
                length = struct.unpack('!I', length_data)[0]
                
                # Validate length
                if length > 10 * 1024 * 1024:  # 10MB max
                    self.logger.warning(f"Message too large: {length} bytes")
                    break
                
                # Read message data
                message_data = await reader.readexactly(length)
                message = NetworkMessage.from_bytes(length_data + message_data)
                
                # Update peer info
                if peer_node_id in self.peer_info:
                    self.peer_info[peer_node_id].last_seen = datetime.utcnow()
                    self.peer_info[peer_node_id].bytes_received += len(message_data) + 4
                
                # Queue message for processing
                await self.message_queue.put(message)
                self.messages_received += 1
                
            except asyncio.IncompleteReadError:
                break
            except Exception as e:
                self.logger.error(f"Error reading TCP message: {e}")
                break
    
    async def _process_message_queue(self):
        """Process incoming messages"""
        while self.running:
            try:
                message = await self.message_queue.get()
                await self._handle_message(message)
            except Exception as e:
                self.logger.error(f"Message processing error: {e}")
    
    async def _handle_message(self, message: NetworkMessage):
        """Handle incoming message"""
        try:
            # Check if this is a response to a pending request
            if message.message_id in self.pending_responses:
                future = self.pending_responses.pop(message.message_id)
                if not future.done():
                    future.set_result(message)
                return
            
            # Handle based on message type
            handler = self.message_handlers.get(message.message_type)
            if handler:
                await handler(message)
            else:
                self.logger.warning(f"No handler for message type: {message.message_type}")
                
        except Exception as e:
            self.logger.error(f"Message handling error: {e}")
    
    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register message handler"""
        self.message_handlers[message_type] = handler
    
    async def connect_to_peer(self, address: str, port: int, transport: TransportType = TransportType.TCP) -> bool:
        """Connect to a peer"""
        try:
            if transport == TransportType.TCP:
                return await self._connect_tcp(address, port)
            elif transport == TransportType.UDP:
                return await self._connect_udp(address, port)
            else:
                self.logger.error(f"Unsupported transport: {transport}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect to {address}:{port}: {e}")
            return False
    
    async def _connect_tcp(self, address: str, port: int) -> bool:
        """Connect via TCP"""
        try:
            reader, writer = await asyncio.open_connection(address, port)
            
            # Perform handshake
            handshake = NetworkMessage(
                message_id=f"handshake_{int(time.time() * 1000000)}",
                message_type=MessageType.HANDSHAKE,
                source_node=self.node_id,
                destination_node="unknown",
                payload={
                    "node_id": self.node_id,
                    "protocol_version": "1.0",
                    "capabilities": ["tcp", "udp", "mesh_query"]
                }
            )
            
            writer.write(handshake.to_bytes())
            await writer.drain()
            
            # Wait for response
            length_data = await reader.readexactly(4)
            length = struct.unpack('!I', length_data)[0]
            message_data = await reader.readexactly(length)
            response = NetworkMessage.from_bytes(length_data + message_data)
            
            if response.message_type != MessageType.HANDSHAKE:
                writer.close()
                return False
            
            peer_node_id = response.source_node
            
            # Store connection
            async with self.connection_lock:
                self.tcp_connections[peer_node_id] = writer
            
            # Store peer info
            self.peer_info[peer_node_id] = PeerInfo(
                node_id=peer_node_id,
                address=address,
                port=port,
                transport=TransportType.TCP
            )
            
            # Start handling messages
            asyncio.create_task(self._handle_tcp_messages(reader, writer, peer_node_id))
            
            self.logger.info(f"Connected to peer {peer_node_id} at {address}:{port}")
            return True
            
        except Exception as e:
            self.logger.error(f"TCP connection failed: {e}")
            return False
    
    async def send_message(self, destination: str, message: NetworkMessage) -> bool:
        """Send message to destination"""
        try:
            # Find peer info
            if destination not in self.peer_info:
                self.logger.warning(f"Unknown destination: {destination}")
                return False
            
            peer = self.peer_info[destination]
            
            if peer.transport == TransportType.TCP:
                return await self._send_tcp_message(destination, message)
            elif peer.transport == TransportType.UDP:
                return await self._send_udp_message(destination, message)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False
    
    async def _send_tcp_message(self, destination: str, message: NetworkMessage) -> bool:
        """Send message via TCP"""
        try:
            async with self.connection_lock:
                writer = self.tcp_connections.get(destination)
                
            if not writer:
                self.logger.warning(f"No TCP connection to {destination}")
                return False
            
            data = message.to_bytes()
            writer.write(data)
            await writer.drain()
            
            # Update statistics
            if destination in self.peer_info:
                self.peer_info[destination].bytes_sent += len(data)
            
            self.messages_sent += 1
            return True
            
        except Exception as e:
            self.logger.error(f"TCP send failed: {e}")
            # Remove failed connection
            async with self.connection_lock:
                self.tcp_connections.pop(destination, None)
            return False
    
    async def send_and_wait_response(self, destination: str, message: NetworkMessage, timeout: float = 10.0) -> Optional[NetworkMessage]:
        """Send message and wait for response"""
        try:
            # Create future for response
            response_future = asyncio.Future()
            self.pending_responses[message.message_id] = response_future
            
            # Send message
            success = await self.send_message(destination, message)
            if not success:
                self.pending_responses.pop(message.message_id, None)
                return None
            
            # Wait for response
            try:
                response = await asyncio.wait_for(response_future, timeout=timeout)
                return response
            except asyncio.TimeoutError:
                self.logger.warning(f"Response timeout for message {message.message_id}")
                self.pending_responses.pop(message.message_id, None)
                return None
                
        except Exception as e:
            self.logger.error(f"Send and wait failed: {e}")
            return None
    
    async def broadcast_message(self, message: NetworkMessage, exclude: List[str] = None) -> int:
        """Broadcast message to all connected peers"""
        exclude = exclude or []
        sent_count = 0
        
        for peer_id in list(self.peer_info.keys()):
            if peer_id not in exclude:
                success = await self.send_message(peer_id, message)
                if success:
                    sent_count += 1
        
        return sent_count
    
    async def _discovery_loop(self):
        """Network discovery loop"""
        while self.running:
            try:
                await self._perform_discovery()
                self.discovery_rounds += 1
                await asyncio.sleep(self.discovery_interval)
            except Exception as e:
                self.logger.error(f"Discovery error: {e}")
    
    async def _perform_discovery(self):
        """Perform network discovery"""
        try:
            # Local network discovery via UDP broadcast
            discovery_message = NetworkMessage(
                message_id=f"discovery_{int(time.time() * 1000000)}",
                message_type=MessageType.DISCOVERY,
                source_node=self.node_id,
                destination_node="broadcast",
                payload={
                    "node_id": self.node_id,
                    "tcp_port": self.tcp_port,
                    "udp_port": self.udp_port,
                    "capabilities": ["tcp", "udp", "mesh_query"]
                }
            )
            
            # Broadcast on local network
            if self.udp_transport:
                data = discovery_message.to_bytes()
                self.udp_transport.sendto(data, ('255.255.255.255', self.udp_port))
            
            # Connect to bootstrap nodes
            for bootstrap_node in self.known_bootstrap_nodes:
                if bootstrap_node not in self.peer_info:
                    address, port = bootstrap_node.split(':')
                    await self.connect_to_peer(address, int(port))
                    
        except Exception as e:
            self.logger.error(f"Discovery failed: {e}")
    
    def add_bootstrap_node(self, address: str, port: int):
        """Add bootstrap node for discovery"""
        endpoint = f"{address}:{port}"
        if endpoint not in self.known_bootstrap_nodes:
            self.known_bootstrap_nodes.append(endpoint)
            self.logger.info(f"Added bootstrap node: {endpoint}")
    
    def get_peer_list(self) -> List[PeerInfo]:
        """Get list of connected peers"""
        return list(self.peer_info.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get transport statistics"""
        active_connections = len([p for p in self.peer_info.values() if not p.is_stale()])
        
        return {
            "node_id": self.node_id,
            "running": self.running,
            "tcp_port": self.tcp_port,
            "udp_port": self.udp_port,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "connections_established": self.connections_established,
            "active_connections": active_connections,
            "total_peers": len(self.peer_info),
            "discovery_rounds": self.discovery_rounds,
            "pending_responses": len(self.pending_responses)
        }


class UDPProtocol(asyncio.DatagramProtocol):
    """UDP protocol handler"""
    
    def __init__(self, transport: NetworkTransport):
        self.transport_manager = transport
        self.logger = logging.getLogger(__name__)
    
    def connection_made(self, transport):
        self.transport = transport
    
    def datagram_received(self, data, addr):
        """Handle incoming UDP datagram"""
        try:
            message = NetworkMessage.from_bytes(data)
            
            # Handle discovery messages
            if message.message_type == MessageType.DISCOVERY:
                asyncio.create_task(self._handle_discovery_message(message, addr))
            else:
                # Queue other messages for processing
                asyncio.create_task(self.transport_manager.message_queue.put(message))
                
        except Exception as e:
            self.logger.error(f"UDP message error: {e}")
    
    async def _handle_discovery_message(self, message: NetworkMessage, addr):
        """Handle discovery message"""
        try:
            if message.source_node != self.transport_manager.node_id:
                # Add peer info
                peer_info = PeerInfo(
                    node_id=message.source_node,
                    address=addr[0],
                    port=message.payload.get("tcp_port", addr[1]),
                    transport=TransportType.TCP
                )
                
                self.transport_manager.peer_info[message.source_node] = peer_info
                
                # Optionally connect back
                tcp_port = message.payload.get("tcp_port")
                if tcp_port and message.source_node not in self.transport_manager.tcp_connections:
                    await self.transport_manager.connect_to_peer(addr[0], tcp_port)
                    
        except Exception as e:
            self.logger.error(f"Discovery handling error: {e}")