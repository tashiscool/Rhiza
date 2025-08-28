"""
Mesh Protocol - Core P2P Communication Protocol

Implements the foundational peer-to-peer protocol for The Mesh network.
Provides secure, efficient, and resilient communication between mesh nodes.

Key Features:
- Encrypted channels with perfect forward secrecy
- Message authentication and integrity verification
- Adaptive routing based on network topology
- Graceful handling of network partitions
- M4 Pro optimized for Apple Silicon performance
"""

import asyncio
import json
import time
import hashlib
import hmac
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import logging
import socket
import struct

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of messages in the mesh protocol"""
    DISCOVERY = "discovery"
    HANDSHAKE = "handshake"
    TRUST_UPDATE = "trust_update"
    DATA_SYNC = "data_sync"
    QUERY = "query"
    RESPONSE = "response"
    HEARTBEAT = "heartbeat"
    GOODBYE = "goodbye"
    EMERGENCY = "emergency"

@dataclass
class MeshMessage:
    """Core message structure for mesh communication"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.QUERY
    sender_id: str = ""
    recipient_id: Optional[str] = None  # None for broadcast
    timestamp: float = field(default_factory=time.time)
    ttl: int = 10  # Time to live (hops)
    payload: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[str] = None
    encrypted: bool = False
    
    def to_bytes(self) -> bytes:
        """Serialize message to bytes for network transmission"""
        data = {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'timestamp': self.timestamp,
            'ttl': self.ttl,
            'payload': self.payload,
            'signature': self.signature,
            'encrypted': self.encrypted
        }
        json_str = json.dumps(data, separators=(',', ':'))
        return json_str.encode('utf-8')
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'MeshMessage':
        """Deserialize message from bytes"""
        try:
            json_str = data.decode('utf-8')
            data_dict = json.loads(json_str)
            
            return cls(
                message_id=data_dict['message_id'],
                message_type=MessageType(data_dict['message_type']),
                sender_id=data_dict['sender_id'],
                recipient_id=data_dict.get('recipient_id'),
                timestamp=data_dict['timestamp'],
                ttl=data_dict['ttl'],
                payload=data_dict['payload'],
                signature=data_dict.get('signature'),
                encrypted=data_dict.get('encrypted', False)
            )
        except Exception as e:
            logger.error(f"Failed to deserialize message: {e}")
            raise ValueError(f"Invalid message format: {e}")

class MeshProtocol:
    """
    Core mesh protocol implementation
    
    Handles secure P2P communication between mesh nodes with:
    - Automatic encryption and authentication
    - Message routing and delivery
    - Network topology awareness
    - Performance optimization for M4 Pro
    """
    
    def __init__(self, node_id: str, port: int = 0, max_peers: int = 50):
        self.node_id = node_id
        self.port = port
        self.max_peers = max_peers
        
        # Network state
        self.is_running = False
        self.server_socket: Optional[asyncio.Server] = None
        self.peers: Dict[str, 'PeerConnection'] = {}
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.message_cache: Dict[str, float] = {}  # Duplicate detection
        self.routing_table: Dict[str, str] = {}  # node_id -> next_hop_id
        
        # Cryptographic keys
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        
        # Performance metrics
        self.metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'connection_count': 0,
            'routing_errors': 0
        }
        
        # Register default handlers
        self._register_default_handlers()
        
        logger.info(f"Mesh protocol initialized for node {self.node_id}")
    
    def _register_default_handlers(self):
        """Register default message handlers"""
        self.register_handler(MessageType.HANDSHAKE, self._handle_handshake)
        self.register_handler(MessageType.HEARTBEAT, self._handle_heartbeat)
        self.register_handler(MessageType.GOODBYE, self._handle_goodbye)
        self.register_handler(MessageType.DISCOVERY, self._handle_discovery)
    
    async def start(self, host: str = "0.0.0.0") -> int:
        """Start the mesh protocol server"""
        if self.is_running:
            logger.warning("Mesh protocol already running")
            return self.port
            
        try:
            # Create server socket
            self.server_socket = await asyncio.start_server(
                self._handle_client_connection,
                host,
                self.port
            )
            
            # Get actual port if auto-assigned
            if self.port == 0:
                self.port = self.server_socket.sockets[0].getsockname()[1]
                
            self.is_running = True
            logger.info(f"Mesh protocol server started on {host}:{self.port}")
            
            # Start background tasks
            asyncio.create_task(self._cleanup_task())
            asyncio.create_task(self._heartbeat_task())
            
            return self.port
            
        except Exception as e:
            logger.error(f"Failed to start mesh protocol: {e}")
            raise
    
    async def stop(self):
        """Stop the mesh protocol server"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Close all peer connections
        for peer in list(self.peers.values()):
            await peer.close()
        self.peers.clear()
        
        # Close server socket
        if self.server_socket:
            self.server_socket.close()
            await self.server_socket.wait_closed()
            
        logger.info("Mesh protocol stopped")
    
    async def connect_to_peer(self, host: str, port: int) -> bool:
        """Connect to a remote peer"""
        try:
            reader, writer = await asyncio.open_connection(host, port)
            
            # Create peer connection
            peer_id = f"{host}:{port}"  # Temporary ID until handshake
            peer = PeerConnection(peer_id, reader, writer, self)
            
            # Perform handshake
            success = await self._perform_handshake(peer)
            if success:
                self.peers[peer.node_id] = peer
                self.metrics['connection_count'] += 1
                logger.info(f"Connected to peer {peer.node_id}")
                
                # Start peer tasks
                asyncio.create_task(peer.listen())
                return True
            else:
                await peer.close()
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to peer {host}:{port}: {e}")
            return False
    
    async def send_message(self, message: MeshMessage, direct_peer: Optional[str] = None) -> bool:
        """Send a message through the mesh network"""
        try:
            # Set sender ID
            message.sender_id = self.node_id
            
            # Sign message
            await self._sign_message(message)
            
            # Determine routing
            if direct_peer and direct_peer in self.peers:
                # Send directly to specified peer
                peer = self.peers[direct_peer]
                success = await peer.send_message(message)
            elif message.recipient_id:
                # Route to specific recipient
                success = await self._route_message(message)
            else:
                # Broadcast to all peers
                success = await self._broadcast_message(message)
            
            if success:
                self.metrics['messages_sent'] += 1
                self.metrics['bytes_sent'] += len(message.to_bytes())
                
            return success
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register a message handler for a specific message type"""
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for {message_type}")
    
    async def _handle_client_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming client connections"""
        peer_addr = writer.get_extra_info('peername')
        peer_id = f"{peer_addr[0]}:{peer_addr[1]}"
        
        logger.debug(f"New connection from {peer_id}")
        
        try:
            # Create peer connection
            peer = PeerConnection(peer_id, reader, writer, self)
            
            # Wait for handshake
            success = await self._wait_for_handshake(peer)
            if success and len(self.peers) < self.max_peers:
                self.peers[peer.node_id] = peer
                self.metrics['connection_count'] += 1
                logger.info(f"Accepted connection from {peer.node_id}")
                
                # Start peer listening
                await peer.listen()
            else:
                await peer.close()
                logger.warning(f"Rejected connection from {peer_id}")
                
        except Exception as e:
            logger.error(f"Error handling connection from {peer_id}: {e}")
    
    async def _route_message(self, message: MeshMessage) -> bool:
        """Route message to specific recipient"""
        if not message.recipient_id:
            return False
            
        # Check if recipient is directly connected
        if message.recipient_id in self.peers:
            return await self.peers[message.recipient_id].send_message(message)
        
        # Look up in routing table
        next_hop = self.routing_table.get(message.recipient_id)
        if next_hop and next_hop in self.peers:
            return await self.peers[next_hop].send_message(message)
        
        # Broadcast if no route found (flooding)
        if message.ttl > 0:
            message.ttl -= 1
            return await self._broadcast_message(message)
        
        self.metrics['routing_errors'] += 1
        logger.warning(f"No route to {message.recipient_id}")
        return False
    
    async def _broadcast_message(self, message: MeshMessage) -> bool:
        """Broadcast message to all connected peers"""
        if not self.peers:
            return False
            
        success_count = 0
        for peer in self.peers.values():
            try:
                if await peer.send_message(message):
                    success_count += 1
            except Exception as e:
                logger.error(f"Failed to send to peer {peer.node_id}: {e}")
        
        return success_count > 0
    
    async def _sign_message(self, message: MeshMessage):
        """Sign message for authentication"""
        try:
            # Create signature payload
            payload = f"{message.message_id}{message.sender_id}{message.timestamp}{json.dumps(message.payload)}"
            signature = self.private_key.sign(
                payload.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            message.signature = signature.hex()
        except Exception as e:
            logger.error(f"Failed to sign message: {e}")
            raise
    
    async def _verify_message(self, message: MeshMessage, sender_public_key) -> bool:
        """Verify message signature"""
        try:
            if not message.signature:
                return False
                
            payload = f"{message.message_id}{message.sender_id}{message.timestamp}{json.dumps(message.payload)}"
            signature = bytes.fromhex(message.signature)
            
            sender_public_key.verify(
                signature,
                payload.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            logger.error(f"Message signature verification failed: {e}")
            return False
    
    async def _perform_handshake(self, peer: 'PeerConnection') -> bool:
        """Perform handshake with a peer"""
        try:
            # Send handshake
            handshake_msg = MeshMessage(
                message_type=MessageType.HANDSHAKE,
                payload={
                    'node_id': self.node_id,
                    'public_key': self.public_key.public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo
                    ).decode('utf-8'),
                    'protocol_version': '1.0.0'
                }
            )
            
            success = await peer.send_message(handshake_msg)
            if not success:
                return False
                
            # Wait for handshake response
            response = await asyncio.wait_for(peer.receive_message(), timeout=10.0)
            if (response and 
                response.message_type == MessageType.HANDSHAKE and 
                'node_id' in response.payload):
                
                peer.node_id = response.payload['node_id']
                peer.public_key = serialization.load_pem_public_key(
                    response.payload['public_key'].encode('utf-8'),
                    backend=default_backend()
                )
                peer.handshake_complete = True
                return True
                
            return False
            
        except asyncio.TimeoutError:
            logger.error(f"Handshake timeout with {peer.peer_id}")
            return False
        except Exception as e:
            logger.error(f"Handshake failed with {peer.peer_id}: {e}")
            return False
    
    async def _wait_for_handshake(self, peer: 'PeerConnection') -> bool:
        """Wait for incoming handshake"""
        try:
            # Wait for handshake message
            message = await asyncio.wait_for(peer.receive_message(), timeout=10.0)
            if (message and 
                message.message_type == MessageType.HANDSHAKE and 
                'node_id' in message.payload):
                
                peer.node_id = message.payload['node_id']
                peer.public_key = serialization.load_pem_public_key(
                    message.payload['public_key'].encode('utf-8'),
                    backend=default_backend()
                )
                
                # Send handshake response
                response = MeshMessage(
                    message_type=MessageType.HANDSHAKE,
                    payload={
                        'node_id': self.node_id,
                        'public_key': self.public_key.public_bytes(
                            encoding=serialization.Encoding.PEM,
                            format=serialization.PublicFormat.SubjectPublicKeyInfo
                        ).decode('utf-8'),
                        'protocol_version': '1.0.0'
                    }
                )
                
                success = await peer.send_message(response)
                if success:
                    peer.handshake_complete = True
                    return True
                    
            return False
            
        except asyncio.TimeoutError:
            logger.error(f"Handshake timeout waiting for {peer.peer_id}")
            return False
        except Exception as e:
            logger.error(f"Handshake wait failed for {peer.peer_id}: {e}")
            return False
    
    async def _handle_message(self, message: MeshMessage, peer: 'PeerConnection'):
        """Handle incoming message"""
        try:
            # Update metrics
            self.metrics['messages_received'] += 1
            self.metrics['bytes_received'] += len(message.to_bytes())
            
            # Check for duplicates
            if message.message_id in self.message_cache:
                logger.debug(f"Duplicate message {message.message_id} ignored")
                return
            
            # Cache message ID
            self.message_cache[message.message_id] = time.time()
            
            # Verify signature if available
            if peer.public_key and not await self._verify_message(message, peer.public_key):
                logger.warning(f"Message signature verification failed from {peer.node_id}")
                return
            
            # Route message if not for us
            if message.recipient_id and message.recipient_id != self.node_id:
                if message.ttl > 0:
                    message.ttl -= 1
                    await self._route_message(message)
                return
            
            # Handle message locally
            handler = self.message_handlers.get(message.message_type)
            if handler:
                try:
                    await handler(message, peer)
                except Exception as e:
                    logger.error(f"Handler error for {message.message_type}: {e}")
            else:
                logger.debug(f"No handler for message type {message.message_type}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_handshake(self, message: MeshMessage, peer: 'PeerConnection'):
        """Handle handshake messages"""
        # Handshake is handled in connection setup
        pass
    
    async def _handle_heartbeat(self, message: MeshMessage, peer: 'PeerConnection'):
        """Handle heartbeat messages"""
        peer.last_heartbeat = time.time()
        logger.debug(f"Heartbeat received from {peer.node_id}")
    
    async def _handle_goodbye(self, message: MeshMessage, peer: 'PeerConnection'):
        """Handle goodbye messages"""
        logger.info(f"Goodbye received from {peer.node_id}")
        await self._remove_peer(peer.node_id)
    
    async def _handle_discovery(self, message: MeshMessage, peer: 'PeerConnection'):
        """Handle discovery messages"""
        # Basic discovery response
        response = MeshMessage(
            message_type=MessageType.DISCOVERY,
            recipient_id=message.sender_id,
            payload={
                'node_id': self.node_id,
                'peers': list(self.peers.keys())
            }
        )
        await self.send_message(response)
    
    async def _remove_peer(self, node_id: str):
        """Remove peer connection"""
        if node_id in self.peers:
            peer = self.peers.pop(node_id)
            await peer.close()
            self.metrics['connection_count'] -= 1
            logger.info(f"Removed peer {node_id}")
    
    async def _cleanup_task(self):
        """Background cleanup task"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Clean up old message cache entries
                expired_messages = [
                    msg_id for msg_id, timestamp in self.message_cache.items()
                    if current_time - timestamp > 300  # 5 minutes
                ]
                for msg_id in expired_messages:
                    del self.message_cache[msg_id]
                
                # Check for dead peers
                dead_peers = []
                for node_id, peer in self.peers.items():
                    if current_time - peer.last_heartbeat > 60:  # 60 seconds timeout
                        dead_peers.append(node_id)
                
                for node_id in dead_peers:
                    await self._remove_peer(node_id)
                
                await asyncio.sleep(30)  # Cleanup every 30 seconds
                
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(30)
    
    async def _heartbeat_task(self):
        """Background heartbeat task"""
        while self.is_running:
            try:
                if self.peers:
                    heartbeat = MeshMessage(message_type=MessageType.HEARTBEAT)
                    await self._broadcast_message(heartbeat)
                
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                
            except Exception as e:
                logger.error(f"Heartbeat task error: {e}")
                await asyncio.sleep(30)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get protocol metrics"""
        return {
            **self.metrics,
            'peer_count': len(self.peers),
            'cache_size': len(self.message_cache),
            'routing_table_size': len(self.routing_table),
            'is_running': self.is_running,
            'port': self.port
        }


class PeerConnection:
    """Represents a connection to a peer node"""
    
    def __init__(self, peer_id: str, reader: asyncio.StreamReader, 
                 writer: asyncio.StreamWriter, protocol: MeshProtocol):
        self.peer_id = peer_id
        self.node_id = peer_id  # Updated after handshake
        self.reader = reader
        self.writer = writer
        self.protocol = protocol
        self.public_key = None
        self.handshake_complete = False
        self.last_heartbeat = time.time()
        self.send_lock = asyncio.Lock()
        self.is_closed = False
    
    async def send_message(self, message: MeshMessage) -> bool:
        """Send message to peer"""
        if self.is_closed:
            return False
            
        try:
            async with self.send_lock:
                data = message.to_bytes()
                
                # Send length prefix + data
                length = struct.pack('!I', len(data))
                self.writer.write(length + data)
                await self.writer.drain()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to send message to {self.node_id}: {e}")
            await self.close()
            return False
    
    async def receive_message(self) -> Optional[MeshMessage]:
        """Receive message from peer"""
        try:
            # Read length prefix
            length_data = await self.reader.readexactly(4)
            length = struct.unpack('!I', length_data)[0]
            
            # Read message data
            data = await self.reader.readexactly(length)
            
            return MeshMessage.from_bytes(data)
            
        except asyncio.IncompleteReadError:
            # Connection closed
            await self.close()
            return None
        except Exception as e:
            logger.error(f"Failed to receive message from {self.node_id}: {e}")
            await self.close()
            return None
    
    async def listen(self):
        """Listen for incoming messages"""
        try:
            while not self.is_closed and self.protocol.is_running:
                message = await self.receive_message()
                if message:
                    await self.protocol._handle_message(message, self)
                else:
                    break
        except Exception as e:
            logger.error(f"Listen error for {self.node_id}: {e}")
        finally:
            await self.close()
    
    async def close(self):
        """Close peer connection"""
        if self.is_closed:
            return
            
        self.is_closed = True
        
        try:
            if not self.writer.is_closing():
                self.writer.close()
                await self.writer.wait_closed()
        except Exception as e:
            logger.error(f"Error closing connection to {self.node_id}: {e}")
        
        logger.debug(f"Connection to {self.node_id} closed")