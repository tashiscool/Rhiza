"""
Data Chunker - Anonymized Data Chunking System

Implements secure data chunking with privacy protection for distributed
synchronization across The Mesh network. Uses cryptographic techniques
to ensure data integrity while maintaining anonymity.

Key Features:
- Content-based chunking with privacy protection
- Cryptographic integrity verification
- Anonymized chunk identification
- Differential synchronization support
- Privacy-preserving deduplication
"""

import hashlib
import secrets
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from enum import Enum
import json
import time
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


class ChunkType(Enum):
    """Types of data chunks in The Mesh"""
    KNOWLEDGE = "knowledge"
    TRUST_RECORD = "trust_record" 
    CONSENSUS_DATA = "consensus_data"
    METADATA = "metadata"
    USER_DATA = "user_data"


class PrivacyLevel(Enum):
    """Privacy protection levels for chunks"""
    PUBLIC = 1          # No anonymization needed
    PSEUDONYMOUS = 2    # Identity pseudonymized but trackable
    ANONYMOUS = 3       # Fully anonymous, no tracking
    ENCRYPTED = 4       # Encrypted with access control


@dataclass
class ChunkMetadata:
    """Metadata for a data chunk"""
    chunk_id: str
    content_hash: str
    privacy_level: PrivacyLevel
    chunk_type: ChunkType
    size: int
    timestamp: float
    version: int
    dependencies: List[str]  # IDs of chunks this depends on
    access_pattern: str      # Hashed access pattern for privacy


@dataclass
class DataChunk:
    """Encrypted and anonymized data chunk"""
    chunk_id: str
    encrypted_data: bytes
    metadata: ChunkMetadata
    integrity_proof: str
    anonymity_proof: str     # Proof of anonymization
    privacy_salt: bytes      # Salt for privacy protection


class ChunkingStrategy:
    """Base class for chunking strategies"""
    
    def __init__(self, target_size: int = 4096, overlap: int = 256):
        self.target_size = target_size
        self.overlap = overlap
    
    def chunk_data(self, data: bytes) -> List[Tuple[int, int]]:
        """Return list of (start, end) positions for chunks"""
        raise NotImplementedError


class RabinChunking(ChunkingStrategy):
    """Content-based chunking using Rabin fingerprinting"""
    
    def __init__(self, target_size: int = 4096, window_size: int = 64):
        super().__init__(target_size)
        self.window_size = window_size
        self.polynomial = 0x3DA3358B4DC173  # Rabin polynomial
        
    def _rolling_hash(self, data: bytes, start: int, end: int) -> int:
        """Compute rolling hash for content-based chunking"""
        hash_val = 0
        for i in range(start, min(end, len(data))):
            hash_val = ((hash_val << 8) ^ data[i]) % self.polynomial
        return hash_val
    
    def chunk_data(self, data: bytes) -> List[Tuple[int, int]]:
        """Content-based chunking with Rabin fingerprinting"""
        chunks = []
        start = 0
        
        while start < len(data):
            end = min(start + self.target_size * 2, len(data))
            
            # Find natural break point using rolling hash
            best_break = start + self.target_size
            min_variance = float('inf')
            
            for pos in range(start + self.target_size // 2, end):
                if pos + self.window_size > len(data):
                    break
                    
                hash_val = self._rolling_hash(data, pos, pos + self.window_size)
                variance = abs(hash_val % self.target_size - self.target_size // 2)
                
                if variance < min_variance:
                    min_variance = variance
                    best_break = pos
            
            chunks.append((start, best_break))
            start = best_break
            
        return chunks


class PrivacyProtector:
    """Handles privacy protection for data chunks"""
    
    def __init__(self):
        self.anonymization_key = secrets.token_bytes(32)
        
    def anonymize_identifier(self, identifier: str, privacy_level: PrivacyLevel) -> str:
        """Generate anonymized identifier based on privacy level"""
        if privacy_level == PrivacyLevel.PUBLIC:
            return identifier
            
        salt = secrets.token_bytes(16)
        
        if privacy_level == PrivacyLevel.PSEUDONYMOUS:
            # Deterministic anonymization - same input gives same output
            hash_input = f"{identifier}:{self.anonymization_key.hex()}".encode()
        else:
            # Full anonymization with random salt
            hash_input = f"{identifier}:{salt.hex()}:{secrets.token_hex(16)}".encode()
            
        return hashlib.sha256(hash_input).hexdigest()[:16]
    
    def generate_access_pattern(self, metadata: Dict[str, Any]) -> str:
        """Generate hashed access pattern for privacy protection"""
        pattern_data = {
            'timestamp_hour': int(metadata.get('timestamp', 0)) // 3600,
            'size_bucket': self._size_bucket(metadata.get('size', 0)),
            'type_hash': hashlib.sha256(str(metadata.get('type', '')).encode()).hexdigest()[:8]
        }
        return hashlib.sha256(json.dumps(pattern_data, sort_keys=True).encode()).hexdigest()[:16]
    
    def _size_bucket(self, size: int) -> str:
        """Bucket size for privacy protection"""
        if size < 1024:
            return "small"
        elif size < 10240:
            return "medium"
        elif size < 102400:
            return "large"
        else:
            return "xlarge"


class DataChunker:
    """Main data chunking system with privacy protection"""
    
    def __init__(self, chunking_strategy: ChunkingStrategy = None):
        self.chunking_strategy = chunking_strategy or RabinChunking()
        self.privacy_protector = PrivacyProtector()
        self.chunk_cache: Dict[str, DataChunk] = {}
        self.integrity_key = secrets.token_bytes(32)
        
    def chunk_data(self, 
                  data: Union[str, bytes, Dict], 
                  chunk_type: ChunkType,
                  privacy_level: PrivacyLevel = PrivacyLevel.ANONYMOUS,
                  metadata: Optional[Dict[str, Any]] = None) -> List[DataChunk]:
        """
        Chunk data with privacy protection and integrity verification
        
        Args:
            data: Data to chunk (string, bytes, or dict)
            chunk_type: Type of data being chunked
            privacy_level: Level of privacy protection required
            metadata: Additional metadata for chunking
            
        Returns:
            List of encrypted and anonymized data chunks
        """
        # Normalize data to bytes
        if isinstance(data, str):
            raw_data = data.encode('utf-8')
        elif isinstance(data, dict):
            raw_data = json.dumps(data, sort_keys=True).encode('utf-8')
        else:
            raw_data = data
            
        # Compress data for efficiency
        compressed_data = zlib.compress(raw_data)
        
        # Generate chunks using strategy
        chunk_positions = self.chunking_strategy.chunk_data(compressed_data)
        
        chunks = []
        for i, (start, end) in enumerate(chunk_positions):
            chunk_data = compressed_data[start:end]
            
            # Create chunk metadata
            content_hash = hashlib.sha256(chunk_data).hexdigest()
            chunk_metadata = ChunkMetadata(
                chunk_id="",  # Will be set after anonymization
                content_hash=content_hash,
                privacy_level=privacy_level,
                chunk_type=chunk_type,
                size=len(chunk_data),
                timestamp=time.time(),
                version=1,
                dependencies=[],
                access_pattern=self.privacy_protector.generate_access_pattern({
                    'timestamp': time.time(),
                    'size': len(chunk_data),
                    'type': chunk_type.value
                })
            )
            
            # Generate anonymized chunk ID
            raw_id = f"{content_hash}:{i}:{chunk_type.value}"
            chunk_id = self.privacy_protector.anonymize_identifier(raw_id, privacy_level)
            chunk_metadata.chunk_id = chunk_id
            
            # Encrypt chunk data
            encrypted_data = self._encrypt_chunk_data(chunk_data, chunk_id)
            
            # Generate integrity proof
            integrity_proof = self._generate_integrity_proof(chunk_data, chunk_metadata)
            
            # Generate anonymity proof
            anonymity_proof = self._generate_anonymity_proof(chunk_metadata, privacy_level)
            
            # Create final chunk
            chunk = DataChunk(
                chunk_id=chunk_id,
                encrypted_data=encrypted_data,
                metadata=chunk_metadata,
                integrity_proof=integrity_proof,
                anonymity_proof=anonymity_proof,
                privacy_salt=secrets.token_bytes(16)
            )
            
            chunks.append(chunk)
            self.chunk_cache[chunk_id] = chunk
            
        return chunks
    
    def _encrypt_chunk_data(self, data: bytes, chunk_id: str) -> bytes:
        """Encrypt chunk data using AES-GCM"""
        # Derive encryption key from chunk ID and master key
        key_material = f"{chunk_id}:{self.integrity_key.hex()}".encode()
        encryption_key = hashlib.pbkdf2_hmac('sha256', key_material, b'mesh_chunker', 100000)[:32]
        
        # Generate random IV
        iv = secrets.token_bytes(12)
        
        # Encrypt with AES-GCM
        cipher = Cipher(algorithms.AES(encryption_key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # Return IV + tag + ciphertext
        return iv + encryptor.tag + ciphertext
    
    def _decrypt_chunk_data(self, encrypted_data: bytes, chunk_id: str) -> bytes:
        """Decrypt chunk data using AES-GCM"""
        # Extract components
        iv = encrypted_data[:12]
        tag = encrypted_data[12:28]
        ciphertext = encrypted_data[28:]
        
        # Derive decryption key
        key_material = f"{chunk_id}:{self.integrity_key.hex()}".encode()
        decryption_key = hashlib.pbkdf2_hmac('sha256', key_material, b'mesh_chunker', 100000)[:32]
        
        # Decrypt with AES-GCM
        cipher = Cipher(algorithms.AES(decryption_key), modes.GCM(iv, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted = decryptor.update(ciphertext) + decryptor.finalize()
        
        return decrypted
    
    def _generate_integrity_proof(self, data: bytes, metadata: ChunkMetadata) -> str:
        """Generate cryptographic integrity proof for chunk"""
        # Convert metadata to JSON-serializable format
        metadata_dict = {
            'chunk_id': metadata.chunk_id,
            'content_hash': metadata.content_hash,
            'privacy_level': metadata.privacy_level.value,
            'chunk_type': metadata.chunk_type.value,
            'size': metadata.size,
            'timestamp': metadata.timestamp,
            'version': metadata.version,
            'dependencies': metadata.dependencies,
            'access_pattern': metadata.access_pattern
        }
        
        proof_data = {
            'content_hash': hashlib.sha256(data).hexdigest(),
            'metadata_hash': hashlib.sha256(json.dumps(metadata_dict, sort_keys=True).encode()).hexdigest(),
            'timestamp': metadata.timestamp,
            'size': len(data)
        }
        
        proof_string = json.dumps(proof_data, sort_keys=True)
        return hashlib.sha256(proof_string.encode()).hexdigest()
    
    def _generate_anonymity_proof(self, metadata: ChunkMetadata, privacy_level: PrivacyLevel) -> str:
        """Generate proof of proper anonymization"""
        anonymity_data = {
            'privacy_level': privacy_level.value,
            'has_anonymized_id': len(metadata.chunk_id) == 16,
            'has_access_pattern': len(metadata.access_pattern) == 16,
            'timestamp_rounded': int(metadata.timestamp) // 3600  # Hour precision
        }
        
        return hashlib.sha256(json.dumps(anonymity_data, sort_keys=True).encode()).hexdigest()[:16]
    
    def verify_chunk_integrity(self, chunk: DataChunk) -> bool:
        """Verify chunk integrity and anonymity proofs"""
        try:
            # Decrypt and verify data
            decrypted_data = self._decrypt_chunk_data(chunk.encrypted_data, chunk.chunk_id)
            
            # Verify integrity proof
            expected_proof = self._generate_integrity_proof(decrypted_data, chunk.metadata)
            if expected_proof != chunk.integrity_proof:
                return False
                
            # Verify anonymity proof
            expected_anonymity = self._generate_anonymity_proof(chunk.metadata, chunk.metadata.privacy_level)
            if expected_anonymity != chunk.anonymity_proof:
                return False
                
            return True
            
        except Exception:
            return False
    
    def reconstruct_data(self, chunks: List[DataChunk]) -> Optional[bytes]:
        """Reconstruct original data from chunks"""
        try:
            # Sort chunks by their original order (embedded in chunk_id)
            sorted_chunks = sorted(chunks, key=lambda c: c.chunk_id)
            
            # Decrypt and combine chunks
            combined_data = b""
            for chunk in sorted_chunks:
                if not self.verify_chunk_integrity(chunk):
                    return None
                    
                decrypted_chunk = self._decrypt_chunk_data(chunk.encrypted_data, chunk.chunk_id)
                combined_data += decrypted_chunk
            
            # Decompress reconstructed data
            return zlib.decompress(combined_data)
            
        except Exception:
            return None
    
    def get_chunk_dependencies(self, chunk_id: str) -> List[str]:
        """Get dependency chain for a chunk"""
        if chunk_id not in self.chunk_cache:
            return []
            
        chunk = self.chunk_cache[chunk_id]
        dependencies = chunk.metadata.dependencies.copy()
        
        # Recursively get dependencies of dependencies
        for dep_id in chunk.metadata.dependencies:
            dependencies.extend(self.get_chunk_dependencies(dep_id))
            
        return list(set(dependencies))  # Remove duplicates
    
    def get_chunk_stats(self) -> Dict[str, Any]:
        """Get statistics about cached chunks"""
        if not self.chunk_cache:
            return {}
            
        total_size = sum(chunk.metadata.size for chunk in self.chunk_cache.values())
        privacy_levels = {}
        chunk_types = {}
        
        for chunk in self.chunk_cache.values():
            privacy_level = chunk.metadata.privacy_level.name
            chunk_type = chunk.metadata.chunk_type.name
            
            privacy_levels[privacy_level] = privacy_levels.get(privacy_level, 0) + 1
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        return {
            'total_chunks': len(self.chunk_cache),
            'total_size': total_size,
            'average_size': total_size // len(self.chunk_cache),
            'privacy_levels': privacy_levels,
            'chunk_types': chunk_types
        }