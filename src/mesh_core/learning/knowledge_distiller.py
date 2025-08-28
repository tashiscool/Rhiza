"""
Mesh Knowledge Distiller
========================

Component 8.1: Knowledge Distillation Protocols
Share learning efficiently between nodes

Implements knowledge distillation, compression, and sharing
protocols to enable efficient learning transfer across the mesh.
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import gzip
import base64

logger = logging.getLogger(__name__)


class DistillationType(Enum):
    """Types of knowledge distillation"""
    MODEL_DISTILLATION = "model_distillation"      # Distill model weights
    PATTERN_DISTILLATION = "pattern_distillation"  # Distill learned patterns
    EXPERIENCE_DISTILLATION = "experience_distillation"  # Distill experiences
    RULE_DISTILLATION = "rule_distillation"        # Distill learned rules
    METADATA_DISTILLATION = "metadata_distillation"  # Distill metadata


class DistillationStatus(Enum):
    """Status of distillation processes"""
    PENDING = "pending"                  # Waiting to be processed
    PROCESSING = "processing"            # Currently being distilled
    COMPLETED = "completed"              # Distillation finished
    FAILED = "failed"                    # Distillation failed
    SHARED = "shared"                    # Knowledge shared with network


@dataclass
class KnowledgePackage:
    """A package of distilled knowledge for sharing"""
    package_id: str
    source_node_id: str
    distillation_type: DistillationType
    created_at: datetime
    
    # Knowledge content
    knowledge_data: Dict[str, Any]
    compression_ratio: float = 0.0  # 0.0 to 1.0
    quality_score: float = 0.0      # 0.0 to 1.0
    
    # Sharing metadata
    target_nodes: List[str] = field(default_factory=list)
    sharing_priority: int = 5        # 1-10, higher = more important
    expiration_date: Optional[datetime] = None
    
    # Network metadata
    hops_traveled: int = 0
    nodes_reached: List[str] = field(default_factory=list)
    last_shared: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.package_id:
            self.package_id = self._generate_package_id()
    
    def _generate_package_id(self) -> str:
        """Generate unique package ID"""
        content = f"{self.source_node_id}{self.distillation_type.value}{self.created_at.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert knowledge package to dictionary"""
        return {
            "package_id": self.package_id,
            "source_node_id": self.source_node_id,
            "distillation_type": self.distillation_type.value,
            "created_at": self.created_at.isoformat(),
            "knowledge_data": self.knowledge_data,
            "compression_ratio": self.compression_ratio,
            "quality_score": self.quality_score,
            "target_nodes": self.target_nodes,
            "sharing_priority": self.sharing_priority,
            "expiration_date": self.expiration_date.isoformat() if self.expiration_date else None,
            "hops_traveled": self.hops_traveled,
            "nodes_reached": self.nodes_reached,
            "last_shared": self.last_shared.isoformat() if self.last_shared else None
        }


@dataclass
class DistillationRequest:
    """A request to distill knowledge"""
    request_id: str
    requester_node_id: str
    distillation_type: DistillationType
    created_at: datetime
    
    # Request details
    source_knowledge: Dict[str, Any]
    target_compression: float = 0.5  # Target compression ratio
    quality_requirements: Dict[str, float] = field(default_factory=dict)
    
    # Processing metadata
    status: DistillationStatus = DistillationStatus.PENDING
    assigned_processor: Optional[str] = None
    processing_started: Optional[datetime] = None
    processing_completed: Optional[datetime] = None
    
    # Results
    result_package_id: Optional[str] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if not self.request_id:
            self.request_id = self._generate_request_id()
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        content = f"{self.requester_node_id}{self.distillation_type.value}{self.created_at.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert distillation request to dictionary"""
        return {
            "request_id": self.request_id,
            "requester_node_id": self.requester_node_id,
            "distillation_type": self.distillation_type.value,
            "created_at": self.created_at.isoformat(),
            "source_knowledge": self.source_knowledge,
            "target_compression": self.target_compression,
            "quality_requirements": self.quality_requirements,
            "status": self.status.value,
            "assigned_processor": self.assigned_processor,
            "processing_started": self.processing_started.isoformat() if self.processing_started else None,
            "processing_completed": self.processing_completed.isoformat() if self.processing_completed else None,
            "result_package_id": self.result_package_id,
            "error_message": self.error_message
        }


class KnowledgeDistiller:
    """
    Distills and shares knowledge across the mesh network
    
    Compresses learning outcomes, patterns, and experiences
    into shareable packages for efficient network learning.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        
        # Knowledge storage
        self.knowledge_packages: Dict[str, KnowledgePackage] = {}
        self.distillation_requests: Dict[str, DistillationRequest] = {}
        self.received_knowledge: Dict[str, KnowledgePackage] = {}
        
        # Processing state
        self.processing_queue: List[str] = []
        self.active_processors: Dict[str, str] = {}  # processor_id -> request_id
        
        # Configuration
        self.max_package_size = 1024 * 1024  # 1MB
        self.min_quality_threshold = 0.6
        self.max_compression_ratio = 0.9
        self.sharing_radius = 3  # Maximum hops for knowledge sharing
        
        # Performance metrics
        self.packages_created = 0
        self.packages_shared = 0
        self.packages_received = 0
        self.distillation_requests_processed = 0
        
        # Threading
        self.lock = threading.RLock()
        
        logger.info(f"KnowledgeDistiller initialized for node: {self.node_id}")
    
    def create_knowledge_package(self, distillation_type: DistillationType,
                                knowledge_data: Dict[str, Any], **kwargs) -> str:
        """Create a new knowledge package"""
        try:
            with self.lock:
                # Compress knowledge data
                compressed_data, compression_ratio = self._compress_knowledge(knowledge_data)
                
                # Calculate quality score
                quality_score = self._calculate_quality_score(knowledge_data, compressed_data)
                
                # Create package
                package = KnowledgePackage(
                    package_id="",
                    source_node_id=self.node_id,
                    distillation_type=distillation_type,
                    created_at=datetime.utcnow(),
                    knowledge_data=compressed_data,
                    compression_ratio=compression_ratio,
                    quality_score=quality_score,
                    **kwargs
                )
                
                # Store package
                self.knowledge_packages[package.package_id] = package
                self.packages_created += 1
                
                logger.info(f"Created knowledge package: {package.package_id}")
                return package.package_id
                
        except Exception as e:
            logger.error(f"Failed to create knowledge package: {e}")
            raise
    
    def _compress_knowledge(self, knowledge_data: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Compress knowledge data for efficient sharing"""
        try:
            # Convert to JSON string
            json_str = json.dumps(knowledge_data, separators=(',', ':'))
            original_size = len(json_str.encode('utf-8'))
            
            # Compress using gzip
            compressed_bytes = gzip.compress(json_str.encode('utf-8'))
            compressed_size = len(compressed_bytes)
            
            # Encode as base64 for safe transmission
            encoded_data = base64.b64encode(compressed_bytes).decode('utf-8')
            
            compressed_data = {
                "compressed": True,
                "encoding": "gzip+base64",
                "original_size": original_size,
                "compressed_size": compressed_size,
                "data": encoded_data
            }
            
            compression_ratio = 1.0 - (compressed_size / original_size)
            
            return compressed_data, compression_ratio
            
        except Exception as e:
            logger.error(f"Failed to compress knowledge: {e}")
            # Return uncompressed data if compression fails
            return knowledge_data, 0.0
    
    def _calculate_quality_score(self, original_data: Dict[str, Any], 
                                compressed_data: Dict[str, Any]) -> float:
        """Calculate quality score for knowledge package"""
        try:
            # Base quality score
            quality_score = 0.7
            
            # Boost for good compression
            if compressed_data.get("compressed", False):
                compression_ratio = compressed_data.get("compression_ratio", 0.0)
                if compression_ratio > 0.5:
                    quality_score += 0.2
                elif compression_ratio > 0.3:
                    quality_score += 0.1
            
            # Boost for structured data
            if isinstance(original_data, dict) and len(original_data) > 0:
                structure_score = min(1.0, len(original_data) / 10.0)
                quality_score += structure_score * 0.1
            
            # Boost for metadata
            if "metadata" in original_data or "timestamp" in original_data:
                quality_score += 0.1
            
            return min(1.0, quality_score)
            
        except Exception as e:
            logger.error(f"Failed to calculate quality score: {e}")
            return 0.5
    
    def request_distillation(self, distillation_type: DistillationType,
                            source_knowledge: Dict[str, Any], **kwargs) -> str:
        """Request knowledge distillation from another node"""
        try:
            with self.lock:
                # Create distillation request
                request = DistillationRequest(
                    request_id="",
                    requester_node_id=self.node_id,
                    distillation_type=distillation_type,
                    created_at=datetime.utcnow(),
                    source_knowledge=source_knowledge,
                    **kwargs
                )
                
                # Store request
                self.distillation_requests[request.request_id] = request
                
                # Add to processing queue
                self.processing_queue.append(request.request_id)
                
                logger.info(f"Created distillation request: {request.request_id}")
                return request.request_id
                
        except Exception as e:
            logger.error(f"Failed to create distillation request: {e}")
            raise
    
    def process_distillation_request(self, request_id: str) -> Optional[str]:
        """Process a distillation request"""
        try:
            with self.lock:
                if request_id not in self.distillation_requests:
                    logger.warning(f"Distillation request not found: {request_id}")
                    return None
                
                request = self.distillation_requests[request_id]
                
                # Check if already being processed
                if request.status != DistillationStatus.PENDING:
                    logger.warning(f"Request {request_id} is not pending")
                    return None
                
                # Mark as processing
                request.status = DistillationStatus.PROCESSING
                request.assigned_processor = self.node_id
                request.processing_started = datetime.utcnow()
                
                # Process the request
                try:
                    # Create knowledge package from source knowledge
                    package_id = self.create_knowledge_package(
                        distillation_type=request.distillation_type,
                        knowledge_data=request.source_knowledge,
                        target_compression=request.target_compression,
                        quality_requirements=request.quality_requirements
                    )
                    
                    # Update request
                    request.status = DistillationStatus.COMPLETED
                    request.processing_completed = datetime.utcnow()
                    request.result_package_id = package_id
                    
                    self.distillation_requests_processed += 1
                    
                    logger.info(f"Processed distillation request: {request_id}")
                    return package_id
                    
                except Exception as e:
                    # Mark as failed
                    request.status = DistillationStatus.FAILED
                    request.processing_completed = datetime.utcnow()
                    request.error_message = str(e)
                    
                    logger.error(f"Failed to process distillation request {request_id}: {e}")
                    return None
                
        except Exception as e:
            logger.error(f"Error processing distillation request: {e}")
            return None
    
    def share_knowledge_package(self, package_id: str, target_nodes: List[str]) -> bool:
        """Share a knowledge package with target nodes"""
        try:
            with self.lock:
                if package_id not in self.knowledge_packages:
                    logger.warning(f"Knowledge package not found: {package_id}")
                    return False
                
                package = self.knowledge_packages[package_id]
                
                # Update package metadata
                package.target_nodes = target_nodes
                package.last_shared = datetime.utcnow()
                package.hops_traveled += 1
                
                # Simulate sharing (in real implementation, this would send to network)
                logger.info(f"Sharing knowledge package {package_id} with {len(target_nodes)} nodes")
                
                # Update statistics
                self.packages_shared += 1
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to share knowledge package: {e}")
            return False
    
    def receive_knowledge_package(self, package: KnowledgePackage) -> bool:
        """Receive a knowledge package from another node"""
        try:
            with self.lock:
                # Check if package is expired
                if package.expiration_date and datetime.utcnow() > package.expiration_date:
                    logger.warning(f"Received expired knowledge package: {package.package_id}")
                    return False
                
                # Check quality threshold
                if package.quality_score < self.min_quality_threshold:
                    logger.warning(f"Received low-quality knowledge package: {package.package_id}")
                    return False
                
                # Store received package
                self.received_knowledge[package.package_id] = package
                self.packages_received += 1
                
                # Update package metadata
                package.nodes_reached.append(self.node_id)
                
                logger.info(f"Received knowledge package: {package.package_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to receive knowledge package: {e}")
            return False
    
    def decompress_knowledge(self, compressed_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Decompress knowledge data"""
        try:
            if not compressed_data.get("compressed", False):
                return compressed_data
            
            # Decode base64
            compressed_bytes = base64.b64decode(compressed_data["data"])
            
            # Decompress gzip
            decompressed_bytes = gzip.decompress(compressed_bytes)
            
            # Parse JSON
            knowledge_data = json.loads(decompressed_bytes.decode('utf-8'))
            
            return knowledge_data
            
        except Exception as e:
            logger.error(f"Failed to decompress knowledge: {e}")
            return None
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get comprehensive knowledge distillation summary"""
        with self.lock:
            # Count packages by type
            package_counts = {}
            for package in self.knowledge_packages.values():
                p_type = package.distillation_type.value
                package_counts[p_type] = package_counts.get(p_type, 0) + 1
            
            # Count requests by status
            request_status_counts = {}
            for request in self.distillation_requests.values():
                status = request.status.value
                request_status_counts[status] = request_status_counts.get(status, 0) + 1
            
            return {
                "node_id": self.node_id,
                "packages_created": self.packages_created,
                "packages_shared": self.packages_shared,
                "packages_received": self.packages_received,
                "package_counts": package_counts,
                "total_requests": len(self.distillation_requests),
                "requests_processed": self.distillation_requests_processed,
                "request_status_counts": request_status_counts,
                "max_package_size": self.max_package_size,
                "min_quality_threshold": self.min_quality_threshold,
                "max_compression_ratio": self.max_compression_ratio,
                "sharing_radius": self.sharing_radius
            }
    
    def cleanup_expired_packages(self) -> int:
        """Remove expired knowledge packages"""
        try:
            with self.lock:
                expired_packages = []
                current_time = datetime.utcnow()
                
                # Check all packages
                for package_id, package in self.knowledge_packages.items():
                    if package.expiration_date and current_time > package.expiration_date:
                        expired_packages.append(package_id)
                
                # Remove expired packages
                for package_id in expired_packages:
                    del self.knowledge_packages[package_id]
                
                logger.info(f"Cleaned up {len(expired_packages)} expired packages")
                return len(expired_packages)
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired packages: {e}")
            return 0

