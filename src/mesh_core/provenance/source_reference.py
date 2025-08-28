"""
Source Reference System
======================

Manages anonymized but traceable source references for information
flowing through The Mesh network. Provides audit trails without
compromising privacy.
"""

import time
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SourceType(Enum):
    """Types of information sources"""
    USER_INPUT = "user"           # Direct user input
    PEER_NODE = "peer"           # Information from peer node
    EXTERNAL_API = "external"    # External API or service
    DOCUMENT = "document"        # Document or file
    SENSOR = "sensor"           # Sensor data
    COMPUTED = "computed"       # Computed/derived information
    CONSENSUS = "consensus"     # Network consensus result

class ConfidenceLevel(Enum):
    """Source confidence levels"""
    UNKNOWN = "unknown"         # Unknown reliability
    LOW = "low"                # Low confidence source
    MEDIUM = "medium"          # Medium confidence source  
    HIGH = "high"              # High confidence source
    VERIFIED = "verified"      # Cryptographically verified source

@dataclass
class SourceReference:
    """Anonymized but traceable source reference"""
    source_id: str              # Unique anonymous identifier
    source_type: SourceType
    confidence_level: ConfidenceLevel
    created_at: float
    last_verified: float
    verification_count: int
    attributes: Dict           # Non-identifying attributes
    trace_hash: str           # Cryptographic trace for auditing
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['source_type'] = self.source_type.value
        data['confidence_level'] = self.confidence_level.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SourceReference':
        data['source_type'] = SourceType(data['source_type'])
        data['confidence_level'] = ConfidenceLevel(data['confidence_level'])
        return cls(**data)

class SourceReferenceManager:
    """
    Manages source references with privacy protection
    
    Creates anonymized references that can be traced for auditing
    purposes while protecting source identity.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.source_references: Dict[str, SourceReference] = {}
        self.reference_mappings: Dict[str, str] = {}  # Maps real IDs to anonymous IDs
        self.verification_history: Dict[str, List[Dict]] = {}
        
    def _generate_source_id(self, real_source_id: str, source_type: SourceType) -> str:
        """Generate anonymous source ID"""
        # Create deterministic but non-reversible anonymous ID
        combined = f"{self.node_id}:{real_source_id}:{source_type.value}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _create_trace_hash(self, real_source_id: str, metadata: Dict) -> str:
        """Create cryptographic trace hash for auditing"""
        trace_data = {
            'node_id': self.node_id,
            'real_source_id': real_source_id,
            'metadata': metadata,
            'timestamp': time.time()
        }
        combined = json.dumps(trace_data, sort_keys=True)
        return hashlib.sha256(combined.encode()).hexdigest()
    
    async def create_source_reference(
        self,
        real_source_id: str,
        source_type: SourceType,
        confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM,
        attributes: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> SourceReference:
        """Create new anonymized source reference"""
        
        if attributes is None:
            attributes = {}
        if metadata is None:
            metadata = {}
            
        # Generate anonymous source ID
        source_id = self._generate_source_id(real_source_id, source_type)
        
        # Create trace hash for auditing
        trace_hash = self._create_trace_hash(real_source_id, metadata)
        
        # Create source reference
        reference = SourceReference(
            source_id=source_id,
            source_type=source_type,
            confidence_level=confidence_level,
            created_at=time.time(),
            last_verified=time.time(),
            verification_count=0,
            attributes=attributes,
            trace_hash=trace_hash
        )
        
        # Store reference and mapping
        self.source_references[source_id] = reference
        self.reference_mappings[real_source_id] = source_id
        self.verification_history[source_id] = []
        
        logger.info(f"Created source reference {source_id} for type {source_type.value}")
        return reference
    
    async def get_source_reference(self, source_id: str) -> Optional[SourceReference]:
        """Get source reference by anonymous ID"""
        return self.source_references.get(source_id)
    
    async def get_reference_by_real_id(self, real_source_id: str) -> Optional[SourceReference]:
        """Get source reference by real source ID"""
        anonymous_id = self.reference_mappings.get(real_source_id)
        if anonymous_id:
            return self.source_references.get(anonymous_id)
        return None
    
    async def update_confidence_level(
        self,
        source_id: str,
        new_confidence: ConfidenceLevel,
        reason: str,
        verifier_id: str
    ) -> bool:
        """Update confidence level of source reference"""
        
        reference = self.source_references.get(source_id)
        if not reference:
            logger.error(f"Source reference {source_id} not found")
            return False
        
        # Record verification event
        verification_event = {
            'timestamp': time.time(),
            'previous_confidence': reference.confidence_level.value,
            'new_confidence': new_confidence.value,
            'reason': reason,
            'verifier_id': verifier_id
        }
        
        self.verification_history[source_id].append(verification_event)
        
        # Update reference
        reference.confidence_level = new_confidence
        reference.last_verified = time.time()
        reference.verification_count += 1
        
        logger.info(f"Updated confidence for {source_id}: {new_confidence.value} ({reason})")
        return True
    
    async def verify_source_trace(self, source_id: str, expected_trace: str) -> bool:
        """Verify source trace hash for auditing"""
        
        reference = self.source_references.get(source_id)
        if not reference:
            return False
        
        return reference.trace_hash == expected_trace
    
    async def get_verification_history(self, source_id: str) -> List[Dict]:
        """Get verification history for source"""
        return self.verification_history.get(source_id, [])
    
    async def search_sources(
        self,
        source_type: Optional[SourceType] = None,
        confidence_level: Optional[ConfidenceLevel] = None,
        min_verifications: Optional[int] = None,
        time_range: Optional[Tuple[float, float]] = None
    ) -> List[SourceReference]:
        """Search source references by criteria"""
        
        results = []
        
        for reference in self.source_references.values():
            # Filter by source type
            if source_type and reference.source_type != source_type:
                continue
                
            # Filter by confidence level
            if confidence_level and reference.confidence_level != confidence_level:
                continue
            
            # Filter by verification count
            if min_verifications and reference.verification_count < min_verifications:
                continue
            
            # Filter by time range
            if time_range:
                start_time, end_time = time_range
                if not (start_time <= reference.created_at <= end_time):
                    continue
            
            results.append(reference)
        
        return results
    
    async def get_source_statistics(self) -> Dict:
        """Get statistics about source references"""
        
        type_counts = {}
        confidence_counts = {}
        total_verifications = 0
        
        for reference in self.source_references.values():
            # Count by type
            source_type = reference.source_type.value
            type_counts[source_type] = type_counts.get(source_type, 0) + 1
            
            # Count by confidence level
            confidence = reference.confidence_level.value
            confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
            
            # Total verifications
            total_verifications += reference.verification_count
        
        return {
            'total_sources': len(self.source_references),
            'by_type': type_counts,
            'by_confidence': confidence_counts,
            'total_verifications': total_verifications,
            'average_verifications': total_verifications / len(self.source_references) if self.source_references else 0,
            'node_id': self.node_id
        }
    
    async def export_references(self, include_trace_hashes: bool = False) -> Dict:
        """Export source references (for backup or transfer)"""
        
        references_data = []
        for reference in self.source_references.values():
            data = reference.to_dict()
            if not include_trace_hashes:
                data.pop('trace_hash', None)
            references_data.append(data)
        
        return {
            'node_id': self.node_id,
            'references': references_data,
            'exported_at': time.time(),
            'include_traces': include_trace_hashes
        }
    
    async def import_references(self, data: Dict) -> int:
        """Import source references from external data"""
        
        imported_count = 0
        
        try:
            for ref_data in data.get('references', []):
                reference = SourceReference.from_dict(ref_data)
                
                # Only import if not already present
                if reference.source_id not in self.source_references:
                    self.source_references[reference.source_id] = reference
                    self.verification_history[reference.source_id] = []
                    imported_count += 1
            
            logger.info(f"Imported {imported_count} source references")
            return imported_count
            
        except Exception as e:
            logger.error(f"Failed to import references: {e}")
            return 0