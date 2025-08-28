"""
Provenance Tracker - Core provenance management system

Tracks the complete lifecycle of information in the Mesh network,
including sources, transformations, confidence evolution, and contextual
metadata.
"""

import asyncio
import logging
try:
    import spacy
except ImportError:
    from .mock_spacy import *
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid

try:
    from mesh_core.config_manager import get_component_config
except ImportError:
    try:
        # Try relative import
        from ..config_manager import get_component_config
    except ImportError:
        # Fallback for development
        def get_component_config(component):
            return {}

logger = logging.getLogger(__name__)

class ProvenanceType(Enum):
    """Types of provenance information"""
    SOURCE_REFERENCE = "source_reference"
    CONFIDENCE_HISTORY = "confidence_history"
    CONTEXT_FRAMING = "context_framing"
    FLOW_TRACKING = "flow_tracking"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"

class ProvenanceStatus(Enum):
    """Status of provenance information"""
    ACTIVE = "active"
    VERIFIED = "verified"
    DISPUTED = "disputed"
    EXPIRED = "expired"
    INVALID = "invalid"

@dataclass
class ProvenanceRecord:
    """Complete provenance record for a piece of information"""
    record_id: str
    information_id: str
    provenance_type: ProvenanceType
    status: ProvenanceStatus
    timestamp: datetime
    source_node: str
    metadata: Dict[str, Any]
    confidence_score: float
    context_framing: Dict[str, Any]
    flow_path: List[str]
    transformations: List[Dict[str, Any]]
    validations: List[Dict[str, Any]]
    expires_at: Optional[datetime] = None
    parent_record: Optional[str] = None
    child_records: List[str] = field(default_factory=list)

@dataclass
class SourceReference:
    """Reference to an information source"""
    source_id: str
    source_type: str  # "node", "external", "consensus", "user_input"
    source_identifier: str
    source_metadata: Dict[str, Any]
    anonymity_level: str  # "full", "partial", "none"
    verification_status: str
    timestamp: datetime

@dataclass
class ConfidenceHistory:
    """History of confidence score changes"""
    history_id: str
    information_id: str
    confidence_score: float
    change_reason: str
    change_metadata: Dict[str, Any]
    timestamp: datetime
    node_id: str

@dataclass
class ContextFraming:
    """Contextual information about data"""
    context_id: str
    information_id: str
    cultural_context: Optional[str]
    regional_context: Optional[str]
    temporal_context: Optional[str]
    domain_context: Optional[str]
    use_context: Optional[str]
    metadata: Dict[str, Any]

class ProvenanceTracker:
    """Core system for tracking information provenance in the Mesh network"""
    
    def __init__(self):
        # Try to get config, fall back to defaults if not available
        try:
            self.config = get_component_config("mesh_provenance")
            self.retention_days = self.config.get("retention_days", 365)
            self.max_records_per_info = self.config.get("max_records_per_info", 1000)
            self.anonymity_enabled = self.config.get("anonymity_enabled", True)
            self.verification_required = self.config.get("verification_required", True)
        except Exception:
            # Use default values if config is not available
            self.config = {}
            self.retention_days = 365
            self.max_records_per_info = 1000
            self.anonymity_enabled = True
            self.verification_required = True
        
        # Storage
        self.provenance_records: Dict[str, ProvenanceRecord] = {}
        self.source_references: Dict[str, SourceReference] = {}
        self.confidence_histories: Dict[str, List[ConfidenceHistory]] = {}
        self.context_framings: Dict[str, ContextFraming] = {}
        
        # Indexes
        self.information_index: Dict[str, List[str]] = {}  # info_id -> record_ids
        self.node_index: Dict[str, List[str]] = {}         # node_id -> record_ids
        self.type_index: Dict[str, List[str]] = {}         # type -> record_ids
        
        # Coordination
        self.tracking_lock = asyncio.Lock()
        
        logger.info("Provenance tracker initialized")
    
    async def create_provenance_record(self, 
                                     information_id: str,
                                     provenance_type: ProvenanceType,
                                     source_node: str,
                                     metadata: Dict[str, Any],
                                     confidence_score: float,
                                     context_framing: Optional[Dict[str, Any]] = None,
                                     parent_record: Optional[str] = None) -> str:
        """Create a new provenance record"""
        try:
            async with self.tracking_lock:
                # Generate record ID
                record_id = f"prov_{information_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
                
                # Create source reference
                source_ref = await self._create_source_reference(source_node, metadata)
                
                # Create context framing
                context = await self._create_context_framing(information_id, context_framing or {})
                
                # Create provenance record
                record = ProvenanceRecord(
                    record_id=record_id,
                    information_id=information_id,
                    provenance_type=provenance_type,
                    status=ProvenanceStatus.ACTIVE,
                    timestamp=datetime.now(),
                    source_node=source_node,
                    metadata=metadata,
                    confidence_score=confidence_score,
                    context_framing=context.context_id if context else None,
                    flow_path=[source_node],
                    transformations=[],
                    validations=[],
                    parent_record=parent_record
                )
                
                # Store record
                self.provenance_records[record_id] = record
                
                # Update indexes
                await self._update_indexes(record)
                
                # Create confidence history entry
                await self._create_confidence_history(information_id, confidence_score, "initial", metadata, source_node)
                
                logger.info(f"Created provenance record {record_id} for information {information_id}")
                return record_id
                
        except Exception as e:
            logger.error(f"Error creating provenance record: {e}")
            raise
    
    async def update_provenance_record(self, 
                                     record_id: str,
                                     updates: Dict[str, Any]) -> bool:
        """Update an existing provenance record"""
        try:
            async with self.tracking_lock:
                if record_id not in self.provenance_records:
                    logger.warning(f"Provenance record {record_id} not found")
                    return False
                
                record = self.provenance_records[record_id]
                
                # Update fields
                for key, value in updates.items():
                    if hasattr(record, key):
                        setattr(record, key, value)
                
                # Update timestamp
                record.timestamp = datetime.now()
                
                # Update confidence history if confidence changed
                if "confidence_score" in updates:
                    await self._create_confidence_history(
                        record.information_id,
                        updates["confidence_score"],
                        "update",
                        {"reason": "provenance_update", "updates": updates},
                        "system"
                    )
                
                logger.info(f"Updated provenance record {record_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating provenance record: {e}")
            return False
    
    async def add_transformation(self, 
                               record_id: str,
                               transformation_type: str,
                               transformation_data: Dict[str, Any],
                               node_id: str) -> bool:
        """Add a transformation to a provenance record"""
        try:
            async with self.tracking_lock:
                if record_id not in self.provenance_records:
                    return False
                
                record = self.provenance_records[record_id]
                
                transformation = {
                    "id": f"trans_{uuid.uuid4().hex[:8]}",
                    "type": transformation_type,
                    "data": transformation_data,
                    "node_id": node_id,
                    "timestamp": datetime.now().isoformat()
                }
                
                record.transformations.append(transformation)
                
                # Update flow path
                if node_id not in record.flow_path:
                    record.flow_path.append(node_id)
                
                logger.info(f"Added transformation to record {record_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error adding transformation: {e}")
            return False
    
    async def add_validation(self, 
                           record_id: str,
                           validation_type: str,
                           validation_result: bool,
                           validation_data: Dict[str, Any],
                           node_id: str) -> bool:
        """Add a validation to a provenance record"""
        try:
            async with self.tracking_lock:
                if record_id not in self.provenance_records:
                    return False
                
                record = self.provenance_records[record_id]
                
                validation = {
                    "id": f"val_{uuid.uuid4().hex[:8]}",
                    "type": validation_type,
                    "result": validation_result,
                    "data": validation_data,
                    "node_id": node_id,
                    "timestamp": datetime.now().isoformat()
                }
                
                record.validations.append(validation)
                
                # Update status based on validation results
                if validation_result:
                    if record.status == ProvenanceStatus.ACTIVE:
                        record.status = ProvenanceStatus.VERIFIED
                else:
                    if record.status == ProvenanceStatus.VERIFIED:
                        record.status = ProvenanceStatus.DISPUTED
                
                logger.info(f"Added validation to record {record_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error adding validation: {e}")
            return False
    
    async def get_provenance_record(self, record_id: str) -> Optional[ProvenanceRecord]:
        """Get a provenance record by ID"""
        return self.provenance_records.get(record_id)
    
    async def get_information_provenance(self, information_id: str) -> List[ProvenanceRecord]:
        """Get all provenance records for a piece of information"""
        try:
            record_ids = self.information_index.get(information_id, [])
            records = []
            
            for record_id in record_ids:
                if record_id in self.provenance_records:
                    records.append(self.provenance_records[record_id])
            
            # Sort by timestamp (newest first)
            records.sort(key=lambda x: x.timestamp, reverse=True)
            return records
            
        except Exception as e:
            logger.error(f"Error getting information provenance: {e}")
            return []
    
    async def get_node_provenance(self, node_id: str) -> List[ProvenanceRecord]:
        """Get all provenance records created by a node"""
        try:
            record_ids = self.node_index.get(node_id, [])
            records = []
            
            for record_id in record_ids:
                if record_id in self.provenance_records:
                    records.append(self.provenance_records[record_id])
            
            # Sort by timestamp (newest first)
            records.sort(key=lambda x: x.timestamp, reverse=True)
            return records
            
        except Exception as e:
            logger.error(f"Error getting node provenance: {e}")
            return []
    
    async def get_provenance_summary(self, information_id: str) -> Dict[str, Any]:
        """Get a summary of provenance information"""
        try:
            records = await self.get_information_provenance(information_id)
            if not records:
                return {"status": "no_provenance", "information_id": information_id}
            
            # Get latest record
            latest_record = records[0]
            
            # Get confidence history
            confidence_history = self.confidence_histories.get(information_id, [])
            
            # Get context framing
            context = None
            if latest_record.context_framing:
                context = self.context_framings.get(latest_record.context_framing)
            
            # Calculate statistics
            total_transformations = sum(len(r.transformations) for r in records)
            total_validations = sum(len(r.validations) for r in records)
            unique_nodes = len(set(r.source_node for r in records))
            
            # Get flow path
            flow_path = latest_record.flow_path if latest_record.flow_path else []
            
            return {
                "status": "success",
                "information_id": information_id,
                "latest_record": {
                    "record_id": latest_record.record_id,
                    "timestamp": latest_record.timestamp.isoformat(),
                    "status": latest_record.status.value,
                    "confidence_score": latest_record.confidence_score,
                    "source_node": latest_record.source_node
                },
                "statistics": {
                    "total_records": len(records),
                    "total_transformations": total_transformations,
                    "total_validations": total_validations,
                    "unique_nodes": unique_nodes,
                    "flow_path_length": len(flow_path)
                },
                "confidence_history": [
                    {
                        "score": ch.confidence_score,
                        "reason": ch.change_reason,
                        "timestamp": ch.timestamp.isoformat(),
                        "node_id": ch.node_id
                    }
                    for ch in confidence_history[-10:]  # Last 10 changes
                ],
                "context_framing": {
                    "cultural_context": context.cultural_context if context else None,
                    "regional_context": context.regional_context if context else None,
                    "temporal_context": context.temporal_context if context else None,
                    "domain_context": context.domain_context if context else None
                } if context else None,
                "flow_path": flow_path
            }
            
        except Exception as e:
            logger.error(f"Error getting provenance summary: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _create_source_reference(self, source_node: str, metadata: Dict[str, Any]) -> SourceReference:
        """Create a source reference for a provenance record"""
        try:
            source_id = f"src_{uuid.uuid4().hex[:8]}"
            
            # Determine source type
            source_type = metadata.get("source_type", "node")
            if source_node.startswith("ext_"):
                source_type = "external"
            elif source_node.startswith("consensus_"):
                source_type = "consensus"
            elif source_node.startswith("user_"):
                source_type = "user_input"
            
            # Determine anonymity level
            anonymity_level = "partial" if self.anonymity_enabled else "none"
            
            # Create source reference
            source_ref = SourceReference(
                source_id=source_id,
                source_type=source_type,
                source_identifier=source_node,
                source_metadata=metadata.get("source_metadata", {}),
                anonymity_level=anonymity_level,
                verification_status="pending",
                timestamp=datetime.now()
            )
            
            # Store reference
            self.source_references[source_id] = source_ref
            
            return source_ref
            
        except Exception as e:
            logger.error(f"Error creating source reference: {e}")
            raise
    
    async def _create_context_framing(self, information_id: str, context_data: Dict[str, Any]) -> Optional[ContextFraming]:
        """Create context framing for a piece of information"""
        try:
            context_id = f"ctx_{information_id}_{uuid.uuid4().hex[:8]}"
            
            context = ContextFraming(
                context_id=context_id,
                information_id=information_id,
                cultural_context=context_data.get("cultural_context"),
                regional_context=context_data.get("regional_context"),
                temporal_context=context_data.get("temporal_context"),
                domain_context=context_data.get("domain_context"),
                use_context=context_data.get("use_context"),
                metadata=context_data.get("metadata", {})
            )
            
            # Store context
            self.context_framings[context_id] = context
            
            return context
            
        except Exception as e:
            logger.error(f"Error creating context framing: {e}")
            return None
    
    async def _create_confidence_history(self, 
                                       information_id: str,
                                       confidence_score: float,
                                       change_reason: str,
                                       change_metadata: Dict[str, Any],
                                       node_id: str):
        """Create a confidence history entry"""
        try:
            history_id = f"conf_{information_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            history_entry = ConfidenceHistory(
                history_id=history_id,
                information_id=information_id,
                confidence_score=confidence_score,
                change_reason=change_reason,
                change_metadata=change_metadata,
                timestamp=datetime.now(),
                node_id=node_id
            )
            
            # Store in confidence histories
            if information_id not in self.confidence_histories:
                self.confidence_histories[information_id] = []
            
            self.confidence_histories[information_id].append(history_entry)
            
            # Limit history size
            if len(self.confidence_histories[information_id]) > self.max_records_per_info:
                self.confidence_histories[information_id] = self.confidence_histories[information_id][-self.max_records_per_info:]
            
        except Exception as e:
            logger.error(f"Error creating confidence history: {e}")
    
    async def _update_indexes(self, record: ProvenanceRecord):
        """Update all indexes with the new record"""
        try:
            # Information index
            if record.information_id not in self.information_index:
                self.information_index[record.information_id] = []
            self.information_index[record.information_id].append(record.record_id)
            
            # Node index
            if record.source_node not in self.node_index:
                self.node_index[record.source_node] = []
            self.node_index[record.source_node].append(record.record_id)
            
            # Type index
            type_key = record.provenance_type.value
            if type_key not in self.type_index:
                self.type_index[type_key] = []
            self.type_index[type_key].append(record.record_id)
            
        except Exception as e:
            logger.error(f"Error updating indexes: {e}")
    
    async def cleanup_expired_records(self):
        """Clean up expired provenance records"""
        try:
            async with self.tracking_lock:
                current_time = datetime.now()
                expired_records = []
                
                for record_id, record in self.provenance_records.items():
                    # Check if record has expired
                    if record.expires_at and current_time > record.expires_at:
                        expired_records.append(record_id)
                    # Check retention policy
                    elif (current_time - record.timestamp).days > self.retention_days:
                        expired_records.append(record_id)
                
                # Remove expired records
                for record_id in expired_records:
                    await self._remove_record(record_id)
                
                if expired_records:
                    logger.info(f"Cleaned up {len(expired_records)} expired provenance records")
                
        except Exception as e:
            logger.error(f"Error cleaning up expired records: {e}")
    
    async def _remove_record(self, record_id: str):
        """Remove a provenance record and clean up indexes"""
        try:
            if record_id not in self.provenance_records:
                return
            
            record = self.provenance_records[record_id]
            
            # Remove from indexes
            if record.information_id in self.information_index:
                self.information_index[record.information_id] = [
                    rid for rid in self.information_index[record.information_id] 
                    if rid != record_id
                ]
            
            if record.source_node in self.node_index:
                self.node_index[record.source_node] = [
                    rid for rid in self.node_index[record.source_node] 
                    if rid != record_id
                ]
            
            type_key = record.provenance_type.value
            if type_key in self.type_index:
                self.type_index[type_key] = [
                    rid for rid in self.type_index[type_key] 
                    if rid != record_id
                ]
            
            # Remove record
            del self.provenance_records[record_id]
            
        except Exception as e:
            logger.error(f"Error removing record {record_id}: {e}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            total_records = len(self.provenance_records)
            total_sources = len(self.source_references)
            total_contexts = len(self.context_framings)
            total_confidence_entries = sum(len(hist) for hist in self.confidence_histories.values())
            
            # Count by type
            type_counts = {}
            for type_key, record_ids in self.type_index.items():
                type_counts[type_key] = len(record_ids)
            
            # Count by status
            status_counts = {}
            for record in self.provenance_records.values():
                status = record.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                "total_provenance_records": total_records,
                "total_source_references": total_sources,
                "total_context_framings": total_contexts,
                "total_confidence_entries": total_confidence_entries,
                "type_breakdown": type_counts,
                "status_breakdown": status_counts,
                "retention_days": self.retention_days,
                "max_records_per_info": self.max_records_per_info
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"error": str(e)}
