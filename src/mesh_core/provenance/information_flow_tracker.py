"""
Information Flow Tracking System
===============================

Tracks the flow of information through the mesh network, creating
comprehensive audit trails of how data moves between nodes and
undergoes transformations.
"""

import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class FlowEventType(Enum):
    """Types of information flow events"""
    CREATED = "created"           # Information originally created
    RECEIVED = "received"         # Received from another node
    TRANSFORMED = "transformed"   # Content transformed/processed
    ENRICHED = "enriched"        # Additional information added
    FILTERED = "filtered"        # Content filtered or modified
    FORWARDED = "forwarded"      # Sent to another node
    CACHED = "cached"            # Stored in cache
    VALIDATED = "validated"      # Validation performed
    MERGED = "merged"            # Merged with other information
    DELETED = "deleted"          # Information deleted/removed

class TransformationType(Enum):
    """Types of information transformations"""
    TRANSLATION = "translation"     # Language translation
    SUMMARIZATION = "summary"       # Content summarization
    ENRICHMENT = "enrichment"      # Added metadata/context
    FILTERING = "filtering"        # Content filtering
    VALIDATION = "validation"      # Truth/fact validation
    NORMALIZATION = "normalization" # Data normalization
    AGGREGATION = "aggregation"    # Data aggregation
    ANALYSIS = "analysis"          # Content analysis

@dataclass
class FlowEvent:
    """Single information flow event"""
    event_id: str
    information_id: str
    event_type: FlowEventType
    node_id: str
    timestamp: float
    previous_node: Optional[str]
    next_node: Optional[str]
    transformation_type: Optional[TransformationType]
    content_hash: str              # Hash of content at this point
    metadata: Dict
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['event_type'] = self.event_type.value
        if self.transformation_type:
            data['transformation_type'] = self.transformation_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FlowEvent':
        data['event_type'] = FlowEventType(data['event_type'])
        if data.get('transformation_type'):
            data['transformation_type'] = TransformationType(data['transformation_type'])
        return cls(**data)

@dataclass
class InformationTrail:
    """Complete trail of information flow"""
    information_id: str
    origin_node: str
    created_at: float
    current_nodes: Set[str]       # Nodes currently holding this information
    flow_events: List[FlowEvent]
    transformations: List[Dict]   # Summary of transformations
    access_log: List[Dict]        # Access history
    
    def get_path_summary(self) -> List[str]:
        """Get simplified path summary"""
        path = [self.origin_node]
        for event in self.flow_events:
            if event.event_type == FlowEventType.FORWARDED and event.next_node:
                if event.next_node not in path:
                    path.append(event.next_node)
        return path

class InformationFlowTracker:
    """
    Information flow tracking system
    
    Maintains comprehensive audit trails of information flow through
    the mesh network, enabling provenance tracking and analysis.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.information_trails: Dict[str, InformationTrail] = {}
        self.node_flows: Dict[str, List[str]] = {}  # node_id -> information_ids
        self.flow_statistics: Dict[str, Dict] = {}
        
    def _generate_event_id(self, information_id: str, event_type: FlowEventType) -> str:
        """Generate unique event ID"""
        data = f"{information_id}:{event_type.value}:{self.node_id}:{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate hash of content"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def track_information_creation(
        self,
        information_id: str,
        content: str,
        creator_id: str,
        metadata: Optional[Dict] = None
    ) -> FlowEvent:
        """Track creation of new information"""
        
        if metadata is None:
            metadata = {}
        
        event = FlowEvent(
            event_id=self._generate_event_id(information_id, FlowEventType.CREATED),
            information_id=information_id,
            event_type=FlowEventType.CREATED,
            node_id=self.node_id,
            timestamp=time.time(),
            previous_node=None,
            next_node=None,
            transformation_type=None,
            content_hash=self._calculate_content_hash(content),
            metadata={**metadata, 'creator_id': creator_id, 'content_length': len(content)}
        )
        
        # Create new trail
        trail = InformationTrail(
            information_id=information_id,
            origin_node=self.node_id,
            created_at=time.time(),
            current_nodes={self.node_id},
            flow_events=[event],
            transformations=[],
            access_log=[]
        )
        
        self.information_trails[information_id] = trail
        
        # Update node flows
        if self.node_id not in self.node_flows:
            self.node_flows[self.node_id] = []
        self.node_flows[self.node_id].append(information_id)
        
        logger.info(f"Tracked creation of information {information_id}")
        return event
    
    async def track_information_receipt(
        self,
        information_id: str,
        content: str,
        sender_node: str,
        metadata: Optional[Dict] = None
    ) -> FlowEvent:
        """Track receipt of information from another node"""
        
        if metadata is None:
            metadata = {}
        
        event = FlowEvent(
            event_id=self._generate_event_id(information_id, FlowEventType.RECEIVED),
            information_id=information_id,
            event_type=FlowEventType.RECEIVED,
            node_id=self.node_id,
            timestamp=time.time(),
            previous_node=sender_node,
            next_node=None,
            transformation_type=None,
            content_hash=self._calculate_content_hash(content),
            metadata={**metadata, 'sender': sender_node}
        )
        
        # Get or create trail
        trail = self.information_trails.get(information_id)
        if trail:
            trail.flow_events.append(event)
            trail.current_nodes.add(self.node_id)
        else:
            # Create new trail for received information
            trail = InformationTrail(
                information_id=information_id,
                origin_node=sender_node,  # Best guess
                created_at=time.time(),
                current_nodes={self.node_id},
                flow_events=[event],
                transformations=[],
                access_log=[]
            )
            self.information_trails[information_id] = trail
        
        # Update node flows
        if self.node_id not in self.node_flows:
            self.node_flows[self.node_id] = []
        if information_id not in self.node_flows[self.node_id]:
            self.node_flows[self.node_id].append(information_id)
        
        logger.info(f"Tracked receipt of information {information_id} from {sender_node}")
        return event
    
    async def track_transformation(
        self,
        information_id: str,
        transformation_type: TransformationType,
        original_content: str,
        transformed_content: str,
        metadata: Optional[Dict] = None
    ) -> FlowEvent:
        """Track information transformation"""
        
        if metadata is None:
            metadata = {}
        
        event = FlowEvent(
            event_id=self._generate_event_id(information_id, FlowEventType.TRANSFORMED),
            information_id=information_id,
            event_type=FlowEventType.TRANSFORMED,
            node_id=self.node_id,
            timestamp=time.time(),
            previous_node=None,
            next_node=None,
            transformation_type=transformation_type,
            content_hash=self._calculate_content_hash(transformed_content),
            metadata={
                **metadata,
                'original_hash': self._calculate_content_hash(original_content),
                'transformation': transformation_type.value,
                'size_change': len(transformed_content) - len(original_content)
            }
        )
        
        # Update trail
        trail = self.information_trails.get(information_id)
        if trail:
            trail.flow_events.append(event)
            
            # Add transformation summary
            transformation_summary = {
                'type': transformation_type.value,
                'timestamp': time.time(),
                'node_id': self.node_id,
                'original_hash': self._calculate_content_hash(original_content),
                'result_hash': self._calculate_content_hash(transformed_content),
                'metadata': metadata
            }
            trail.transformations.append(transformation_summary)
        
        logger.info(f"Tracked transformation of {information_id}: {transformation_type.value}")
        return event
    
    async def track_forwarding(
        self,
        information_id: str,
        destination_node: str,
        metadata: Optional[Dict] = None
    ) -> FlowEvent:
        """Track forwarding of information to another node"""
        
        if metadata is None:
            metadata = {}
        
        event = FlowEvent(
            event_id=self._generate_event_id(information_id, FlowEventType.FORWARDED),
            information_id=information_id,
            event_type=FlowEventType.FORWARDED,
            node_id=self.node_id,
            timestamp=time.time(),
            previous_node=None,
            next_node=destination_node,
            transformation_type=None,
            content_hash="",  # Content not changed
            metadata={**metadata, 'destination': destination_node}
        )
        
        # Update trail
        trail = self.information_trails.get(information_id)
        if trail:
            trail.flow_events.append(event)
            trail.current_nodes.add(destination_node)
        
        logger.info(f"Tracked forwarding of {information_id} to {destination_node}")
        return event
    
    async def track_access(
        self,
        information_id: str,
        accessor_id: str,
        access_type: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Track access to information"""
        
        if metadata is None:
            metadata = {}
        
        access_record = {
            'timestamp': time.time(),
            'accessor_id': accessor_id,
            'access_type': access_type,
            'node_id': self.node_id,
            'metadata': metadata
        }
        
        trail = self.information_trails.get(information_id)
        if trail:
            trail.access_log.append(access_record)
            logger.debug(f"Tracked access to {information_id} by {accessor_id}")
            return True
        
        return False
    
    async def track_validation(
        self,
        information_id: str,
        validation_result: Dict,
        validator_id: str
    ) -> FlowEvent:
        """Track validation of information"""
        
        event = FlowEvent(
            event_id=self._generate_event_id(information_id, FlowEventType.VALIDATED),
            information_id=information_id,
            event_type=FlowEventType.VALIDATED,
            node_id=self.node_id,
            timestamp=time.time(),
            previous_node=None,
            next_node=None,
            transformation_type=TransformationType.VALIDATION,
            content_hash="",  # Content not changed by validation
            metadata={
                'validator_id': validator_id,
                'validation_result': validation_result,
                'confidence_score': validation_result.get('confidence', 0.0)
            }
        )
        
        # Update trail
        trail = self.information_trails.get(information_id)
        if trail:
            trail.flow_events.append(event)
        
        logger.info(f"Tracked validation of {information_id} by {validator_id}")
        return event
    
    def get_information_trail(self, information_id: str) -> Optional[InformationTrail]:
        """Get complete information trail"""
        return self.information_trails.get(information_id)
    
    def get_flow_path(self, information_id: str) -> List[str]:
        """Get the path information has taken through the network"""
        
        trail = self.information_trails.get(information_id)
        if not trail:
            return []
        
        return trail.get_path_summary()
    
    def get_transformations_summary(self, information_id: str) -> List[Dict]:
        """Get summary of all transformations applied to information"""
        
        trail = self.information_trails.get(information_id)
        if not trail:
            return []
        
        return trail.transformations
    
    def get_access_history(self, information_id: str) -> List[Dict]:
        """Get access history for information"""
        
        trail = self.information_trails.get(information_id)
        if not trail:
            return []
        
        return trail.access_log
    
    def find_information_by_content_hash(self, content_hash: str) -> List[str]:
        """Find information items by content hash"""
        
        matching_items = []
        
        for info_id, trail in self.information_trails.items():
            for event in trail.flow_events:
                if event.content_hash == content_hash:
                    matching_items.append(info_id)
                    break
        
        return matching_items
    
    def get_node_flow_statistics(self, node_id: Optional[str] = None) -> Dict:
        """Get flow statistics for node"""
        
        target_node = node_id or self.node_id
        
        # Count events by type for the node
        event_counts = {}
        total_transformations = 0
        unique_information = set()
        
        for trail in self.information_trails.values():
            for event in trail.flow_events:
                if event.node_id == target_node:
                    unique_information.add(event.information_id)
                    event_type = event.event_type.value
                    event_counts[event_type] = event_counts.get(event_type, 0) + 1
                    
                    if event.transformation_type:
                        total_transformations += 1
        
        return {
            'node_id': target_node,
            'unique_information_items': len(unique_information),
            'total_events': sum(event_counts.values()),
            'events_by_type': event_counts,
            'total_transformations': total_transformations,
            'information_items': list(self.node_flows.get(target_node, []))
        }
    
    def get_network_flow_analysis(self) -> Dict:
        """Get network-wide flow analysis"""
        
        all_nodes = set()
        all_connections = set()
        information_count = len(self.information_trails)
        
        for trail in self.information_trails.values():
            all_nodes.update(trail.current_nodes)
            all_nodes.add(trail.origin_node)
            
            # Track connections
            for event in trail.flow_events:
                if event.previous_node and event.node_id:
                    all_connections.add((event.previous_node, event.node_id))
                if event.node_id and event.next_node:
                    all_connections.add((event.node_id, event.next_node))
        
        return {
            'total_nodes': len(all_nodes),
            'total_connections': len(all_connections),
            'total_information_items': information_count,
            'network_density': len(all_connections) / max(1, len(all_nodes) * (len(all_nodes) - 1)),
            'nodes': list(all_nodes),
            'connections': list(all_connections)
        }
    
    def export_trail(self, information_id: str) -> Optional[Dict]:
        """Export complete information trail"""
        
        trail = self.information_trails.get(information_id)
        if not trail:
            return None
        
        return {
            'information_id': information_id,
            'origin_node': trail.origin_node,
            'created_at': trail.created_at,
            'current_nodes': list(trail.current_nodes),
            'flow_events': [event.to_dict() for event in trail.flow_events],
            'transformations': trail.transformations,
            'access_log': trail.access_log,
            'exported_at': time.time(),
            'exported_by': self.node_id
        }
    
    async def import_trail(self, trail_data: Dict) -> bool:
        """Import information trail"""
        
        try:
            information_id = trail_data['information_id']
            
            # Don't overwrite existing trails
            if information_id in self.information_trails:
                logger.warning(f"Trail for {information_id} already exists")
                return False
            
            # Reconstruct trail
            flow_events = [FlowEvent.from_dict(event_data) for event_data in trail_data['flow_events']]
            
            trail = InformationTrail(
                information_id=information_id,
                origin_node=trail_data['origin_node'],
                created_at=trail_data['created_at'],
                current_nodes=set(trail_data['current_nodes']),
                flow_events=flow_events,
                transformations=trail_data['transformations'],
                access_log=trail_data['access_log']
            )
            
            self.information_trails[information_id] = trail
            
            # Update node flows if this node is involved
            if self.node_id in trail.current_nodes:
                if self.node_id not in self.node_flows:
                    self.node_flows[self.node_id] = []
                if information_id not in self.node_flows[self.node_id]:
                    self.node_flows[self.node_id].append(information_id)
            
            logger.info(f"Imported trail for information {information_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import trail: {e}")
            return False