"""
Trust Ledger - Core Distributed Trust Tracking System

Implements a decentralized ledger for tracking trust relationships between
mesh nodes with:
- Cryptographic integrity of trust records
- Distributed consensus on trust scores
- Temporal trust evolution and decay
- Multi-dimensional trust metrics
- Byzantine fault tolerance
"""

import asyncio
import time
import hashlib
import json
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import math
from collections import defaultdict, deque
import uuid

logger = logging.getLogger(__name__)

class TrustDimension(Enum):
    """Dimensions of trust measurement"""
    RELIABILITY = "reliability"      # How reliable the node is
    ACCURACY = "accuracy"           # How accurate the node's information is
    HONESTY = "honesty"            # How honest the node appears to be
    COMPETENCE = "competence"      # How competent the node is
    TIMELINESS = "timeliness"      # How timely the node's responses are
    COOPERATION = "cooperation"    # How cooperative the node is

class TrustContext(Enum):
    """Context in which trust is evaluated"""
    GENERAL = "general"
    INFORMATION = "information"
    COMPUTATION = "computation"
    ROUTING = "routing"
    STORAGE = "storage"
    SOCIAL = "social"

@dataclass
class TrustScore:
    """Multi-dimensional trust score"""
    node_id: str
    observer_id: str
    context: TrustContext
    dimensions: Dict[TrustDimension, float] = field(default_factory=dict)
    overall_score: float = 0.0
    confidence: float = 0.0
    last_updated: float = field(default_factory=time.time)
    interaction_count: int = 0
    
    def __post_init__(self):
        """Initialize default dimension scores"""
        if not self.dimensions:
            for dimension in TrustDimension:
                self.dimensions[dimension] = 0.5  # Neutral starting point
        self.calculate_overall_score()
    
    def calculate_overall_score(self):
        """Calculate weighted overall trust score"""
        if not self.dimensions:
            self.overall_score = 0.5
            return
        
        # Weight different dimensions based on context
        weights = self._get_dimension_weights()
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for dimension, score in self.dimensions.items():
            weight = weights.get(dimension, 1.0)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight > 0:
            self.overall_score = weighted_sum / total_weight
        else:
            self.overall_score = 0.5
        
        # Calculate confidence based on interaction count
        self.confidence = min(1.0, self.interaction_count / 100.0)
    
    def _get_dimension_weights(self) -> Dict[TrustDimension, float]:
        """Get dimension weights based on context"""
        weights = {
            TrustDimension.RELIABILITY: 1.0,
            TrustDimension.ACCURACY: 1.0,
            TrustDimension.HONESTY: 1.0,
            TrustDimension.COMPETENCE: 1.0,
            TrustDimension.TIMELINESS: 1.0,
            TrustDimension.COOPERATION: 1.0
        }
        
        # Adjust weights based on context
        if self.context == TrustContext.INFORMATION:
            weights[TrustDimension.ACCURACY] = 2.0
            weights[TrustDimension.HONESTY] = 2.0
        elif self.context == TrustContext.ROUTING:
            weights[TrustDimension.RELIABILITY] = 2.0
            weights[TrustDimension.TIMELINESS] = 1.5
        elif self.context == TrustContext.SOCIAL:
            weights[TrustDimension.HONESTY] = 2.0
            weights[TrustDimension.COOPERATION] = 1.5
        
        return weights
    
    def update_dimension(self, dimension: TrustDimension, 
                        new_score: float, weight: float = 1.0):
        """Update a specific trust dimension"""
        current_score = self.dimensions.get(dimension, 0.5)
        
        # Apply exponential moving average with decay
        alpha = min(0.3, weight / (self.interaction_count + 1))
        self.dimensions[dimension] = (1 - alpha) * current_score + alpha * new_score
        
        self.interaction_count += 1
        self.last_updated = time.time()
        self.calculate_overall_score()
    
    def apply_temporal_decay(self, decay_rate: float = 0.99):
        """Apply temporal decay to trust scores"""
        time_factor = math.exp(-decay_rate * (time.time() - self.last_updated) / 86400)  # Daily decay
        
        for dimension in self.dimensions:
            current = self.dimensions[dimension]
            # Decay towards neutral (0.5)
            self.dimensions[dimension] = 0.5 + (current - 0.5) * time_factor
        
        self.calculate_overall_score()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'node_id': self.node_id,
            'observer_id': self.observer_id,
            'context': self.context.value,
            'dimensions': {dim.value: score for dim, score in self.dimensions.items()},
            'overall_score': self.overall_score,
            'confidence': self.confidence,
            'last_updated': self.last_updated,
            'interaction_count': self.interaction_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TrustScore':
        """Create from dictionary"""
        dimensions = {
            TrustDimension(dim): score 
            for dim, score in data.get('dimensions', {}).items()
        }
        
        score = cls(
            node_id=data['node_id'],
            observer_id=data['observer_id'],
            context=TrustContext(data['context']),
            dimensions=dimensions,
            last_updated=data.get('last_updated', time.time()),
            interaction_count=data.get('interaction_count', 0)
        )
        
        score.overall_score = data.get('overall_score', 0.5)
        score.confidence = data.get('confidence', 0.0)
        
        return score

@dataclass
class TrustRecord:
    """Individual trust record in the ledger"""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trust_score: TrustScore = None
    timestamp: float = field(default_factory=time.time)
    evidence: Dict[str, Any] = field(default_factory=dict)
    witnesses: Set[str] = field(default_factory=set)
    signature: Optional[str] = None
    hash: Optional[str] = None
    
    def __post_init__(self):
        """Calculate record hash"""
        if not self.hash:
            self.calculate_hash()
    
    def calculate_hash(self):
        """Calculate cryptographic hash of the record"""
        data = {
            'record_id': self.record_id,
            'trust_score': self.trust_score.to_dict() if self.trust_score else {},
            'timestamp': self.timestamp,
            'evidence': self.evidence,
            'witnesses': sorted(list(self.witnesses))
        }
        
        json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        self.hash = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify record integrity"""
        old_hash = self.hash
        self.calculate_hash()
        return old_hash == self.hash

class TrustLedger:
    """
    Distributed trust ledger for mesh networks
    
    Maintains cryptographically secure records of trust relationships
    with consensus mechanisms and Byzantine fault tolerance.
    """
    
    def __init__(self, node_id: str, config: Dict = None):
        self.node_id = node_id
        self.config = config or {}
        
        # Trust storage
        self.trust_records: Dict[str, TrustRecord] = {}  # record_id -> TrustRecord
        self.trust_scores: Dict[Tuple[str, str, TrustContext], TrustScore] = {}  # (observer, target, context) -> TrustScore
        self.aggregated_scores: Dict[Tuple[str, TrustContext], float] = {}  # (target, context) -> aggregated score
        
        # Consensus tracking
        self.pending_validations: Dict[str, Set[str]] = defaultdict(set)  # record_id -> validator_nodes
        self.consensus_threshold = self.config.get('consensus_threshold', 0.67)  # 67% agreement needed
        self.min_validators = self.config.get('min_validators', 3)
        
        # Trust network graph
        self.trust_graph: Dict[str, Dict[str, float]] = defaultdict(dict)  # source -> target -> score
        self.trust_paths: Dict[Tuple[str, str], List[str]] = {}  # (source, target) -> path
        
        # Temporal management
        self.decay_rate = self.config.get('trust_decay_rate', 0.01)  # 1% decay per day
        self.max_age = self.config.get('max_trust_age', 86400 * 30)  # 30 days
        
        # Performance tracking
        self.metrics = {
            'records_created': 0,
            'records_validated': 0,
            'consensus_achieved': 0,
            'consensus_failed': 0,
            'trust_updates': 0,
            'aggregations_calculated': 0
        }
        
        # Background tasks
        self.is_running = False
        
        logger.info(f"Trust ledger initialized for node {self.node_id}")
    
    async def start(self):
        """Start the trust ledger"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._maintenance_task())
        asyncio.create_task(self._aggregation_task())
        
        logger.info("Trust ledger started")
    
    async def stop(self):
        """Stop the trust ledger"""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("Trust ledger stopped")
    
    async def record_trust_interaction(self, target_node: str, 
                                     context: TrustContext,
                                     dimension_updates: Dict[TrustDimension, float],
                                     evidence: Dict[str, Any] = None,
                                     witnesses: Set[str] = None) -> str:
        """Record a trust interaction"""
        try:
            # Get or create trust score
            key = (self.node_id, target_node, context)
            if key not in self.trust_scores:
                self.trust_scores[key] = TrustScore(
                    node_id=target_node,
                    observer_id=self.node_id,
                    context=context
                )
            
            trust_score = self.trust_scores[key]
            
            # Update trust dimensions
            for dimension, score in dimension_updates.items():
                trust_score.update_dimension(dimension, score)
            
            # Create trust record
            record = TrustRecord(
                trust_score=trust_score,
                evidence=evidence or {},
                witnesses=witnesses or set()
            )
            
            # Store record
            self.trust_records[record.record_id] = record
            
            # Update trust graph
            self.trust_graph[self.node_id][target_node] = trust_score.overall_score
            
            # Trigger aggregation
            asyncio.create_task(self._update_aggregated_score(target_node, context))
            
            self.metrics['records_created'] += 1
            self.metrics['trust_updates'] += 1
            
            logger.debug(f"Recorded trust interaction: {self.node_id} -> {target_node} "
                        f"({context.value}) = {trust_score.overall_score:.3f}")
            
            return record.record_id
            
        except Exception as e:
            logger.error(f"Error recording trust interaction: {e}")
            raise
    
    async def get_trust_score(self, target_node: str, 
                            context: TrustContext = TrustContext.GENERAL,
                            include_indirect: bool = True) -> float:
        """Get trust score for a target node"""
        try:
            # Check direct trust first
            key = (self.node_id, target_node, context)
            if key in self.trust_scores:
                direct_score = self.trust_scores[key].overall_score
                
                if not include_indirect:
                    return direct_score
                
                # Get indirect trust if available
                indirect_score = await self._calculate_indirect_trust(target_node, context)
                
                # Combine direct and indirect trust
                if indirect_score is not None:
                    # Weight direct trust higher
                    combined_score = 0.7 * direct_score + 0.3 * indirect_score
                    return combined_score
                
                return direct_score
            
            # Only indirect trust available
            if include_indirect:
                indirect_score = await self._calculate_indirect_trust(target_node, context)
                if indirect_score is not None:
                    return indirect_score
            
            # No trust information available
            return 0.5  # Neutral default
            
        except Exception as e:
            logger.error(f"Error getting trust score for {target_node}: {e}")
            return 0.5
    
    async def get_aggregated_trust_score(self, target_node: str,
                                       context: TrustContext = TrustContext.GENERAL) -> float:
        """Get network-aggregated trust score for a target node"""
        key = (target_node, context)
        return self.aggregated_scores.get(key, 0.5)
    
    async def validate_trust_record(self, record_id: str, 
                                  validator_scores: Dict[str, bool]) -> bool:
        """Validate a trust record with consensus"""
        try:
            if record_id not in self.trust_records:
                logger.warning(f"Trust record {record_id} not found for validation")
                return False
            
            record = self.trust_records[record_id]
            
            # Add validator responses
            for validator_id, is_valid in validator_scores.items():
                if is_valid:
                    self.pending_validations[record_id].add(validator_id)
            
            # Check if consensus is reached
            validator_count = len(validator_scores)
            agreement_count = len(self.pending_validations[record_id])
            
            if (validator_count >= self.min_validators and 
                agreement_count / validator_count >= self.consensus_threshold):
                
                # Consensus achieved
                record.witnesses.update(self.pending_validations[record_id])
                self.pending_validations.pop(record_id, None)
                
                self.metrics['records_validated'] += 1
                self.metrics['consensus_achieved'] += 1
                
                logger.info(f"Consensus achieved for trust record {record_id} "
                           f"({agreement_count}/{validator_count} validators)")
                
                return True
            
            elif validator_count >= self.min_validators:
                # Consensus failed
                self.metrics['consensus_failed'] += 1
                logger.warning(f"Consensus failed for trust record {record_id} "
                              f"({agreement_count}/{validator_count} validators)")
                
                return False
            
            # Still waiting for more validators
            return None
            
        except Exception as e:
            logger.error(f"Error validating trust record {record_id}: {e}")
            return False
    
    async def _calculate_indirect_trust(self, target_node: str,
                                      context: TrustContext) -> Optional[float]:
        """Calculate indirect trust through trust network"""
        try:
            # Find trust paths to target
            paths = await self._find_trust_paths(self.node_id, target_node, max_depth=3)
            
            if not paths:
                return None
            
            # Calculate trust along each path
            path_trusts = []
            for path in paths:
                path_trust = 1.0
                
                for i in range(len(path) - 1):
                    source, intermediate = path[i], path[i + 1]
                    
                    # Get trust score along this edge
                    edge_key = (source, intermediate, context)
                    if edge_key in self.trust_scores:
                        edge_trust = self.trust_scores[edge_key].overall_score
                        path_trust *= edge_trust
                    else:
                        # Use aggregated score if available
                        agg_key = (intermediate, context)
                        edge_trust = self.aggregated_scores.get(agg_key, 0.5)
                        path_trust *= edge_trust
                
                path_trusts.append(path_trust)
            
            # Average path trusts, weighted by path length
            if path_trusts:
                weights = [1.0 / len(path) for path in paths]
                weighted_sum = sum(trust * weight for trust, weight in zip(path_trusts, weights))
                total_weight = sum(weights)
                
                return weighted_sum / total_weight if total_weight > 0 else None
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating indirect trust for {target_node}: {e}")
            return None
    
    async def _find_trust_paths(self, source: str, target: str, 
                              max_depth: int = 3) -> List[List[str]]:
        """Find trust paths between source and target nodes"""
        try:
            # Use BFS to find paths
            queue = deque([(source, [source])])
            paths = []
            visited = set()
            
            while queue and len(paths) < 10:  # Limit number of paths
                current_node, path = queue.popleft()
                
                if len(path) > max_depth:
                    continue
                
                if current_node == target:
                    paths.append(path)
                    continue
                
                if current_node in visited:
                    continue
                visited.add(current_node)
                
                # Add neighbors with trust relationships
                if current_node in self.trust_graph:
                    for neighbor, trust_score in self.trust_graph[current_node].items():
                        if neighbor not in path and trust_score > 0.5:  # Only follow trusted links
                            queue.append((neighbor, path + [neighbor]))
            
            return paths
            
        except Exception as e:
            logger.error(f"Error finding trust paths {source} -> {target}: {e}")
            return []
    
    async def _update_aggregated_score(self, target_node: str, context: TrustContext):
        """Update aggregated trust score for a node"""
        try:
            # Collect all trust scores for this target
            scores = []
            confidences = []
            
            for (observer, target, ctx), trust_score in self.trust_scores.items():
                if target == target_node and ctx == context:
                    scores.append(trust_score.overall_score)
                    confidences.append(trust_score.confidence)
            
            if not scores:
                return
            
            # Calculate weighted average
            if all(c == 0 for c in confidences):
                # Equal weight if no confidence info
                aggregated = sum(scores) / len(scores)
            else:
                # Weight by confidence
                weighted_sum = sum(score * conf for score, conf in zip(scores, confidences))
                total_weight = sum(confidences)
                aggregated = weighted_sum / total_weight if total_weight > 0 else sum(scores) / len(scores)
            
            # Store aggregated score
            key = (target_node, context)
            self.aggregated_scores[key] = aggregated
            
            self.metrics['aggregations_calculated'] += 1
            
            logger.debug(f"Updated aggregated trust for {target_node} ({context.value}): {aggregated:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating aggregated score for {target_node}: {e}")
    
    async def _maintenance_task(self):
        """Background maintenance task"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Apply temporal decay to trust scores
                for trust_score in self.trust_scores.values():
                    trust_score.apply_temporal_decay(self.decay_rate)
                
                # Remove old records
                old_records = [
                    record_id for record_id, record in self.trust_records.items()
                    if current_time - record.timestamp > self.max_age
                ]
                
                for record_id in old_records:
                    del self.trust_records[record_id]
                    self.pending_validations.pop(record_id, None)
                
                if old_records:
                    logger.info(f"Removed {len(old_records)} old trust records")
                
                # Update trust graph
                self._update_trust_graph()
                
                await asyncio.sleep(3600)  # Maintenance every hour
                
            except Exception as e:
                logger.error(f"Trust maintenance error: {e}")
                await asyncio.sleep(3600)
    
    async def _aggregation_task(self):
        """Background aggregation task"""
        while self.is_running:
            try:
                # Update all aggregated scores
                targets_contexts = set()
                for (observer, target, context) in self.trust_scores.keys():
                    targets_contexts.add((target, context))
                
                for target, context in targets_contexts:
                    await self._update_aggregated_score(target, context)
                
                await asyncio.sleep(300)  # Aggregate every 5 minutes
                
            except Exception as e:
                logger.error(f"Trust aggregation error: {e}")
                await asyncio.sleep(300)
    
    def _update_trust_graph(self):
        """Update trust graph with current scores"""
        try:
            # Clear old graph
            self.trust_graph.clear()
            
            # Rebuild from current trust scores
            for (observer, target, context), trust_score in self.trust_scores.items():
                if trust_score.overall_score > 0.1:  # Only include meaningful trust
                    self.trust_graph[observer][target] = trust_score.overall_score
                    
        except Exception as e:
            logger.error(f"Error updating trust graph: {e}")
    
    def get_trust_network_stats(self) -> Dict:
        """Get trust network statistics"""
        try:
            total_records = len(self.trust_records)
            total_scores = len(self.trust_scores)
            total_nodes = len(set(
                [observer for observer, _, _ in self.trust_scores.keys()] +
                [target for _, target, _ in self.trust_scores.keys()]
            ))
            
            # Calculate trust distribution
            all_scores = [score.overall_score for score in self.trust_scores.values()]
            avg_trust = sum(all_scores) / len(all_scores) if all_scores else 0.5
            
            # Count high/low trust relationships
            high_trust = len([s for s in all_scores if s > 0.8])
            low_trust = len([s for s in all_scores if s < 0.3])
            
            return {
                'total_records': total_records,
                'total_trust_scores': total_scores,
                'total_nodes': total_nodes,
                'average_trust': avg_trust,
                'high_trust_relationships': high_trust,
                'low_trust_relationships': low_trust,
                'pending_validations': len(self.pending_validations),
                'aggregated_scores': len(self.aggregated_scores),
                'trust_graph_nodes': len(self.trust_graph),
                'trust_graph_edges': sum(len(targets) for targets in self.trust_graph.values())
            }
            
        except Exception as e:
            logger.error(f"Error getting trust network stats: {e}")
            return {}
    
    def export_trust_data(self, filename: str):
        """Export trust data to file"""
        try:
            data = {
                'trust_scores': {
                    f"{observer}_{target}_{context.value}": score.to_dict()
                    for (observer, target, context), score in self.trust_scores.items()
                },
                'aggregated_scores': {
                    f"{target}_{context.value}": score
                    for (target, context), score in self.aggregated_scores.items()
                },
                'trust_records': {
                    record_id: {
                        'trust_score': record.trust_score.to_dict() if record.trust_score else None,
                        'timestamp': record.timestamp,
                        'evidence': record.evidence,
                        'witnesses': list(record.witnesses)
                    }
                    for record_id, record in self.trust_records.items()
                },
                'metrics': self.metrics,
                'stats': self.get_trust_network_stats(),
                'export_timestamp': time.time()
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Exported trust data to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting trust data: {e}")
    
    def get_metrics(self) -> Dict:
        """Get trust ledger metrics"""
        return {
            **self.metrics,
            **self.get_trust_network_stats(),
            'is_running': self.is_running
        }