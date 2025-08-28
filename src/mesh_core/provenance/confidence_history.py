"""
Confidence History Tracking System
=================================

Tracks the evolution of confidence scores over time, providing
detailed history of how trust and confidence in information
changes as it flows through the network.
"""

import time
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ChangeType(Enum):
    """Types of confidence changes"""
    INITIAL = "initial"           # Initial confidence assignment
    VERIFICATION = "verification" # Verification-based change
    PEER_FEEDBACK = "peer"       # Peer feedback
    CONSENSUS = "consensus"      # Network consensus
    DEGRADATION = "degradation"  # Time-based degradation
    CORRELATION = "correlation"  # Cross-reference correlation
    MANUAL = "manual"           # Manual adjustment

class ConfidenceMetric(Enum):
    """Different confidence metrics tracked"""
    SOURCE_TRUST = "source_trust"         # Trust in the source
    CONTENT_ACCURACY = "accuracy"         # Content accuracy score
    TEMPORAL_RELEVANCE = "temporal"       # Time-based relevance
    PEER_CONSENSUS = "consensus"          # Peer agreement level
    VERIFICATION_SCORE = "verification"   # Verification confidence
    OVERALL_CONFIDENCE = "overall"        # Combined confidence score

@dataclass
class ConfidenceEntry:
    """Single confidence history entry"""
    timestamp: float
    metric: ConfidenceMetric
    value: float                 # 0.0 to 1.0
    change_type: ChangeType
    change_reason: str
    previous_value: float
    verifier_id: Optional[str]
    metadata: Dict
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['metric'] = self.metric.value
        data['change_type'] = self.change_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConfidenceEntry':
        data['metric'] = ConfidenceMetric(data['metric'])
        data['change_type'] = ChangeType(data['change_type'])
        return cls(**data)

class ConfidenceHistory:
    """
    Confidence history tracking system
    
    Maintains detailed history of confidence score evolution
    for information items as they move through the network.
    """
    
    def __init__(self, item_id: str):
        self.item_id = item_id
        self.history: Dict[ConfidenceMetric, List[ConfidenceEntry]] = {}
        self.current_values: Dict[ConfidenceMetric, float] = {}
        self.created_at = time.time()
        
        # Initialize all metrics
        for metric in ConfidenceMetric:
            self.history[metric] = []
            self.current_values[metric] = 0.5  # Start with neutral confidence
    
    async def record_confidence_change(
        self,
        metric: ConfidenceMetric,
        new_value: float,
        change_type: ChangeType,
        reason: str,
        verifier_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> ConfidenceEntry:
        """Record a confidence change"""
        
        if metadata is None:
            metadata = {}
            
        # Validate value range
        new_value = max(0.0, min(1.0, new_value))
        
        # Get previous value
        previous_value = self.current_values.get(metric, 0.5)
        
        # Create history entry
        entry = ConfidenceEntry(
            timestamp=time.time(),
            metric=metric,
            value=new_value,
            change_type=change_type,
            change_reason=reason,
            previous_value=previous_value,
            verifier_id=verifier_id,
            metadata=metadata
        )
        
        # Update history and current value
        self.history[metric].append(entry)
        self.current_values[metric] = new_value
        
        # Update overall confidence if not directly set
        if metric != ConfidenceMetric.OVERALL_CONFIDENCE:
            await self._update_overall_confidence()
        
        logger.debug(f"Recorded confidence change for {self.item_id}: {metric.value} = {new_value:.3f} ({reason})")
        return entry
    
    async def _update_overall_confidence(self):
        """Update overall confidence based on component metrics"""
        
        # Weights for different metrics in overall score
        weights = {
            ConfidenceMetric.SOURCE_TRUST: 0.25,
            ConfidenceMetric.CONTENT_ACCURACY: 0.30,
            ConfidenceMetric.TEMPORAL_RELEVANCE: 0.15,
            ConfidenceMetric.PEER_CONSENSUS: 0.20,
            ConfidenceMetric.VERIFICATION_SCORE: 0.10
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in self.current_values:
                weighted_sum += self.current_values[metric] * weight
                total_weight += weight
        
        if total_weight > 0:
            overall_confidence = weighted_sum / total_weight
            
            # Record the calculated overall confidence
            await self.record_confidence_change(
                metric=ConfidenceMetric.OVERALL_CONFIDENCE,
                new_value=overall_confidence,
                change_type=ChangeType.CORRELATION,
                reason="calculated_from_components",
                metadata={'weights': weights, 'component_values': dict(self.current_values)}
            )
    
    def get_current_confidence(self, metric: ConfidenceMetric) -> float:
        """Get current confidence value for metric"""
        return self.current_values.get(metric, 0.5)
    
    def get_all_current_confidences(self) -> Dict[str, float]:
        """Get all current confidence values"""
        return {metric.value: value for metric, value in self.current_values.items()}
    
    def get_history_for_metric(self, metric: ConfidenceMetric) -> List[ConfidenceEntry]:
        """Get complete history for specific metric"""
        return self.history.get(metric, [])
    
    def get_recent_changes(self, hours: int = 24) -> List[ConfidenceEntry]:
        """Get recent confidence changes within time window"""
        
        cutoff_time = time.time() - (hours * 3600)
        recent_changes = []
        
        for metric_history in self.history.values():
            for entry in metric_history:
                if entry.timestamp >= cutoff_time:
                    recent_changes.append(entry)
        
        # Sort by timestamp
        recent_changes.sort(key=lambda x: x.timestamp, reverse=True)
        return recent_changes
    
    def get_confidence_trend(self, metric: ConfidenceMetric, hours: int = 168) -> Dict:
        """Get confidence trend analysis for metric over time period"""
        
        cutoff_time = time.time() - (hours * 3600)
        metric_history = self.history.get(metric, [])
        
        # Filter to time range
        relevant_entries = [
            entry for entry in metric_history
            if entry.timestamp >= cutoff_time
        ]
        
        if not relevant_entries:
            return {
                'trend': 'stable',
                'change': 0.0,
                'confidence': self.get_current_confidence(metric),
                'sample_count': 0
            }
        
        # Calculate trend
        values = [entry.value for entry in relevant_entries]
        timestamps = [entry.timestamp for entry in relevant_entries]
        
        if len(values) >= 2:
            # Simple trend calculation
            first_value = values[0]
            last_value = values[-1]
            change = last_value - first_value
            
            # Determine trend direction
            if abs(change) < 0.05:
                trend = 'stable'
            elif change > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
        else:
            trend = 'stable'
            change = 0.0
        
        # Calculate statistics
        mean_confidence = statistics.mean(values) if values else 0.5
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
        
        return {
            'trend': trend,
            'change': change,
            'mean_confidence': mean_confidence,
            'std_deviation': std_dev,
            'current_confidence': self.get_current_confidence(metric),
            'sample_count': len(values),
            'time_span_hours': (timestamps[-1] - timestamps[0]) / 3600 if len(timestamps) > 1 else 0
        }
    
    def get_confidence_volatility(self, metric: ConfidenceMetric, hours: int = 24) -> float:
        """Calculate confidence volatility (stability measure)"""
        
        cutoff_time = time.time() - (hours * 3600)
        metric_history = self.history.get(metric, [])
        
        relevant_entries = [
            entry for entry in metric_history
            if entry.timestamp >= cutoff_time
        ]
        
        if len(relevant_entries) < 2:
            return 0.0
        
        # Calculate changes between consecutive entries
        changes = []
        for i in range(1, len(relevant_entries)):
            change = abs(relevant_entries[i].value - relevant_entries[i-1].value)
            changes.append(change)
        
        # Return average absolute change as volatility measure
        return statistics.mean(changes) if changes else 0.0
    
    def analyze_change_patterns(self) -> Dict:
        """Analyze patterns in confidence changes"""
        
        all_entries = []
        for metric_history in self.history.values():
            all_entries.extend(metric_history)
        
        if not all_entries:
            return {'total_changes': 0}
        
        # Count by change type
        change_type_counts = {}
        for entry in all_entries:
            change_type = entry.change_type.value
            change_type_counts[change_type] = change_type_counts.get(change_type, 0) + 1
        
        # Count by verifier
        verifier_counts = {}
        for entry in all_entries:
            if entry.verifier_id:
                verifier_counts[entry.verifier_id] = verifier_counts.get(entry.verifier_id, 0) + 1
        
        # Calculate average change magnitude
        changes = [abs(entry.value - entry.previous_value) for entry in all_entries]
        avg_change = statistics.mean(changes) if changes else 0.0
        
        # Find most active time periods
        timestamps = [entry.timestamp for entry in all_entries]
        time_range = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
        
        return {
            'total_changes': len(all_entries),
            'by_change_type': change_type_counts,
            'by_verifier': verifier_counts,
            'average_change_magnitude': avg_change,
            'time_span_hours': time_range / 3600,
            'change_frequency': len(all_entries) / max(1, time_range / 3600),  # changes per hour
            'most_active_verifier': max(verifier_counts.items(), key=lambda x: x[1])[0] if verifier_counts else None
        }
    
    async def apply_temporal_decay(self, decay_rate: float = 0.1, hours: int = 24):
        """Apply temporal decay to confidence scores"""
        
        current_time = time.time()
        
        for metric in [ConfidenceMetric.TEMPORAL_RELEVANCE]:
            current_value = self.current_values.get(metric, 0.5)
            
            # Calculate time-based decay
            hours_elapsed = (current_time - self.created_at) / 3600
            decay_factor = max(0.1, 1.0 - (decay_rate * hours_elapsed / hours))
            new_value = current_value * decay_factor
            
            # Record the decay
            await self.record_confidence_change(
                metric=metric,
                new_value=new_value,
                change_type=ChangeType.DEGRADATION,
                reason=f"temporal_decay_after_{hours_elapsed:.1f}_hours",
                metadata={'decay_rate': decay_rate, 'decay_factor': decay_factor}
            )
    
    def export_history(self) -> Dict:
        """Export complete confidence history"""
        
        exported_history = {}
        for metric, entries in self.history.items():
            exported_history[metric.value] = [entry.to_dict() for entry in entries]
        
        return {
            'item_id': self.item_id,
            'created_at': self.created_at,
            'current_values': {metric.value: value for metric, value in self.current_values.items()},
            'history': exported_history,
            'exported_at': time.time()
        }
    
    @classmethod
    async def import_history(cls, data: Dict) -> 'ConfidenceHistory':
        """Import confidence history from exported data"""
        
        history_obj = cls(data['item_id'])
        history_obj.created_at = data.get('created_at', time.time())
        
        # Import current values
        for metric_name, value in data.get('current_values', {}).items():
            try:
                metric = ConfidenceMetric(metric_name)
                history_obj.current_values[metric] = value
            except ValueError:
                logger.warning(f"Unknown confidence metric: {metric_name}")
        
        # Import history entries
        for metric_name, entries_data in data.get('history', {}).items():
            try:
                metric = ConfidenceMetric(metric_name)
                history_obj.history[metric] = [
                    ConfidenceEntry.from_dict(entry_data)
                    for entry_data in entries_data
                ]
            except ValueError:
                logger.warning(f"Unknown confidence metric in history: {metric_name}")
        
        return history_obj