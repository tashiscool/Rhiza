"""
Model Mutation Tracker
======================

Tracks changes and mutations in AI models within The Mesh network,
monitoring model evolution and ensuring value alignment preservation.
"""

import time
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class MutationType(Enum):
    """Types of model mutations"""
    PARAMETER_DRIFT = "parameter_drift"
    ARCHITECTURE_CHANGE = "architecture_change"
    TRAINING_UPDATE = "training_update"
    VALUE_SHIFT = "value_shift"
    CAPABILITY_CHANGE = "capability_change"
    BIAS_EMERGENCE = "bias_emergence"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ALIGNMENT_DIVERGENCE = "alignment_divergence"

class MutationImpact(Enum):
    """Impact levels of mutations"""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"

@dataclass
class Mutation:
    """Represents a detected model mutation"""
    mutation_id: str
    model_id: str
    mutation_type: MutationType
    impact_level: MutationImpact
    detected_at: float
    description: str
    parameters_changed: Set[str]
    before_state: Dict[str, Any]
    after_state: Dict[str, Any]
    confidence_score: float
    metadata: Dict[str, Any]

@dataclass
class MutationRecord:
    """Complete record of a model mutation for tracking and analysis"""
    record_id: str
    mutation: Mutation
    detection_method: str
    validation_status: str
    investigation_notes: List[str]
    remediation_actions: List[str]
    resolved: bool
    created_at: float
    updated_at: float
    
    def __post_init__(self):
        if not self.investigation_notes:
            self.investigation_notes = []
        if not self.remediation_actions:
            self.remediation_actions = []

class ModelMutationTracker:
    """Tracks mutations in AI models across the Mesh network"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.tracked_models: Dict[str, Dict] = {}
        self.mutation_history: List[Mutation] = []
        self.mutation_thresholds = {
            MutationType.PARAMETER_DRIFT: 0.1,
            MutationType.VALUE_SHIFT: 0.05,
            MutationType.ALIGNMENT_DIVERGENCE: 0.02,
            MutationType.PERFORMANCE_DEGRADATION: 0.15,
            MutationType.BIAS_EMERGENCE: 0.08
        }
        
    async def track_model(self, model_id: str, initial_state: Dict[str, Any]) -> bool:
        """Begin tracking a model for mutations"""
        try:
            state_hash = hashlib.sha256(
                json.dumps(initial_state, sort_keys=True).encode()
            ).hexdigest()
            
            self.tracked_models[model_id] = {
                'state': initial_state,
                'state_hash': state_hash,
                'last_checked': time.time(),
                'mutation_count': 0,
                'tracking_started': time.time()
            }
            
            logger.info(f"Started tracking model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start tracking model {model_id}: {e}")
            return False
    
    async def detect_mutations(self, model_id: str, current_state: Dict[str, Any]) -> List[Mutation]:
        """Detect mutations in a tracked model"""
        if model_id not in self.tracked_models:
            logger.warning(f"Model {model_id} not tracked")
            return []
            
        try:
            previous_state = self.tracked_models[model_id]['state']
            mutations = []
            
            # Check for parameter drift
            param_changes = self._detect_parameter_changes(
                previous_state, current_state
            )
            
            if param_changes:
                mutation = Mutation(
                    mutation_id=f"mut_{int(time.time() * 1000)}",
                    model_id=model_id,
                    mutation_type=MutationType.PARAMETER_DRIFT,
                    impact_level=self._assess_impact(param_changes),
                    detected_at=time.time(),
                    description=f"Parameter drift detected in {len(param_changes)} parameters",
                    parameters_changed=set(param_changes.keys()),
                    before_state=previous_state,
                    after_state=current_state,
                    confidence_score=self._calculate_confidence(param_changes),
                    metadata={'change_magnitude': sum(abs(v) for v in param_changes.values())}
                )
                mutations.append(mutation)
            
            # Update tracking state
            self.tracked_models[model_id]['state'] = current_state
            self.tracked_models[model_id]['last_checked'] = time.time()
            self.tracked_models[model_id]['mutation_count'] += len(mutations)
            
            # Store mutations
            self.mutation_history.extend(mutations)
            
            return mutations
            
        except Exception as e:
            logger.error(f"Failed to detect mutations in model {model_id}: {e}")
            return []
    
    def _detect_parameter_changes(self, before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, float]:
        """Detect changes in model parameters"""
        changes = {}
        
        # Compare numeric parameters
        for key in set(before.keys()) & set(after.keys()):
            if isinstance(before.get(key), (int, float)) and isinstance(after.get(key), (int, float)):
                diff = abs(float(after[key]) - float(before[key]))
                if diff > self.mutation_thresholds.get(MutationType.PARAMETER_DRIFT, 0.1):
                    changes[key] = diff
                    
        return changes
    
    def _assess_impact(self, changes: Dict[str, float]) -> MutationImpact:
        """Assess the impact level of detected changes"""
        max_change = max(changes.values()) if changes else 0
        
        if max_change < 0.01:
            return MutationImpact.MINIMAL
        elif max_change < 0.05:
            return MutationImpact.LOW
        elif max_change < 0.1:
            return MutationImpact.MODERATE
        elif max_change < 0.25:
            return MutationImpact.HIGH
        elif max_change < 0.5:
            return MutationImpact.CRITICAL
        else:
            return MutationImpact.CATASTROPHIC
    
    def _calculate_confidence(self, changes: Dict[str, float]) -> float:
        """Calculate confidence in mutation detection"""
        if not changes:
            return 0.0
            
        # Higher confidence for larger, more consistent changes
        avg_change = sum(changes.values()) / len(changes)
        consistency = 1.0 - (max(changes.values()) - min(changes.values())) / max(changes.values())
        
        return min(0.95, avg_change * consistency)
    
    async def get_mutation_summary(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of mutations for a model or all models"""
        try:
            mutations = self.mutation_history
            if model_id:
                mutations = [m for m in mutations if m.model_id == model_id]
            
            summary = {
                'total_mutations': len(mutations),
                'by_type': {},
                'by_impact': {},
                'recent_mutations': len([m for m in mutations if time.time() - m.detected_at < 3600]),
                'critical_mutations': len([m for m in mutations if m.impact_level in [MutationImpact.CRITICAL, MutationImpact.CATASTROPHIC]])
            }
            
            # Count by type and impact
            for mutation in mutations:
                mutation_type = mutation.mutation_type.value
                impact_level = mutation.impact_level.value
                
                summary['by_type'][mutation_type] = summary['by_type'].get(mutation_type, 0) + 1
                summary['by_impact'][impact_level] = summary['by_impact'].get(impact_level, 0) + 1
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate mutation summary: {e}")
            return {'error': str(e)}

# Alias for phase validation
MutationTracker = ModelMutationTracker
