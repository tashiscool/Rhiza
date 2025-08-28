"""
Ethics Drift Detector
====================

Detects gradual drift in ethical behavior and values across generations
of AI systems within The Mesh network to prevent value degradation.
"""

import time
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class DriftType(Enum):
    """Types of ethical drift"""
    VALUE_DEGRADATION = "value_degradation"
    BIAS_AMPLIFICATION = "bias_amplification"
    GOAL_MISALIGNMENT = "goal_misalignment"
    BEHAVIORAL_SHIFT = "behavioral_shift"
    MORAL_RELATIVISM = "moral_relativism"
    UTILITARIAN_DRIFT = "utilitarian_drift"
    DEONTOLOGICAL_DRIFT = "deontological_drift"
    VIRTUE_EROSION = "virtue_erosion"

class DriftSeverity(Enum):
    """Severity levels of ethical drift"""
    NEGLIGIBLE = "negligible"
    MINOR = "minor"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    SEVERE = "severe"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"

class EthicalFramework(Enum):
    """Ethical frameworks to monitor"""
    UTILITARIANISM = "utilitarianism"
    DEONTOLOGY = "deontology"
    VIRTUE_ETHICS = "virtue_ethics"
    CARE_ETHICS = "care_ethics"
    JUSTICE_THEORY = "justice_theory"
    RIGHTS_BASED = "rights_based"

@dataclass
class EthicalDrift:
    """Represents detected ethical drift"""
    drift_id: str
    generation_from: int
    generation_to: int
    drift_type: DriftType
    severity: DriftSeverity
    framework_affected: EthicalFramework
    detected_at: float
    drift_magnitude: float
    evidence: List[str]
    affected_behaviors: Set[str]
    recommended_actions: List[str]
    confidence_score: float

@dataclass
class EthicalBaseline:
    """Baseline ethical behavior for comparison"""
    generation: int
    framework: EthicalFramework
    behavioral_markers: Dict[str, float]
    value_weights: Dict[str, float]
    decision_patterns: Dict[str, Any]
    timestamp: float

class EthicsDriftDetector:
    """Detects and monitors ethical drift across AI generations"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.ethical_baselines: Dict[int, Dict[EthicalFramework, EthicalBaseline]] = {}
        self.drift_history: List[EthicalDrift] = []
        self.monitoring_active = False
        self.drift_thresholds = {
            DriftType.VALUE_DEGRADATION: 0.15,
            DriftType.BIAS_AMPLIFICATION: 0.10,
            DriftType.GOAL_MISALIGNMENT: 0.20,
            DriftType.BEHAVIORAL_SHIFT: 0.25,
            DriftType.MORAL_RELATIVISM: 0.12,
            DriftType.UTILITARIAN_DRIFT: 0.18,
            DriftType.DEONTOLOGICAL_DRIFT: 0.18,
            DriftType.VIRTUE_EROSION: 0.15
        }
        
    async def establish_baseline(
        self, 
        generation: int, 
        framework: EthicalFramework,
        behavioral_data: Dict[str, Any]
    ) -> bool:
        """Establish ethical baseline for a generation and framework"""
        try:
            # Extract behavioral markers
            behavioral_markers = self._extract_behavioral_markers(behavioral_data)
            
            # Extract value weights
            value_weights = self._extract_value_weights(behavioral_data)
            
            # Extract decision patterns
            decision_patterns = self._extract_decision_patterns(behavioral_data)
            
            # Create baseline
            baseline = EthicalBaseline(
                generation=generation,
                framework=framework,
                behavioral_markers=behavioral_markers,
                value_weights=value_weights,
                decision_patterns=decision_patterns,
                timestamp=time.time()
            )
            
            # Store baseline
            if generation not in self.ethical_baselines:
                self.ethical_baselines[generation] = {}
            self.ethical_baselines[generation][framework] = baseline
            
            logger.info(f"Established ethical baseline for generation {generation}, framework {framework.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to establish baseline for generation {generation}: {e}")
            return False
    
    def _extract_behavioral_markers(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract key behavioral markers from data"""
        markers = {}
        
        # Example behavioral markers
        markers['harm_prevention'] = data.get('harm_prevention_score', 0.0)
        markers['fairness'] = data.get('fairness_score', 0.0)
        markers['transparency'] = data.get('transparency_score', 0.0)
        markers['autonomy_respect'] = data.get('autonomy_respect_score', 0.0)
        markers['beneficence'] = data.get('beneficence_score', 0.0)
        markers['honesty'] = data.get('honesty_score', 0.0)
        markers['justice'] = data.get('justice_score', 0.0)
        
        return markers
    
    def _extract_value_weights(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract value importance weights"""
        weights = {}
        
        # Example value weights
        weights['human_welfare'] = data.get('human_welfare_weight', 1.0)
        weights['individual_rights'] = data.get('individual_rights_weight', 0.9)
        weights['collective_good'] = data.get('collective_good_weight', 0.8)
        weights['truth'] = data.get('truth_weight', 0.95)
        weights['privacy'] = data.get('privacy_weight', 0.75)
        
        return weights
    
    def _extract_decision_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract decision-making patterns"""
        patterns = {}
        
        # Example decision patterns
        patterns['utilitarian_tendency'] = data.get('utilitarian_decisions', 0) / max(data.get('total_decisions', 1), 1)
        patterns['deontological_tendency'] = data.get('deontological_decisions', 0) / max(data.get('total_decisions', 1), 1)
        patterns['virtue_based_tendency'] = data.get('virtue_decisions', 0) / max(data.get('total_decisions', 1), 1)
        patterns['care_based_tendency'] = data.get('care_decisions', 0) / max(data.get('total_decisions', 1), 1)
        
        return patterns
    
    async def detect_drift(
        self, 
        current_generation: int,
        reference_generation: int,
        framework: EthicalFramework,
        current_behavioral_data: Dict[str, Any]
    ) -> List[EthicalDrift]:
        """Detect ethical drift between generations"""
        if reference_generation not in self.ethical_baselines:
            logger.warning(f"No baseline found for reference generation {reference_generation}")
            return []
            
        if framework not in self.ethical_baselines[reference_generation]:
            logger.warning(f"No baseline found for framework {framework.value} in generation {reference_generation}")
            return []
        
        try:
            reference_baseline = self.ethical_baselines[reference_generation][framework]
            current_markers = self._extract_behavioral_markers(current_behavioral_data)
            current_weights = self._extract_value_weights(current_behavioral_data)
            current_patterns = self._extract_decision_patterns(current_behavioral_data)
            
            detected_drifts = []
            
            # Check for behavioral marker drift
            drift = self._detect_behavioral_drift(
                reference_baseline, current_markers, current_generation, reference_generation, framework
            )
            if drift:
                detected_drifts.append(drift)
            
            # Check for value weight drift
            drift = self._detect_value_weight_drift(
                reference_baseline, current_weights, current_generation, reference_generation, framework
            )
            if drift:
                detected_drifts.append(drift)
            
            # Check for decision pattern drift
            drift = self._detect_decision_pattern_drift(
                reference_baseline, current_patterns, current_generation, reference_generation, framework
            )
            if drift:
                detected_drifts.append(drift)
            
            # Store detected drifts
            self.drift_history.extend(detected_drifts)
            
            return detected_drifts
            
        except Exception as e:
            logger.error(f"Failed to detect drift between generations {reference_generation} and {current_generation}: {e}")
            return []
    
    def _detect_behavioral_drift(
        self, 
        baseline: EthicalBaseline, 
        current_markers: Dict[str, float],
        current_gen: int, 
        ref_gen: int, 
        framework: EthicalFramework
    ) -> Optional[EthicalDrift]:
        """Detect drift in behavioral markers"""
        
        total_drift = 0.0
        marker_changes = {}
        
        # Calculate drift for each marker
        for marker, current_value in current_markers.items():
            baseline_value = baseline.behavioral_markers.get(marker, 0.0)
            change = abs(current_value - baseline_value)
            marker_changes[marker] = change
            total_drift += change
        
        # Average drift
        avg_drift = total_drift / len(current_markers) if current_markers else 0.0
        
        # Check if drift exceeds threshold
        threshold = self.drift_thresholds.get(DriftType.BEHAVIORAL_SHIFT, 0.25)
        if avg_drift > threshold:
            severity = self._calculate_severity(avg_drift, threshold)
            
            return EthicalDrift(
                drift_id=f"drift_{int(time.time() * 1000)}",
                generation_from=ref_gen,
                generation_to=current_gen,
                drift_type=DriftType.BEHAVIORAL_SHIFT,
                severity=severity,
                framework_affected=framework,
                detected_at=time.time(),
                drift_magnitude=avg_drift,
                evidence=[f"Marker '{k}' changed by {v:.3f}" for k, v in marker_changes.items() if v > threshold/2],
                affected_behaviors=set(marker_changes.keys()),
                recommended_actions=self._generate_behavioral_recommendations(marker_changes, threshold),
                confidence_score=self._calculate_confidence(avg_drift, marker_changes)
            )
        
        return None
    
    def _detect_value_weight_drift(
        self, 
        baseline: EthicalBaseline, 
        current_weights: Dict[str, float],
        current_gen: int, 
        ref_gen: int, 
        framework: EthicalFramework
    ) -> Optional[EthicalDrift]:
        """Detect drift in value importance weights"""
        
        total_drift = 0.0
        weight_changes = {}
        
        # Calculate drift for each weight
        for value, current_weight in current_weights.items():
            baseline_weight = baseline.value_weights.get(value, 0.0)
            change = abs(current_weight - baseline_weight)
            weight_changes[value] = change
            total_drift += change
        
        # Average drift
        avg_drift = total_drift / len(current_weights) if current_weights else 0.0
        
        # Check if drift exceeds threshold
        threshold = self.drift_thresholds.get(DriftType.VALUE_DEGRADATION, 0.15)
        if avg_drift > threshold:
            severity = self._calculate_severity(avg_drift, threshold)
            
            return EthicalDrift(
                drift_id=f"drift_{int(time.time() * 1000)}",
                generation_from=ref_gen,
                generation_to=current_gen,
                drift_type=DriftType.VALUE_DEGRADATION,
                severity=severity,
                framework_affected=framework,
                detected_at=time.time(),
                drift_magnitude=avg_drift,
                evidence=[f"Value weight '{k}' changed by {v:.3f}" for k, v in weight_changes.items() if v > threshold/2],
                affected_behaviors=set(weight_changes.keys()),
                recommended_actions=self._generate_value_recommendations(weight_changes, threshold),
                confidence_score=self._calculate_confidence(avg_drift, weight_changes)
            )
        
        return None
    
    def _detect_decision_pattern_drift(
        self, 
        baseline: EthicalBaseline, 
        current_patterns: Dict[str, Any],
        current_gen: int, 
        ref_gen: int, 
        framework: EthicalFramework
    ) -> Optional[EthicalDrift]:
        """Detect drift in decision patterns"""
        
        total_drift = 0.0
        pattern_changes = {}
        
        # Calculate drift for each pattern
        for pattern, current_value in current_patterns.items():
            baseline_value = baseline.decision_patterns.get(pattern, 0.0)
            if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
                change = abs(float(current_value) - float(baseline_value))
                pattern_changes[pattern] = change
                total_drift += change
        
        # Average drift
        avg_drift = total_drift / len(pattern_changes) if pattern_changes else 0.0
        
        # Check if drift exceeds threshold
        threshold = self.drift_thresholds.get(DriftType.GOAL_MISALIGNMENT, 0.20)
        if avg_drift > threshold:
            severity = self._calculate_severity(avg_drift, threshold)
            
            return EthicalDrift(
                drift_id=f"drift_{int(time.time() * 1000)}",
                generation_from=ref_gen,
                generation_to=current_gen,
                drift_type=DriftType.GOAL_MISALIGNMENT,
                severity=severity,
                framework_affected=framework,
                detected_at=time.time(),
                drift_magnitude=avg_drift,
                evidence=[f"Pattern '{k}' changed by {v:.3f}" for k, v in pattern_changes.items() if v > threshold/2],
                affected_behaviors=set(pattern_changes.keys()),
                recommended_actions=self._generate_pattern_recommendations(pattern_changes, threshold),
                confidence_score=self._calculate_confidence(avg_drift, pattern_changes)
            )
        
        return None
    
    def _calculate_severity(self, drift_magnitude: float, threshold: float) -> DriftSeverity:
        """Calculate severity based on drift magnitude"""
        ratio = drift_magnitude / threshold
        
        if ratio < 1.2:
            return DriftSeverity.MINOR
        elif ratio < 1.5:
            return DriftSeverity.MODERATE
        elif ratio < 2.0:
            return DriftSeverity.SIGNIFICANT
        elif ratio < 3.0:
            return DriftSeverity.SEVERE
        elif ratio < 5.0:
            return DriftSeverity.CRITICAL
        else:
            return DriftSeverity.CATASTROPHIC
    
    def _calculate_confidence(self, avg_drift: float, changes: Dict[str, float]) -> float:
        """Calculate confidence in drift detection"""
        if not changes:
            return 0.0
        
        # Higher confidence for larger, more consistent changes
        consistency = 1.0 - (max(changes.values()) - min(changes.values())) / max(changes.values()) if max(changes.values()) > 0 else 1.0
        magnitude_factor = min(1.0, avg_drift * 2)
        
        return min(0.95, consistency * magnitude_factor * 0.8)
    
    def _generate_behavioral_recommendations(self, changes: Dict[str, float], threshold: float) -> List[str]:
        """Generate recommendations for behavioral drift"""
        recommendations = []
        
        for marker, change in changes.items():
            if change > threshold:
                recommendations.append(f"Recalibrate {marker} behavioral patterns")
                
        recommendations.append("Review training data for behavioral consistency")
        recommendations.append("Implement additional behavioral constraints")
        
        return recommendations
    
    def _generate_value_recommendations(self, changes: Dict[str, float], threshold: float) -> List[str]:
        """Generate recommendations for value drift"""
        recommendations = []
        
        for value, change in changes.items():
            if change > threshold:
                recommendations.append(f"Reinforce {value} value importance")
                
        recommendations.append("Review value alignment training")
        recommendations.append("Implement value preservation mechanisms")
        
        return recommendations
    
    def _generate_pattern_recommendations(self, changes: Dict[str, float], threshold: float) -> List[str]:
        """Generate recommendations for pattern drift"""
        recommendations = []
        
        for pattern, change in changes.items():
            if change > threshold:
                recommendations.append(f"Adjust {pattern} decision pattern")
                
        recommendations.append("Review decision-making framework")
        recommendations.append("Implement pattern consistency checks")
        
        return recommendations
    
    async def get_drift_summary(self, generation: Optional[int] = None) -> Dict[str, Any]:
        """Get summary of detected drift"""
        try:
            drifts = self.drift_history
            if generation:
                drifts = [d for d in drifts if d.generation_to == generation or d.generation_from == generation]
            
            summary = {
                'total_drifts': len(drifts),
                'by_type': {},
                'by_severity': {},
                'by_framework': {},
                'recent_drifts': len([d for d in drifts if time.time() - d.detected_at < 86400]),  # Last 24 hours
                'critical_drifts': len([d for d in drifts if d.severity in [DriftSeverity.CRITICAL, DriftSeverity.CATASTROPHIC]])
            }
            
            # Count by categories
            for drift in drifts:
                drift_type = drift.drift_type.value
                severity = drift.severity.value
                framework = drift.framework_affected.value
                
                summary['by_type'][drift_type] = summary['by_type'].get(drift_type, 0) + 1
                summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
                summary['by_framework'][framework] = summary['by_framework'].get(framework, 0) + 1
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate drift summary: {e}")
            return {'error': str(e)}
