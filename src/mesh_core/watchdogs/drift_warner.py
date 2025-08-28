"""
Mesh Drift Warner
================

Component 10.1: Mesh Degeneration Watchdogs
Warn about cultural/value drift

Implements drift detection, cultural change monitoring,
and value system analysis.
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math
import statistics

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of drift to monitor"""
    CULTURAL = "cultural"                # Cultural drift
    VALUE = "value"                      # Value system drift
    BEHAVIORAL = "behavioral"            # Behavioral drift
    NORMATIVE = "normative"              # Normative drift
    IDEOLOGICAL = "ideological"          # Ideological drift
    GENERATIONAL = "generational"        # Generational drift


class DriftSeverity(Enum):
    """Severity levels of drift"""
    NONE = "none"                        # No drift detected
    MINOR = "minor"                      # Minor drift, monitor
    MODERATE = "moderate"                # Moderate drift, investigate
    MAJOR = "major"                      # Major drift, alert
    CRITICAL = "critical"                # Critical drift, immediate action


@dataclass
class DriftAlert:
    """A drift detection alert"""
    alert_id: str
    drift_type: DriftType
    timestamp: datetime
    
    # Alert data
    severity: DriftSeverity
    confidence_score: float  # 0.0 to 1.0
    drift_description: str
    
    # Context
    node_id: str
    affected_communities: List[str] = field(default_factory=list)
    drift_context: Dict[str, Any] = field(default_factory=dict)
    
    # Analysis
    drift_magnitude: float = 0.0  # 0.0 to 1.0
    drift_rate: float = 0.0  # Rate of change
    baseline_comparison: Dict[str, Any] = field(default_factory=dict)
    
    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)
    urgency_level: str = "medium"  # low, medium, high, critical
    
    def __post_init__(self):
        if not self.alert_id:
            self.alert_id = self._generate_alert_id()
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        content = f"{self.drift_type.value}{self.node_id}{self.timestamp.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert drift alert to dictionary"""
        return {
            "alert_id": self.alert_id,
            "drift_type": self.drift_type.value,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "confidence_score": self.confidence_score,
            "drift_description": self.drift_description,
            "node_id": self.node_id,
            "affected_communities": self.affected_communities,
            "drift_context": self.drift_context,
            "drift_magnitude": self.drift_magnitude,
            "drift_rate": self.drift_rate,
            "baseline_comparison": self.baseline_comparison,
            "recommended_actions": self.recommended_actions,
            "urgency_level": self.urgency_level
        }


@dataclass
class CulturalBaseline:
    """Baseline cultural values and norms"""
    baseline_id: str
    community_id: str
    timestamp: datetime
    
    # Cultural metrics
    core_values: Dict[str, float] = field(default_factory=dict)
    behavioral_norms: Dict[str, float] = field(default_factory=dict)
    cultural_practices: Dict[str, float] = field(default_factory=dict)
    
    # Stability metrics
    value_stability: float = 0.0  # How stable values are
    norm_consistency: float = 0.0  # How consistent norms are
    practice_continuity: float = 0.0  # How continuous practices are
    
    def __post_init__(self):
        if not self.baseline_id:
            self.baseline_id = self._generate_baseline_id()
    
    def _generate_baseline_id(self) -> str:
        """Generate unique baseline ID"""
        content = f"{self.community_id}{self.timestamp.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cultural baseline to dictionary"""
        return {
            "baseline_id": self.baseline_id,
            "community_id": self.community_id,
            "timestamp": self.timestamp.isoformat(),
            "core_values": self.core_values,
            "behavioral_norms": self.behavioral_norms,
            "cultural_practices": self.cultural_practices,
            "value_stability": self.value_stability,
            "norm_consistency": self.norm_consistency,
            "practice_continuity": self.practice_continuity
        }


class DriftWarner:
    """
    Warns about cultural and value drift
    
    Monitors cultural changes, value system evolution,
    and behavioral shifts to detect unhealthy drift.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        
        # Storage
        self.drift_alerts: Dict[str, DriftAlert] = {}
        self.cultural_baselines: Dict[str, CulturalBaseline] = {}
        self.drift_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Configuration
        self.drift_thresholds = {
            DriftType.CULTURAL: 0.6,
            DriftType.VALUE: 0.7,
            DriftType.BEHAVIORAL: 0.5,
            DriftType.NORMATIVE: 0.6,
            DriftType.IDEOLOGICAL: 0.8,
            DriftType.GENERATIONAL: 0.5
        }
        
        # Baseline parameters
        self.baseline_update_interval = timedelta(days=30)
        self.drift_detection_window = timedelta(days=90)
        
        # Performance metrics
        self.alerts_generated = 0
        self.baselines_updated = 0
        self.drift_events_detected = 0
        
        logger.info(f"DriftWarner initialized for node: {self.node_id}")
    
    def update_cultural_baseline(self, community_id: str, cultural_data: Dict[str, Any]) -> CulturalBaseline:
        """Update cultural baseline for a community"""
        try:
            # Extract cultural metrics
            core_values = cultural_data.get("core_values", {})
            behavioral_norms = cultural_data.get("behavioral_norms", {})
            cultural_practices = cultural_data.get("cultural_practices", {})
            
            # Calculate stability metrics
            value_stability = self._calculate_value_stability(core_values)
            norm_consistency = self._calculate_norm_consistency(behavioral_norms)
            practice_continuity = self._calculate_practice_continuity(cultural_practices)
            
            # Create or update baseline
            baseline = CulturalBaseline(
                baseline_id="",
                community_id=community_id,
                timestamp=datetime.utcnow(),
                core_values=core_values,
                behavioral_norms=behavioral_norms,
                cultural_practices=cultural_practices,
                value_stability=value_stability,
                norm_consistency=norm_consistency,
                practice_continuity=practice_continuity
            )
            
            # Store baseline
            self.cultural_baselines[community_id] = baseline
            self.baselines_updated += 1
            
            logger.info(f"Updated cultural baseline for community: {community_id}")
            return baseline
            
        except Exception as e:
            logger.error(f"Failed to update cultural baseline: {e}")
            raise
    
    def _calculate_value_stability(self, core_values: Dict[str, float]) -> float:
        """Calculate stability of core values"""
        if not core_values:
            return 0.0
        
        # Higher values indicate more stable values
        # For demo, assume moderate stability
        return 0.7
    
    def _calculate_norm_consistency(self, behavioral_norms: Dict[str, float]) -> float:
        """Calculate consistency of behavioral norms"""
        if not behavioral_norms:
            return 0.0
        
        # Higher values indicate more consistent norms
        # For demo, assume moderate consistency
        return 0.6
    
    def _calculate_practice_continuity(self, cultural_practices: Dict[str, float]) -> float:
        """Calculate continuity of cultural practices"""
        if not cultural_practices:
            return 0.0
        
        # Higher values indicate more continuous practices
        # For demo, assume moderate continuity
        return 0.5
    
    def detect_cultural_drift(self, community_id: str, current_data: Dict[str, Any]) -> Optional[DriftAlert]:
        """Detect cultural drift in a community"""
        try:
            # Get baseline for comparison
            if community_id not in self.cultural_baselines:
                logger.warning(f"No baseline available for community: {community_id}")
                return None
            
            baseline = self.cultural_baselines[community_id]
            
            # Calculate drift metrics
            drift_magnitude = self._calculate_drift_magnitude(current_data, baseline)
            drift_rate = self._calculate_drift_rate(community_id, current_data, baseline)
            
            # Determine drift type and severity
            drift_type = self._determine_drift_type(current_data, baseline)
            severity = self._determine_drift_severity(drift_magnitude, drift_rate)
            
            # Check if drift exceeds threshold
            threshold = self.drift_thresholds.get(drift_type, 0.5)
            if drift_magnitude < threshold:
                return None
            
            # Create drift alert
            alert = self._create_drift_alert(
                drift_type, severity, drift_magnitude, drift_rate,
                community_id, current_data, baseline
            )
            
            # Store alert
            self.drift_alerts[alert.alert_id] = alert
            self.alerts_generated += 1
            
            # Update drift history
            if community_id not in self.drift_history:
                self.drift_history[community_id] = []
            
            self.drift_history[community_id].append({
                "timestamp": datetime.utcnow().isoformat(),
                "drift_magnitude": drift_magnitude,
                "drift_rate": drift_rate,
                "drift_type": drift_type.value
            })
            
            self.drift_events_detected += 1
            
            logger.warning(f"Cultural drift detected in community {community_id}: {drift_type.value} (magnitude: {drift_magnitude:.3f})")
            return alert
            
        except Exception as e:
            logger.error(f"Failed to detect cultural drift: {e}")
            return None
    
    def _calculate_drift_magnitude(self, current_data: Dict[str, Any], baseline: CulturalBaseline) -> float:
        """Calculate magnitude of cultural drift"""
        total_drift = 0.0
        drift_count = 0
        
        # Compare core values
        current_values = current_data.get("core_values", {})
        for value_name, baseline_value in baseline.core_values.items():
            if value_name in current_values:
                current_value = current_values[value_name]
                drift = abs(current_value - baseline_value)
                total_drift += drift
                drift_count += 1
        
        # Compare behavioral norms
        current_norms = current_data.get("behavioral_norms", {})
        for norm_name, baseline_norm in baseline.behavioral_norms.items():
            if norm_name in current_norms:
                current_norm = current_norms[norm_name]
                drift = abs(current_norm - baseline_norm)
                total_drift += drift
                drift_count += 1
        
        # Compare cultural practices
        current_practices = current_data.get("cultural_practices", {})
        for practice_name, baseline_practice in baseline.cultural_practices.items():
            if practice_name in current_practices:
                current_practice = current_practices[practice_name]
                drift = abs(current_practice - baseline_practice)
                total_drift += drift
                drift_count += 1
        
        if drift_count == 0:
            return 0.0
        
        # Return average drift magnitude
        return total_drift / drift_count
    
    def _calculate_drift_rate(self, community_id: str, current_data: Dict[str, Any], 
                            baseline: CulturalBaseline) -> float:
        """Calculate rate of cultural drift"""
        if community_id not in self.drift_history:
            return 0.0
        
        history = self.drift_history[community_id]
        if len(history) < 2:
            return 0.0
        
        # Calculate rate of change over time
        recent_history = history[-5:]  # Last 5 measurements
        
        if len(recent_history) < 2:
            return 0.0
        
        # Calculate average rate of change
        total_rate = 0.0
        rate_count = 0
        
        for i in range(1, len(recent_history)):
            prev_magnitude = recent_history[i-1]["drift_magnitude"]
            curr_magnitude = recent_history[i]["drift_magnitude"]
            
            if prev_magnitude > 0:
                rate = (curr_magnitude - prev_magnitude) / prev_magnitude
                total_rate += abs(rate)
                rate_count += 1
        
        if rate_count == 0:
            return 0.0
        
        return total_rate / rate_count
    
    def _determine_drift_type(self, current_data: Dict[str, Any], baseline: CulturalBaseline) -> DriftType:
        """Determine the type of cultural drift"""
        # Analyze which aspects have drifted most
        value_drift = 0.0
        norm_drift = 0.0
        practice_drift = 0.0
        
        # Calculate value drift
        current_values = current_data.get("core_values", {})
        for value_name, baseline_value in baseline.core_values.items():
            if value_name in current_values:
                current_value = current_values[value_name]
                drift = abs(current_value - baseline_value)
                value_drift = max(value_drift, drift)
        
        # Calculate norm drift
        current_norms = current_data.get("behavioral_norms", {})
        for norm_name, baseline_norm in baseline.behavioral_norms.items():
            if norm_name in current_norms:
                current_norm = current_norms[norm_name]
                drift = abs(current_norm - baseline_norm)
                norm_drift = max(norm_drift, drift)
        
        # Calculate practice drift
        current_practices = current_data.get("cultural_practices", {})
        for practice_name, baseline_practice in baseline.cultural_practices.items():
            if practice_name in current_practices:
                current_practice = current_practices[practice_name]
                drift = abs(current_practice - baseline_practice)
                practice_drift = max(practice_drift, drift)
        
        # Determine primary drift type
        if value_drift > norm_drift and value_drift > practice_drift:
            return DriftType.VALUE
        elif norm_drift > value_drift and norm_drift > practice_drift:
            return DriftType.NORMATIVE
        else:
            return DriftType.CULTURAL
    
    def _determine_drift_severity(self, drift_magnitude: float, drift_rate: float) -> DriftSeverity:
        """Determine severity of cultural drift"""
        # Combine magnitude and rate for severity assessment
        severity_score = (drift_magnitude * 0.7) + (drift_rate * 0.3)
        
        if severity_score > 0.8:
            return DriftSeverity.CRITICAL
        elif severity_score > 0.6:
            return DriftSeverity.MAJOR
        elif severity_score > 0.4:
            return DriftSeverity.MODERATE
        elif severity_score > 0.2:
            return DriftSeverity.MINOR
        else:
            return DriftSeverity.NONE
    
    def _create_drift_alert(self, drift_type: DriftType, severity: DriftSeverity,
                           drift_magnitude: float, drift_rate: float,
                           community_id: str, current_data: Dict[str, Any],
                           baseline: CulturalBaseline) -> DriftAlert:
        """Create a drift alert"""
        
        # Generate description
        descriptions = {
            DriftType.CULTURAL: "Cultural drift detected in community practices and traditions",
            DriftType.VALUE: "Value system drift detected in core community values",
            DriftType.BEHAVIORAL: "Behavioral drift detected in community norms",
            DriftType.NORMATIVE: "Normative drift detected in community standards",
            DriftType.IDEOLOGICAL: "Ideological drift detected in community beliefs",
            DriftType.GENERATIONAL: "Generational drift detected in community continuity"
        }
        
        description = descriptions.get(drift_type, "Cultural drift detected")
        
        # Determine urgency level
        if severity == DriftSeverity.CRITICAL:
            urgency = "critical"
        elif severity == DriftSeverity.MAJOR:
            urgency = "high"
        elif severity == DriftSeverity.MODERATE:
            urgency = "medium"
        else:
            urgency = "low"
        
        # Generate recommended actions
        recommended_actions = self._generate_recommended_actions(drift_type, severity, drift_magnitude)
        
        # Create alert
        alert = DriftAlert(
            alert_id="",
            drift_type=drift_type,
            timestamp=datetime.utcnow(),
            severity=severity,
            confidence_score=min(1.0, drift_magnitude * 1.2),  # Higher drift = higher confidence
            drift_description=description,
            node_id=self.node_id,
            affected_communities=[community_id],
            drift_context=current_data,
            drift_magnitude=drift_magnitude,
            drift_rate=drift_rate,
            baseline_comparison=baseline.to_dict(),
            recommended_actions=recommended_actions,
            urgency_level=urgency
        )
        
        return alert
    
    def _generate_recommended_actions(self, drift_type: DriftType, severity: DriftSeverity,
                                    drift_magnitude: float) -> List[str]:
        """Generate recommended actions for addressing drift"""
        actions = []
        
        if severity in [DriftSeverity.CRITICAL, DriftSeverity.MAJOR]:
            actions.append("Immediate community dialogue and reflection")
            actions.append("Review and reaffirm core values")
            actions.append("Establish cultural preservation measures")
        
        if drift_type == DriftType.VALUE:
            actions.append("Conduct value clarification workshops")
            actions.append("Document and preserve traditional values")
        
        elif drift_type == DriftType.NORMATIVE:
            actions.append("Review and update community guidelines")
            actions.append("Strengthen norm enforcement mechanisms")
        
        elif drift_type == DriftType.GENERATIONAL:
            actions.append("Facilitate intergenerational dialogue")
            actions.append("Create knowledge transfer programs")
        
        if severity == DriftSeverity.MODERATE:
            actions.append("Monitor drift patterns over time")
            actions.append("Engage community leaders in discussion")
        
        return actions
    
    def get_drift_summary(self, community_id: str = None) -> Dict[str, Any]:
        """Get summary of drift monitoring"""
        if community_id:
            alerts = [a for a in self.drift_alerts.values() if community_id in a.affected_communities]
        else:
            alerts = list(self.drift_alerts.values())
        
        summary = {
            "node_id": self.node_id,
            "total_alerts_generated": self.alerts_generated,
            "total_baselines_updated": self.baselines_updated,
            "total_drift_events": self.drift_events_detected,
            "active_alerts": len(alerts),
            "active_baselines": len(self.cultural_baselines)
        }
        
        # Add drift type distribution
        drift_types = {}
        for alert in alerts:
            drift_type = alert.drift_type.value
            drift_types[drift_type] = drift_types.get(drift_type, 0) + 1
        
        summary["drift_type_distribution"] = drift_types
        
        # Add severity distribution
        severity_levels = {}
        for alert in alerts:
            severity = alert.severity.value
            severity_levels[severity] = severity_levels.get(severity, 0) + 1
        
        summary["severity_distribution"] = severity_levels
        
        return summary
    
    def cleanup_old_data(self, max_age_days: int = 90) -> int:
        """Clean up old drift monitoring data"""
        cutoff_time = datetime.utcnow() - timedelta(days=max_age_days)
        cleaned_count = 0
        
        # Remove old alerts
        old_alerts = []
        for alert_id, alert in self.drift_alerts.items():
            if alert.timestamp < cutoff_time:
                old_alerts.append(alert_id)
        
        for alert_id in old_alerts:
            del self.drift_alerts[alert_id]
            cleaned_count += 1
        
        # Remove old drift history
        for community_id in list(self.drift_history.keys()):
            old_entries = []
            for entry in self.drift_history[community_id]:
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if entry_time < cutoff_time:
                    old_entries.append(entry)
            
            for entry in old_entries:
                self.drift_history[community_id].remove(entry)
                cleaned_count += 1
            
            # Remove empty history
            if not self.drift_history[community_id]:
                del self.drift_history[community_id]
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old drift monitoring data points")
        
        return cleaned_count

