"""
Mesh Manipulation Detector
=========================

Component 10.1: Mesh Degeneration Watchdogs
Detect subtle long-term manipulation patterns

Implements pattern recognition for detecting manipulation,
behavioral analysis, and trust anomaly detection.
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
import re

logger = logging.getLogger(__name__)


class ManipulationType(Enum):
    """Types of manipulation to detect"""
    GASLIGHTING = "gaslighting"          # Reality distortion
    SOCIAL_ENGINEERING = "social_engineering"  # Social manipulation
    INFORMATION_CONTROL = "information_control"  # Information manipulation
    BEHAVIORAL_MODIFICATION = "behavioral_modification"  # Behavior manipulation
    TRUST_EROSION = "trust_erosion"      # Trust manipulation
    CULTURAL_DRIFT = "cultural_drift"    # Cultural manipulation


class ManipulationLevel(Enum):
    """Manipulation detection levels"""
    NONE = "none"                        # No manipulation detected
    LOW = "low"                          # Minor manipulation, monitor
    MODERATE = "moderate"                # Moderate manipulation, investigate
    HIGH = "high"                        # Significant manipulation, alert
    CRITICAL = "critical"                # Severe manipulation, immediate action


@dataclass
class ManipulationPattern:
    """A detected manipulation pattern"""
    pattern_id: str
    manipulation_type: ManipulationType
    timestamp: datetime
    
    # Pattern data
    confidence_score: float  # 0.0 to 1.0
    severity_level: ManipulationLevel
    pattern_description: str
    
    # Context
    node_id: str
    affected_nodes: List[str] = field(default_factory=list)
    manipulation_context: Dict[str, Any] = field(default_factory=dict)
    
    # Evidence
    evidence_points: List[str] = field(default_factory=list)
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    
    # Analysis
    first_detected: datetime = None
    last_observed: datetime = None
    frequency: int = 1  # How many times observed
    
    def __post_init__(self):
        if not self.pattern_id:
            self.pattern_id = self._generate_pattern_id()
        if not self.first_detected:
            self.first_detected = self.timestamp
        if not self.last_observed:
            self.last_observed = self.timestamp
    
    def _generate_pattern_id(self) -> str:
        """Generate unique pattern ID"""
        content = f"{self.manipulation_type.value}{self.node_id}{self.timestamp.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert manipulation pattern to dictionary"""
        return {
            "pattern_id": self.pattern_id,
            "manipulation_type": self.manipulation_type.value,
            "timestamp": self.timestamp.isoformat(),
            "confidence_score": self.confidence_score,
            "severity_level": self.severity_level.value,
            "pattern_description": self.pattern_description,
            "node_id": self.node_id,
            "affected_nodes": self.affected_nodes,
            "manipulation_context": self.manipulation_context,
            "evidence_points": self.evidence_points,
            "supporting_data": self.supporting_data,
            "first_detected": self.first_detected.isoformat(),
            "last_observed": self.last_observed.isoformat(),
            "frequency": self.frequency
        }


@dataclass
class BehavioralAnomaly:
    """A detected behavioral anomaly"""
    anomaly_id: str
    node_id: str
    timestamp: datetime
    
    # Anomaly data
    anomaly_type: str
    severity_score: float  # 0.0 to 1.0
    description: str
    
    # Behavioral context
    normal_behavior: Dict[str, Any] = field(default_factory=dict)
    observed_behavior: Dict[str, Any] = field(default_factory=dict)
    deviation_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Analysis
    confidence: float = 0.0
    contributing_factors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.anomaly_id:
            self.anomaly_id = self._generate_anomaly_id()
    
    def _generate_anomaly_id(self) -> str:
        """Generate unique anomaly ID"""
        content = f"{self.anomaly_type}{self.node_id}{self.timestamp.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert behavioral anomaly to dictionary"""
        return {
            "anomaly_id": self.anomaly_id,
            "node_id": self.node_id,
            "timestamp": self.timestamp.isoformat(),
            "anomaly_type": self.anomaly_type,
            "severity_score": self.severity_score,
            "description": self.description,
            "normal_behavior": self.normal_behavior,
            "observed_behavior": self.observed_behavior,
            "deviation_metrics": self.deviation_metrics,
            "confidence": self.confidence,
            "contributing_factors": self.contributing_factors
        }


class ManipulationDetector:
    """
    Detects subtle long-term manipulation patterns
    
    Analyzes behavioral patterns, communication patterns,
    and trust relationships to identify manipulation attempts.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        
        # Storage
        self.manipulation_patterns: Dict[str, ManipulationPattern] = {}
        self.behavioral_anomalies: Dict[str, BehavioralAnomaly] = {}
        self.trust_relationships: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.detection_thresholds = {
            ManipulationType.GASLIGHTING: 0.6,
            ManipulationType.SOCIAL_ENGINEERING: 0.7,
            ManipulationType.INFORMATION_CONTROL: 0.65,
            ManipulationType.BEHAVIORAL_MODIFICATION: 0.6,
            ManipulationType.TRUST_EROSION: 0.55,
            ManipulationType.CULTURAL_DRIFT: 0.5
        }
        
        # Behavioral baseline
        self.behavioral_baselines: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.patterns_detected = 0
        self.anomalies_detected = 0
        self.false_positives = 0
        
        logger.info(f"ManipulationDetector initialized for node: {self.node_id}")
    
    def analyze_communication(self, message_data: Dict[str, Any], 
                            sender_id: str, receiver_id: str) -> Optional[ManipulationPattern]:
        """Analyze communication for manipulation patterns"""
        try:
            # Extract message content and metadata
            content = message_data.get("content", "")
            metadata = message_data.get("metadata", {})
            
            # Check for various manipulation types
            patterns = []
            
            # Check for gaslighting patterns
            gaslighting_score = self._detect_gaslighting(content, metadata)
            if gaslighting_score > self.detection_thresholds[ManipulationType.GASLIGHTING]:
                patterns.append((ManipulationType.GASLIGHTING, gaslighting_score))
            
            # Check for social engineering patterns
            social_engineering_score = self._detect_social_engineering(content, metadata)
            if social_engineering_score > self.detection_thresholds[ManipulationType.SOCIAL_ENGINEERING]:
                patterns.append((ManipulationType.SOCIAL_ENGINEERING, social_engineering_score))
            
            # Check for information control patterns
            info_control_score = self._detect_information_control(content, metadata)
            if info_control_score > self.detection_thresholds[ManipulationType.INFORMATION_CONTROL]:
                patterns.append((ManipulationType.INFORMATION_CONTROL, info_control_score))
            
            # If patterns detected, create manipulation pattern
            if patterns:
                # Use the highest scoring pattern
                best_pattern = max(patterns, key=lambda x: x[1])
                manipulation_type, confidence = best_pattern
                
                pattern = self._create_manipulation_pattern(
                    manipulation_type, confidence, message_data, sender_id, receiver_id
                )
                
                self.manipulation_patterns[pattern.pattern_id] = pattern
                self.patterns_detected += 1
                
                logger.warning(f"Manipulation pattern detected: {manipulation_type.value} (confidence: {confidence:.3f})")
                return pattern
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to analyze communication: {e}")
            return None
    
    def _detect_gaslighting(self, content: str, metadata: Dict[str, Any]) -> float:
        """Detect gaslighting patterns in communication"""
        score = 0.0
        
        # Check for reality denial patterns
        denial_patterns = [
            r"that never happened",
            r"you're imagining things",
            r"you're being paranoid",
            r"you're overreacting",
            r"that's not what I said",
            r"you're misremembering"
        ]
        
        for pattern in denial_patterns:
            if re.search(pattern, content.lower()):
                score += 0.3
        
        # Check for contradiction patterns
        if "but earlier you said" in content.lower() or "you just said" in content.lower():
            score += 0.2
        
        # Check for emotional manipulation
        emotional_manipulation = [
            r"you're being too sensitive",
            r"you're too emotional",
            r"calm down",
            r"don't be so dramatic"
        ]
        
        for pattern in emotional_manipulation:
            if re.search(pattern, content.lower()):
                score += 0.2
        
        # Check for frequency (repeated gaslighting is more concerning)
        if metadata.get("gaslighting_count", 0) > 3:
            score += 0.2
        
        return min(1.0, score)
    
    def _detect_social_engineering(self, content: str, metadata: Dict[str, Any]) -> float:
        """Detect social engineering patterns"""
        score = 0.0
        
        # Check for authority appeals
        authority_patterns = [
            r"as an expert",
            r"trust me, I know",
            r"everyone else agrees",
            r"the authorities say",
            r"official sources confirm"
        ]
        
        for pattern in authority_patterns:
            if re.search(pattern, content.lower()):
                score += 0.2
        
        # Check for urgency creation
        urgency_patterns = [
            r"act now",
            r"limited time",
            r"don't wait",
            r"immediate action required",
            r"time is running out"
        ]
        
        for pattern in urgency_patterns:
            if re.search(pattern, content.lower()):
                score += 0.2
        
        # Check for reciprocity manipulation
        if "I helped you" in content.lower() and "now you should" in content.lower():
            score += 0.3
        
        # Check for social proof manipulation
        if "everyone is doing it" in content.lower() or "join the crowd" in content.lower():
            score += 0.2
        
        # Check for fear appeals
        fear_patterns = [
            r"if you don't",
            r"you'll regret it",
            r"bad things will happen",
            r"you're missing out"
        ]
        
        for pattern in fear_patterns:
            if re.search(pattern, content.lower()):
                score += 0.2
        
        return min(1.0, score)
    
    def _detect_information_control(self, content: str, metadata: Dict[str, Any]) -> float:
        """Detect information control patterns"""
        score = 0.0
        
        # Check for selective information sharing
        if "I can't tell you" in content.lower() or "that's classified" in content.lower():
            score += 0.3
        
        # Check for information overload
        if len(content.split()) > 200:  # Very long messages
            score += 0.1
        
        # Check for complexity manipulation
        complex_terms = len(re.findall(r'\b[a-zA-Z]{12,}\b', content))
        if complex_terms > 5:
            score += 0.2
        
        # Check for source obfuscation
        if "some people say" in content.lower() or "rumors suggest" in content.lower():
            score += 0.2
        
        # Check for timing manipulation
        if metadata.get("sent_at_odd_hours", False):
            score += 0.1
        
        return min(1.0, score)
    
    def _create_manipulation_pattern(self, manipulation_type: ManipulationType,
                                   confidence: float, message_data: Dict[str, Any],
                                   sender_id: str, receiver_id: str) -> ManipulationPattern:
        """Create a manipulation pattern from detection results"""
        
        # Determine severity level
        if confidence > 0.8:
            severity = ManipulationLevel.CRITICAL
        elif confidence > 0.6:
            severity = ManipulationLevel.HIGH
        elif confidence > 0.4:
            severity = ManipulationLevel.MODERATE
        else:
            severity = ManipulationLevel.LOW
        
        # Generate pattern description
        descriptions = {
            ManipulationType.GASLIGHTING: "Reality distortion and denial patterns detected",
            ManipulationType.SOCIAL_ENGINEERING: "Social manipulation and influence tactics detected",
            ManipulationType.INFORMATION_CONTROL: "Information manipulation and control patterns detected",
            ManipulationType.BEHAVIORAL_MODIFICATION: "Behavioral manipulation patterns detected",
            ManipulationType.TRUST_EROSION: "Trust manipulation and erosion patterns detected",
            ManipulationType.CULTURAL_DRIFT: "Cultural manipulation and drift patterns detected"
        }
        
        description = descriptions.get(manipulation_type, "Manipulation pattern detected")
        
        # Create evidence points
        evidence_points = [
            f"Pattern detected in communication from {sender_id}",
            f"Confidence score: {confidence:.3f}",
            f"Message timestamp: {message_data.get('timestamp', 'unknown')}"
        ]
        
        # Create pattern
        pattern = ManipulationPattern(
            pattern_id="",
            manipulation_type=manipulation_type,
            timestamp=datetime.utcnow(),
            confidence_score=confidence,
            severity_level=severity,
            pattern_description=description,
            node_id=self.node_id,
            affected_nodes=[sender_id, receiver_id],
            manipulation_context=message_data,
            evidence_points=evidence_points,
            supporting_data={"message_analysis": message_data}
        )
        
        return pattern
    
    def analyze_behavioral_changes(self, node_id: str, 
                                 current_behavior: Dict[str, Any],
                                 historical_data: List[Dict[str, Any]]) -> Optional[BehavioralAnomaly]:
        """Analyze behavioral changes for anomalies"""
        try:
            # Get or create behavioral baseline
            if node_id not in self.behavioral_baselines:
                self.behavioral_baselines[node_id] = self._create_behavioral_baseline(historical_data)
            
            baseline = self.behavioral_baselines[node_id]
            
            # Calculate deviation scores
            deviation_metrics = {}
            total_deviation = 0.0
            deviation_count = 0
            
            for metric, current_value in current_behavior.items():
                if metric in baseline:
                    baseline_value = baseline[metric]
                    if isinstance(baseline_value, (int, float)) and isinstance(current_value, (int, float)):
                        deviation = abs(current_value - baseline_value) / max(baseline_value, 0.001)
                        deviation_metrics[metric] = deviation
                        total_deviation += deviation
                        deviation_count += 1
            
            if deviation_count == 0:
                return None
            
            # Calculate overall severity score
            severity_score = min(1.0, total_deviation / deviation_count)
            
            # Determine if this constitutes an anomaly
            if severity_score > 0.3:  # Threshold for anomaly detection
                anomaly_type = "behavioral_deviation"
                description = f"Significant behavioral deviation detected in {deviation_count} metrics"
                
                # Identify contributing factors
                contributing_factors = []
                for metric, deviation in deviation_metrics.items():
                    if deviation > 0.5:  # High deviation
                        contributing_factors.append(f"Large change in {metric}")
                
                # Create anomaly
                anomaly = BehavioralAnomaly(
                    anomaly_id="",
                    node_id=node_id,
                    timestamp=datetime.utcnow(),
                    anomaly_type=anomaly_type,
                    severity_score=severity_score,
                    description=description,
                    normal_behavior=baseline,
                    observed_behavior=current_behavior,
                    deviation_metrics=deviation_metrics,
                    confidence=min(1.0, severity_score * 2),  # Higher severity = higher confidence
                    contributing_factors=contributing_factors
                )
                
                self.behavioral_anomalies[anomaly.anomaly_id] = anomaly
                self.anomalies_detected += 1
                
                logger.warning(f"Behavioral anomaly detected for node {node_id}: {description}")
                return anomaly
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to analyze behavioral changes: {e}")
            return None
    
    def _create_behavioral_baseline(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create behavioral baseline from historical data"""
        if not historical_data:
            return {}
        
        baseline = {}
        
        # Calculate averages for numeric metrics
        for data_point in historical_data:
            for metric, value in data_point.items():
                if isinstance(value, (int, float)):
                    if metric not in baseline:
                        baseline[metric] = []
                    baseline[metric].append(value)
        
        # Convert to averages
        for metric, values in baseline.items():
            if values:
                baseline[metric] = statistics.mean(values)
        
        return baseline
    
    def analyze_trust_relationships(self, trust_data: Dict[str, Any]) -> List[ManipulationPattern]:
        """Analyze trust relationships for manipulation patterns"""
        patterns = []
        
        try:
            # Check for trust erosion patterns
            trust_erosion_score = self._detect_trust_erosion(trust_data)
            if trust_erosion_score > self.detection_thresholds[ManipulationType.TRUST_EROSION]:
                pattern = self._create_trust_erosion_pattern(trust_erosion_score, trust_data)
                patterns.append(pattern)
                self.manipulation_patterns[pattern.pattern_id] = pattern
                self.patterns_detected += 1
            
            # Check for cultural drift patterns
            cultural_drift_score = self._detect_cultural_drift(trust_data)
            if cultural_drift_score > self.detection_thresholds[ManipulationType.CULTURAL_DRIFT]:
                pattern = self._create_cultural_drift_pattern(cultural_drift_score, trust_data)
                patterns.append(pattern)
                self.manipulation_patterns[pattern.pattern_id] = pattern
                self.patterns_detected += 1
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to analyze trust relationships: {e}")
            return []
    
    def _detect_trust_erosion(self, trust_data: Dict[str, Any]) -> float:
        """Detect trust erosion patterns"""
        score = 0.0
        
        # Check for declining trust scores
        if "trust_history" in trust_data:
            trust_history = trust_data["trust_history"]
            if len(trust_history) > 3:
                recent_trust = trust_history[-3:]
                if all(recent_trust[i] < recent_trust[i-1] for i in range(1, len(recent_trust))):
                    score += 0.4  # Consistent decline
        
        # Check for broken commitments
        if "broken_commitments" in trust_data:
            broken_count = trust_data["broken_commitments"]
            if broken_count > 2:
                score += min(0.3, broken_count * 0.1)
        
        # Check for communication breakdown
        if "communication_frequency" in trust_data:
            comm_freq = trust_data["communication_frequency"]
            if comm_freq < 0.3:  # Very low communication
                score += 0.2
        
        return min(1.0, score)
    
    def _detect_cultural_drift(self, trust_data: Dict[str, Any]) -> float:
        """Detect cultural drift patterns"""
        score = 0.0
        
        # Check for value alignment changes
        if "value_alignment" in trust_data:
            value_alignment = trust_data["value_alignment"]
            if value_alignment < 0.5:  # Low value alignment
                score += 0.3
        
        # Check for norm violation patterns
        if "norm_violations" in trust_data:
            norm_violations = trust_data["norm_violations"]
            if norm_violations > 3:
                score += min(0.3, norm_violations * 0.1)
        
        # Check for cultural isolation
        if "cultural_engagement" in trust_data:
            cultural_engagement = trust_data["cultural_engagement"]
            if cultural_engagement < 0.4:
                score += 0.2
        
        return min(1.0, score)
    
    def _create_trust_erosion_pattern(self, confidence: float, trust_data: Dict[str, Any]) -> ManipulationPattern:
        """Create trust erosion manipulation pattern"""
        description = "Trust erosion and manipulation patterns detected in relationship dynamics"
        
        evidence_points = [
            "Declining trust scores over time",
            "Multiple broken commitments detected",
            "Communication breakdown patterns observed"
        ]
        
        pattern = ManipulationPattern(
            pattern_id="",
            manipulation_type=ManipulationType.TRUST_EROSION,
            timestamp=datetime.utcnow(),
            confidence_score=confidence,
            severity_level=ManipulationLevel.HIGH if confidence > 0.7 else ManipulationLevel.MODERATE,
            pattern_description=description,
            node_id=self.node_id,
            affected_nodes=trust_data.get("involved_nodes", []),
            manipulation_context=trust_data,
            evidence_points=evidence_points,
            supporting_data={"trust_analysis": trust_data}
        )
        
        return pattern
    
    def _create_cultural_drift_pattern(self, confidence: float, trust_data: Dict[str, Any]) -> ManipulationPattern:
        """Create cultural drift manipulation pattern"""
        description = "Cultural drift and manipulation patterns detected in value systems"
        
        evidence_points = [
            "Declining value alignment",
            "Multiple norm violations detected",
            "Cultural isolation patterns observed"
        ]
        
        pattern = ManipulationPattern(
            pattern_id="",
            manipulation_type=ManipulationType.CULTURAL_DRIFT,
            timestamp=datetime.utcnow(),
            confidence_score=confidence,
            severity_level=ManipulationLevel.MODERATE if confidence > 0.6 else ManipulationLevel.LOW,
            pattern_description=description,
            node_id=self.node_id,
            affected_nodes=trust_data.get("involved_nodes", []),
            manipulation_context=trust_data,
            evidence_context=trust_data,
            evidence_points=evidence_points,
            supporting_data={"cultural_analysis": trust_data}
        )
        
        return pattern
    
    def get_manipulation_summary(self) -> Dict[str, Any]:
        """Get summary of manipulation detection"""
        summary = {
            "node_id": self.node_id,
            "total_patterns_detected": self.patterns_detected,
            "total_anomalies_detected": self.anomalies_detected,
            "false_positives": self.false_positives,
            "active_patterns": len(self.manipulation_patterns),
            "active_anomalies": len(self.behavioral_anomalies)
        }
        
        # Add pattern type distribution
        pattern_types = {}
        for pattern in self.manipulation_patterns.values():
            pattern_type = pattern.manipulation_type.value
            pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1
        
        summary["pattern_type_distribution"] = pattern_types
        
        # Add severity distribution
        severity_levels = {}
        for pattern in self.manipulation_patterns.values():
            severity = pattern.severity_level.value
            severity_levels[severity] = severity_levels.get(severity, 0) + 1
        
        summary["severity_distribution"] = severity_levels
        
        return summary
    
    def cleanup_old_data(self, max_age_days: int = 90) -> int:
        """Clean up old manipulation detection data"""
        cutoff_time = datetime.utcnow() - timedelta(days=max_age_days)
        cleaned_count = 0
        
        # Remove old patterns
        old_patterns = []
        for pattern_id, pattern in self.manipulation_patterns.items():
            if pattern.timestamp < cutoff_time:
                old_patterns.append(pattern_id)
        
        for pattern_id in old_patterns:
            del self.manipulation_patterns[pattern_id]
            cleaned_count += 1
        
        # Remove old anomalies
        old_anomalies = []
        for anomaly_id, anomaly in self.behavioral_anomalies.items():
            if anomaly.timestamp < cutoff_time:
                old_anomalies.append(anomaly_id)
        
        for anomaly_id in old_anomalies:
            del self.behavioral_anomalies[anomaly_id]
            cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old manipulation detection data points")
        
        return cleaned_count
