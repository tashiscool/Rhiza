"""
Corruption Detector - Identifies malicious and corrupted nodes

Uses statistical analysis, behavioral pattern recognition, and trust
divergence to detect nodes that may be compromised or acting maliciously.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

try:
    from mesh_core.trust.trust_ledger import TrustLedger
    from mesh_core.network.network_health import NetworkHealth as NetworkHealthMonitor
    from mesh_core.config_manager import get_component_config
except ImportError:
    # Fallback to relative imports
    from ..trust.trust_ledger import TrustLedger
    from ..network.network_health import NetworkHealth as NetworkHealthMonitor
    from ..config_manager import get_component_config
except ImportError:
    # Mock classes for testing
    class TrustLedger:
        def __init__(self):
            pass
    
    class NetworkHealthMonitor:
        def __init__(self):
            pass
    
    def get_component_config(component):
        return {}

logger = logging.getLogger(__name__)

class CorruptionType(Enum):
    """Types of corruption that can be detected"""
    TRUST_MANIPULATION = "trust_manipulation"
    DATA_POISONING = "data_poisoning"
    SYBIL_ATTACK = "sybil_attack"
    ECLIPSE_ATTACK = "eclipse_attack"
    REPLAY_ATTACK = "replay_attack"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"
    CONSENSUS_MANIPULATION = "consensus_manipulation"

class CorruptionLevel(Enum):
    """Severity levels of corruption"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class CorruptionDetection:
    """Result of corruption detection analysis"""
    node_id: str
    corruption_type: CorruptionType
    corruption_level: CorruptionLevel
    confidence_score: float
    evidence: Dict[str, Any]
    timestamp: datetime
    recommended_action: str
    false_positive_risk: float

@dataclass
class BehavioralBaseline:
    """Baseline behavior for a node"""
    node_id: str
    message_frequency: float  # messages per minute
    trust_score_variance: float
    response_time_mean: float
    response_time_std: float
    consensus_alignment: float
    data_quality_score: float
    last_updated: datetime

class CorruptionDetector:
    """Detects corrupted and malicious nodes in the Mesh network"""
    
    def __init__(self, trust_ledger: TrustLedger, network_health: NetworkHealthMonitor):
        self.trust_ledger = trust_ledger
        self.network_health = network_health
        # Try to get config, fall back to defaults if not available
        try:
            self.config = get_component_config("mesh_immunity")
            self.trust_divergence_threshold = self.config.get("trust_divergence_threshold", 0.3)
            self.behavioral_anomaly_threshold = self.config.get("behavioral_anomaly_threshold", 2.0)
            self.consensus_manipulation_threshold = self.config.get("consensus_manipulation_threshold", 0.4)
        except Exception:
            # Use default values if config is not available
            self.config = {}
            self.trust_divergence_threshold = 0.3
            self.behavioral_anomaly_threshold = 2.0
            self.consensus_manipulation_threshold = 0.4
        
        # Behavioral baselines for each node
        self.behavioral_baselines: Dict[str, BehavioralBaseline] = {}
        
        # Detection history
        self.detection_history: List[CorruptionDetection] = []
        
        # Statistical models
        self.trust_distribution_model = None
        self.behavioral_model = None
        
        logger.info("Corruption detector initialized")
    
    async def detect_corruption(self, node_id: str) -> Optional[CorruptionDetection]:
        """Detect corruption for a specific node"""
        try:
            # Get current node state
            node_state = await self._get_node_state(node_id)
            if not node_state:
                return None
            
            # Run corruption detection algorithms
            detections = []
            
            # Trust manipulation detection
            trust_detection = await self._detect_trust_manipulation(node_id, node_state)
            if trust_detection:
                detections.append(trust_detection)
            
            # Behavioral anomaly detection
            behavioral_detection = await self._detect_behavioral_anomaly(node_id, node_state)
            if behavioral_detection:
                detections.append(behavioral_detection)
            
            # Consensus manipulation detection
            consensus_detection = await self._detect_consensus_manipulation(node_id, node_state)
            if consensus_detection:
                detections.append(consensus_detection)
            
            # Data poisoning detection
            data_detection = await self._detect_data_poisoning(node_id, node_state)
            if data_detection:
                detections.append(data_detection)
            
            # Combine detections if multiple types found
            if detections:
                return await self._combine_detections(node_id, detections)
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting corruption for node {node_id}: {e}")
            return None
    
    async def _detect_trust_manipulation(self, node_id: str, node_state: Dict[str, Any]) -> Optional[CorruptionDetection]:
        """Detect trust manipulation attempts"""
        try:
            # Get trust score history
            trust_history = await self.trust_ledger.get_trust_history(node_id, hours=24)
            if len(trust_history) < 10:
                return None
            
            # Calculate trust score variance
            trust_scores = [entry.trust_score for entry in trust_history]
            trust_variance = np.var(trust_scores)
            
            # Check for suspicious trust score changes
            trust_changes = np.diff(trust_scores)
            large_changes = np.abs(trust_changes) > self.trust_divergence_threshold
            
            if np.any(large_changes) and trust_variance > 0.1:
                # Calculate confidence based on evidence strength
                confidence = min(0.9, np.sum(large_changes) / len(trust_changes) + trust_variance)
                
                return CorruptionDetection(
                    node_id=node_id,
                    corruption_type=CorruptionType.TRUST_MANIPULATION,
                    corruption_level=CorruptionLevel.MEDIUM if confidence < 0.7 else CorruptionLevel.HIGH,
                    confidence_score=confidence,
                    evidence={
                        "trust_variance": trust_variance,
                        "large_changes_count": np.sum(large_changes),
                        "trust_history": trust_scores[-10:],  # Last 10 scores
                        "change_magnitudes": trust_changes[large_changes].tolist()
                    },
                    timestamp=datetime.now(),
                    recommended_action="Monitor trust score changes and validate recent interactions",
                    false_positive_risk=0.2
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting trust manipulation: {e}")
            return None
    
    async def _detect_behavioral_anomaly(self, node_id: str, node_state: Dict[str, Any]) -> Optional[CorruptionDetection]:
        """Detect behavioral anomalies"""
        try:
            # Get or create behavioral baseline
            baseline = self.behavioral_baselines.get(node_id)
            if not baseline:
                baseline = await self._create_behavioral_baseline(node_id)
                self.behavioral_baselines[node_id] = baseline
            
            # Get current behavioral metrics
            current_metrics = await self._get_current_behavioral_metrics(node_id)
            
            # Calculate anomaly scores
            anomaly_scores = {}
            
            # Message frequency anomaly
            if baseline.message_frequency > 0:
                freq_anomaly = abs(current_metrics["message_frequency"] - baseline.message_frequency) / baseline.message_frequency
                anomaly_scores["frequency"] = freq_anomaly
            
            # Response time anomaly
            if baseline.response_time_mean > 0:
                time_anomaly = abs(current_metrics["response_time"] - baseline.response_time_mean) / baseline.response_time_std
                anomaly_scores["response_time"] = time_anomaly
            
            # Consensus alignment anomaly
            alignment_anomaly = abs(current_metrics["consensus_alignment"] - baseline.consensus_alignment)
            anomaly_scores["consensus"] = alignment_anomaly
            
            # Calculate overall anomaly score
            if anomaly_scores:
                overall_anomaly = np.mean(list(anomaly_scores.values()))
                
                if overall_anomaly > self.behavioral_anomaly_threshold:
                    confidence = min(0.9, overall_anomaly / (self.behavioral_anomaly_threshold * 2))
                    
                    return CorruptionDetection(
                        node_id=node_id,
                        corruption_type=CorruptionType.BEHAVIORAL_ANOMALY,
                        corruption_level=CorruptionLevel.LOW if confidence < 0.6 else CorruptionLevel.MEDIUM,
                        confidence_score=confidence,
                        evidence={
                            "anomaly_scores": anomaly_scores,
                            "overall_anomaly": overall_anomaly,
                            "baseline": {
                                "message_frequency": baseline.message_frequency,
                                "response_time_mean": baseline.response_time_mean,
                                "consensus_alignment": baseline.consensus_alignment
                            },
                            "current_metrics": current_metrics
                        },
                        timestamp=datetime.now(),
                        recommended_action="Investigate behavioral changes and validate node identity",
                        false_positive_risk=0.3
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting behavioral anomaly: {e}")
            return None
    
    async def _detect_consensus_manipulation(self, node_id: str, node_state: Dict[str, Any]) -> Optional[CorruptionDetection]:
        """Detect consensus manipulation attempts"""
        try:
            # Get recent consensus decisions
            consensus_history = await self.trust_ledger.get_consensus_history(node_id, hours=6)
            if len(consensus_history) < 5:
                return None
            
            # Calculate consensus alignment
            alignments = [entry.alignment_score for entry in consensus_history]
            current_alignment = np.mean(alignments)
            
            # Check for sudden drops in consensus alignment
            if len(alignments) >= 3:
                recent_alignment = np.mean(alignments[-3:])
                historical_alignment = np.mean(alignments[:-3])
                
                alignment_drop = historical_alignment - recent_alignment
                
                if alignment_drop > self.consensus_manipulation_threshold:
                    confidence = min(0.9, alignment_drop / (self.consensus_manipulation_threshold * 2))
                    
                    return CorruptionDetection(
                        node_id=node_id,
                        corruption_type=CorruptionType.CONSENSUS_MANIPULATION,
                        corruption_level=CorruptionLevel.HIGH if confidence > 0.7 else CorruptionLevel.MEDIUM,
                        confidence_score=confidence,
                        evidence={
                            "alignment_drop": alignment_drop,
                            "historical_alignment": historical_alignment,
                            "recent_alignment": recent_alignment,
                            "consensus_history": alignments
                        },
                        timestamp=datetime.now(),
                        recommended_action="Investigate consensus decisions and validate node integrity",
                        false_positive_risk=0.15
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting consensus manipulation: {e}")
            return None
    
    async def _detect_data_poisoning(self, node_id: str, node_state: Dict[str, Any]) -> Optional[CorruptionDetection]:
        """Detect data poisoning attempts"""
        try:
            # Get recent data submissions
            data_submissions = await self.trust_ledger.get_data_submissions(node_id, hours=12)
            if len(data_submissions) < 3:
                return None
            
            # Calculate data quality metrics
            quality_scores = []
            for submission in data_submissions:
                # Check for suspicious patterns in data
                quality_score = await self._calculate_data_quality(submission)
                quality_scores.append(quality_score)
            
            # Check for sudden drops in data quality
            if len(quality_scores) >= 3:
                recent_quality = np.mean(quality_scores[-3:])
                historical_quality = np.mean(quality_scores[:-3])
                
                quality_drop = historical_quality - recent_quality
                
                if quality_drop > 0.3:  # 30% drop in quality
                    confidence = min(0.9, quality_drop / 0.6)  # Normalize to 0-1
                    
                    return CorruptionDetection(
                        node_id=node_id,
                        corruption_type=CorruptionType.DATA_POISONING,
                        corruption_level=CorruptionLevel.HIGH if confidence > 0.7 else CorruptionLevel.MEDIUM,
                        confidence_score=confidence,
                        evidence={
                            "quality_drop": quality_drop,
                            "historical_quality": historical_quality,
                            "recent_quality": recent_quality,
                            "quality_scores": quality_scores
                        },
                        timestamp=datetime.now(),
                        recommended_action="Review recent data submissions and validate data sources",
                        false_positive_risk=0.25
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting data poisoning: {e}")
            return None
    
    async def _combine_detections(self, node_id: str, detections: List[CorruptionDetection]) -> CorruptionDetection:
        """Combine multiple corruption detections into a single result"""
        if len(detections) == 1:
            return detections[0]
        
        # Calculate combined confidence and determine overall corruption level
        total_confidence = sum(d.confidence_score for d in detections)
        avg_confidence = total_confidence / len(detections)
        
        # Determine corruption level based on highest individual level and combined evidence
        max_level = max(d.corruption_level for d in detections)
        if len(detections) >= 3 and avg_confidence > 0.7:
            # Multiple detection types with high confidence
            if max_level == CorruptionLevel.MEDIUM:
                max_level = CorruptionLevel.HIGH
            elif max_level == CorruptionLevel.HIGH:
                max_level = CorruptionLevel.CRITICAL
        
        # Combine evidence
        combined_evidence = {
            "detection_count": len(detections),
            "detection_types": [d.corruption_type.value for d in detections],
            "individual_detections": [
                {
                    "type": d.corruption_type.value,
                    "level": d.corruption_level.value,
                    "confidence": d.confidence_score,
                    "evidence": d.evidence
                }
                for d in detections
            ]
        }
        
        # Determine recommended action
        if max_level == CorruptionLevel.CRITICAL:
            recommended_action = "Immediate node isolation and investigation required"
        elif max_level == CorruptionLevel.HIGH:
            recommended_action = "Node quarantine and detailed investigation"
        elif max_level == CorruptionLevel.MEDIUM:
            recommended_action = "Enhanced monitoring and validation"
        else:
            recommended_action = "Continue monitoring for additional evidence"
        
        # Calculate false positive risk (lower with multiple detection types)
        false_positive_risk = max(0.05, min(0.4, 0.4 / len(detections)))
        
        return CorruptionDetection(
            node_id=node_id,
            corruption_type=CorruptionType.BEHAVIORAL_ANOMALY,  # Generic type for combined
            corruption_level=max_level,
            confidence_score=avg_confidence,
            evidence=combined_evidence,
            timestamp=datetime.now(),
            recommended_action=recommended_action,
            false_positive_risk=false_positive_risk
        )
    
    async def _get_node_state(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get current state of a node"""
        try:
            # Get trust information
            trust_info = await self.trust_ledger.get_node_trust_info(node_id)
            
            # Get network health information
            network_info = await self.network_health.get_node_health(node_id)
            
            # Get recent activity
            recent_activity = await self.trust_ledger.get_recent_activity(node_id, hours=1)
            
            return {
                "trust_info": trust_info,
                "network_info": network_info,
                "recent_activity": recent_activity
            }
        except Exception as e:
            logger.error(f"Error getting node state: {e}")
            return None
    
    async def _create_behavioral_baseline(self, node_id: str) -> BehavioralBaseline:
        """Create behavioral baseline for a node"""
        try:
            # Get historical behavioral data
            historical_data = await self.trust_ledger.get_behavioral_history(node_id, days=7)
            
            if not historical_data:
                # Create default baseline
                return BehavioralBaseline(
                    node_id=node_id,
                    message_frequency=1.0,
                    trust_score_variance=0.1,
                    response_time_mean=1000.0,  # milliseconds
                    response_time_std=200.0,
                    consensus_alignment=0.8,
                    data_quality_score=0.8,
                    last_updated=datetime.now()
                )
            
            # Calculate baseline metrics
            message_frequencies = [entry.message_frequency for entry in historical_data]
            response_times = [entry.response_time for entry in historical_data]
            consensus_alignments = [entry.consensus_alignment for entry in historical_data]
            data_qualities = [entry.data_quality for entry in historical_data]
            
            return BehavioralBaseline(
                node_id=node_id,
                message_frequency=np.mean(message_frequencies),
                trust_score_variance=np.var([entry.trust_score for entry in historical_data]),
                response_time_mean=np.mean(response_times),
                response_time_std=np.std(response_times),
                consensus_alignment=np.mean(consensus_alignments),
                data_quality_score=np.mean(data_qualities),
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error creating behavioral baseline: {e}")
            # Return default baseline
            return BehavioralBaseline(
                node_id=node_id,
                message_frequency=1.0,
                trust_score_variance=0.1,
                response_time_mean=1000.0,
                response_time_std=200.0,
                consensus_alignment=0.8,
                data_quality_score=0.8,
                last_updated=datetime.now()
            )
    
    async def _get_current_behavioral_metrics(self, node_id: str) -> Dict[str, Any]:
        """Get current behavioral metrics for a node"""
        try:
            # Get recent activity
            recent_activity = await self.trust_ledger.get_recent_activity(node_id, minutes=30)
            
            # Calculate current metrics
            message_count = len(recent_activity)
            message_frequency = message_count / 0.5  # messages per minute (30 min = 0.5 hours)
            
            response_times = [entry.response_time for entry in recent_activity if entry.response_time]
            response_time = np.mean(response_times) if response_times else 1000.0
            
            consensus_alignments = [entry.consensus_alignment for entry in recent_activity if entry.consensus_alignment]
            consensus_alignment = np.mean(consensus_alignments) if consensus_alignments else 0.8
            
            return {
                "message_frequency": message_frequency,
                "response_time": response_time,
                "consensus_alignment": consensus_alignment
            }
            
        except Exception as e:
            logger.error(f"Error getting current behavioral metrics: {e}")
            return {
                "message_frequency": 1.0,
                "response_time": 1000.0,
                "consensus_alignment": 0.8
            }
    
    async def _calculate_data_quality(self, submission: Any) -> float:
        """Calculate quality score for a data submission"""
        try:
            # This is a simplified quality calculation
            # In practice, this would analyze the actual data content
            
            # Check for common poisoning indicators
            quality_score = 0.8  # Base score
            
            # Reduce score for suspicious patterns
            if hasattr(submission, 'content'):
                content = str(submission.content).lower()
                
                # Check for repetitive patterns
                if len(set(content.split())) < len(content.split()) * 0.3:
                    quality_score -= 0.2
                
                # Check for suspicious keywords
                suspicious_keywords = ['spam', 'fake', 'false', 'manipulate']
                if any(keyword in content for keyword in suspicious_keywords):
                    quality_score -= 0.3
                
                # Check for excessive length (potential flooding)
                if len(content) > 10000:
                    quality_score -= 0.1
            
            return max(0.1, quality_score)
            
        except Exception as e:
            logger.error(f"Error calculating data quality: {e}")
            return 0.5  # Default quality score
    
    async def update_behavioral_baseline(self, node_id: str):
        """Update behavioral baseline for a node"""
        try:
            baseline = await self._create_behavioral_baseline(node_id)
            self.behavioral_baselines[node_id] = baseline
            logger.info(f"Updated behavioral baseline for node {node_id}")
        except Exception as e:
            logger.error(f"Error updating behavioral baseline: {e}")
    
    def get_detection_history(self, node_id: Optional[str] = None) -> List[CorruptionDetection]:
        """Get corruption detection history"""
        if node_id:
            return [d for d in self.detection_history if d.node_id == node_id]
        return self.detection_history
    
    def get_corruption_summary(self) -> Dict[str, Any]:
        """Get summary of corruption detection activity"""
        if not self.detection_history:
            return {"total_detections": 0, "active_threats": 0}
        
        # Count by corruption level
        level_counts = {}
        for detection in self.detection_history:
            level = detection.corruption_level.value
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Count active threats (last 24 hours)
        recent_detections = [
            d for d in self.detection_history 
            if d.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        return {
            "total_detections": len(self.detection_history),
            "active_threats": len(recent_detections),
            "level_breakdown": level_counts,
            "recent_detections": len(recent_detections)
        }
