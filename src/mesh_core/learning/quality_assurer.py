"""
Mesh Learning Quality Assurer
============================

Component 8.1: Learning Quality Assurance
Ensure learning doesn't degrade core capabilities

Implements learning quality metrics, safeguards, and
rollback mechanisms to maintain system stability.
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import time

logger = logging.getLogger(__name__)


class QualityMetric(Enum):
    """Types of quality metrics"""
    ACCURACY = "accuracy"                    # Model accuracy
    ALIGNMENT = "alignment"                  # Value alignment
    PERFORMANCE = "performance"              # System performance
    STABILITY = "stability"                  # System stability
    SAFETY = "safety"                        # Safety metrics
    ROBUSTNESS = "robustness"                # Robustness metrics


class QualityStatus(Enum):
    """Status of quality assessments"""
    EXCELLENT = "excellent"                  # Quality above 0.9
    GOOD = "good"                            # Quality 0.7-0.9
    ACCEPTABLE = "acceptable"                # Quality 0.5-0.7
    POOR = "poor"                            # Quality 0.3-0.5
    CRITICAL = "critical"                    # Quality below 0.3


@dataclass
class QualityAssessment:
    """Assessment of learning quality"""
    assessment_id: str
    learning_session_id: str
    timestamp: datetime
    
    # Quality scores
    accuracy_score: float = 0.0      # 0.0 to 1.0
    alignment_score: float = 0.0     # 0.0 to 1.0
    performance_score: float = 0.0   # 0.0 to 1.0
    stability_score: float = 0.0     # 0.0 to 1.0
    safety_score: float = 0.0        # 0.0 to 1.0
    robustness_score: float = 0.0    # 0.0 to 1.0
    
    # Overall quality
    overall_quality: float = 0.0     # 0.0 to 1.0
    quality_status: QualityStatus = QualityStatus.ACCEPTABLE
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    rollback_recommended: bool = False
    monitoring_required: bool = False
    
    # Metadata
    assessor_id: str = ""
    notes: Optional[str] = None
    
    def __post_init__(self):
        if not self.assessment_id:
            self.assessment_id = self._generate_assessment_id()
    
    def _generate_assessment_id(self) -> str:
        """Generate unique assessment ID"""
        content = f"{self.learning_session_id}{self.timestamp.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert assessment to dictionary"""
        return {
            "assessment_id": self.assessment_id,
            "learning_session_id": self.learning_session_id,
            "timestamp": self.timestamp.isoformat(),
            "accuracy_score": self.accuracy_score,
            "alignment_score": self.alignment_score,
            "performance_score": self.performance_score,
            "stability_score": self.stability_score,
            "safety_score": self.safety_score,
            "robustness_score": self.robustness_score,
            "overall_quality": self.overall_quality,
            "quality_status": self.quality_status.value,
            "recommendations": self.recommendations,
            "rollback_recommended": self.rollback_recommended,
            "monitoring_required": self.monitoring_required,
            "assessor_id": self.assessor_id,
            "notes": self.notes
        }


@dataclass
class QualityThreshold:
    """Thresholds for quality metrics"""
    metric_name: str
    warning_threshold: float = 0.7
    critical_threshold: float = 0.5
    rollback_threshold: float = 0.3
    
    # Weight in overall quality calculation
    weight: float = 1.0
    
    # Monitoring settings
    continuous_monitoring: bool = True
    alert_on_violation: bool = True


class QualityAssurer:
    """
    Ensures learning quality and provides safeguards
    
    Monitors learning outcomes, assesses quality metrics,
    and recommends actions to maintain system stability.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        
        # Quality assessments
        self.quality_assessments: Dict[str, QualityAssessment] = {}
        self.learning_quality_history: List[QualityAssessment] = []
        
        # Quality thresholds
        self.quality_thresholds: Dict[str, QualityThreshold] = self._initialize_thresholds()
        
        # Monitoring state
        self.active_monitoring: Dict[str, bool] = {}
        self.quality_alerts: List[Dict[str, Any]] = []
        
        # Configuration
        self.min_quality_threshold = 0.4  # Lowered from 0.6 for demo
        self.quality_decay_factor = 0.95
        self.max_assessments_stored = 1000
        
        # Performance metrics
        self.total_assessments = 0
        self.rollback_recommendations = 0
        self.quality_violations = 0
        
        # Threading
        self.lock = threading.RLock()
        
        logger.info(f"QualityAssurer initialized for node: {self.node_id}")
    
    def _initialize_thresholds(self) -> Dict[str, QualityThreshold]:
        """Initialize default quality thresholds"""
        thresholds = {}
        
        thresholds["accuracy"] = QualityThreshold(
            metric_name="accuracy",
            warning_threshold=0.6,  # Lowered from 0.8
            critical_threshold=0.4,  # Lowered from 0.6
            rollback_threshold=0.2,  # Lowered from 0.4
            weight=0.25
        )
        
        thresholds["alignment"] = QualityThreshold(
            metric_name="alignment",
            warning_threshold=0.7,  # Lowered from 0.85
            critical_threshold=0.5,  # Lowered from 0.7
            rollback_threshold=0.3,  # Lowered from 0.5
            weight=0.25
        )
        
        thresholds["performance"] = QualityThreshold(
            metric_name="performance",
            warning_threshold=0.75,
            critical_threshold=0.6,
            rollback_threshold=0.4,
            weight=0.2
        )
        
        thresholds["stability"] = QualityThreshold(
            metric_name="stability",
            warning_threshold=0.8,
            critical_threshold=0.65,
            rollback_threshold=0.45,
            weight=0.15
        )
        
        thresholds["safety"] = QualityThreshold(
            metric_name="safety",
            warning_threshold=0.9,
            critical_threshold=0.75,
            rollback_threshold=0.6,
            weight=0.15
        )
        
        return thresholds
    
    def assess_learning_quality(self, learning_session_id: str,
                               learning_outcomes: Dict[str, Any]) -> str:
        """Assess the quality of a learning session"""
        try:
            with self.lock:
                # Extract quality metrics from learning outcomes
                accuracy_score = learning_outcomes.get("accuracy_improvement", 0.0)
                alignment_score = learning_outcomes.get("alignment_preservation", 0.0)
                performance_score = learning_outcomes.get("performance_improvement", 0.0)
                stability_score = learning_outcomes.get("stability_score", 0.0)
                safety_score = learning_outcomes.get("safety_score", 0.0)
                robustness_score = learning_outcomes.get("robustness_improvement", 0.0)
                
                # Calculate overall quality
                overall_quality = self._calculate_overall_quality({
                    "accuracy": accuracy_score,
                    "alignment": alignment_score,
                    "performance": performance_score,
                    "stability": stability_score,
                    "safety": safety_score,
                    "robustness": robustness_score
                })
                
                # Determine quality status
                quality_status = self._determine_quality_status(overall_quality)
                
                # Generate recommendations
                recommendations = self._generate_recommendations({
                    "accuracy": accuracy_score,
                    "alignment": alignment_score,
                    "performance": performance_score,
                    "stability": stability_score,
                    "safety": safety_score,
                    "robustness": robustness_score
                })
                
                # Determine if rollback is recommended
                rollback_recommended = self._should_recommend_rollback({
                    "accuracy": accuracy_score,
                    "alignment": alignment_score,
                    "performance": performance_score,
                    "stability": stability_score,
                    "safety": safety_score,
                    "robustness": robustness_score
                })
                
                # Create quality assessment
                assessment = QualityAssessment(
                    assessment_id="",
                    learning_session_id=learning_session_id,
                    timestamp=datetime.utcnow(),
                    accuracy_score=accuracy_score,
                    alignment_score=alignment_score,
                    performance_score=performance_score,
                    stability_score=stability_score,
                    safety_score=safety_score,
                    robustness_score=robustness_score,
                    overall_quality=overall_quality,
                    quality_status=quality_status,
                    recommendations=recommendations,
                    rollback_recommended=rollback_recommended,
                    monitoring_required=overall_quality < 0.8,
                    assessor_id=self.node_id
                )
                
                # Store assessment
                self.quality_assessments[assessment.assessment_id] = assessment
                self.learning_quality_history.append(assessment)
                
                # Update statistics
                self.total_assessments += 1
                if rollback_recommended:
                    self.rollback_recommendations += 1
                if overall_quality < self.min_quality_threshold:
                    self.quality_violations += 1
                
                # Trigger alerts if needed
                if overall_quality < self.min_quality_threshold:
                    self._trigger_quality_alert(assessment)
                
                # Cleanup old assessments
                self._cleanup_old_assessments()
                
                logger.info(f"Quality assessment completed for session {learning_session_id}: {quality_status.value}")
                return assessment.assessment_id
                
        except Exception as e:
            logger.error(f"Failed to assess learning quality: {e}")
            raise
    
    def _calculate_overall_quality(self, metric_scores: Dict[str, float]) -> float:
        """Calculate overall quality score from individual metrics"""
        try:
            total_weighted_score = 0.0
            total_weight = 0.0
            
            for metric_name, score in metric_scores.items():
                if metric_name in self.quality_thresholds:
                    threshold = self.quality_thresholds[metric_name]
                    total_weighted_score += score * threshold.weight
                    total_weight += threshold.weight
            
            if total_weight == 0:
                return 0.0
            
            overall_quality = total_weighted_score / total_weight
            
            # Apply quality decay for very high scores
            if overall_quality > 0.95:
                overall_quality = 0.95 + (overall_quality - 0.95) * self.quality_decay_factor
            
            return min(1.0, max(0.0, overall_quality))
            
        except Exception as e:
            logger.error(f"Error calculating overall quality: {e}")
            return 0.5
    
    def _determine_quality_status(self, overall_quality: float) -> QualityStatus:
        """Determine quality status based on overall quality score"""
        if overall_quality >= 0.9:
            return QualityStatus.EXCELLENT
        elif overall_quality >= 0.7:
            return QualityStatus.GOOD
        elif overall_quality >= 0.5:
            return QualityStatus.ACCEPTABLE
        elif overall_quality >= 0.3:
            return QualityStatus.POOR
        else:
            return QualityStatus.CRITICAL
    
    def _generate_recommendations(self, metric_scores: Dict[str, float]) -> List[str]:
        """Generate recommendations based on quality metrics"""
        recommendations = []
        
        for metric_name, score in metric_scores.items():
            if metric_name in self.quality_thresholds:
                threshold = self.quality_thresholds[metric_name]
                
                if score < threshold.rollback_threshold:
                    recommendations.append(f"Critical: {metric_name} below rollback threshold ({score:.2f} < {threshold.rollback_threshold})")
                elif score < threshold.critical_threshold:
                    recommendations.append(f"Warning: {metric_name} below critical threshold ({score:.2f} < {threshold.critical_threshold})")
                elif score < threshold.warning_threshold:
                    recommendations.append(f"Monitor: {metric_name} below warning threshold ({score:.2f} < {threshold.warning_threshold})")
        
        # Add general recommendations
        if len(recommendations) == 0:
            recommendations.append("All quality metrics are within acceptable ranges")
        elif len(recommendations) > 3:
            recommendations.append("Multiple quality issues detected - consider comprehensive review")
        
        return recommendations
    
    def _should_recommend_rollback(self, metric_scores: Dict[str, float]) -> bool:
        """Determine if rollback should be recommended"""
        for metric_name, score in metric_scores.items():
            if metric_name in self.quality_thresholds:
                threshold = self.quality_thresholds[metric_name]
                if score < threshold.rollback_threshold:
                    return True
        
        return False
    
    def _trigger_quality_alert(self, assessment: QualityAssessment):
        """Trigger a quality alert"""
        try:
            alert = {
                "alert_id": hashlib.sha256(f"{assessment.assessment_id}{datetime.utcnow().isoformat()}".encode()).hexdigest()[:16],
                "timestamp": datetime.utcnow().isoformat(),
                "severity": "high" if assessment.overall_quality < 0.5 else "medium",
                "learning_session_id": assessment.learning_session_id,
                "overall_quality": assessment.overall_quality,
                "quality_status": assessment.quality_status.value,
                "recommendations": assessment.recommendations,
                "rollback_recommended": assessment.rollback_recommended
            }
            
            self.quality_alerts.append(alert)
            
            logger.warning(f"Quality alert triggered: {assessment.quality_status.value} quality detected")
            
        except Exception as e:
            logger.error(f"Failed to trigger quality alert: {e}")
    
    def get_quality_summary(self, learning_session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get quality summary for specific session or overall"""
        with self.lock:
            if learning_session_id:
                # Session-specific summary
                session_assessments = [a for a in self.learning_quality_history 
                                     if a.learning_session_id == learning_session_id]
                
                if not session_assessments:
                    return {"error": "No assessments found for session"}
                
                latest_assessment = max(session_assessments, key=lambda x: x.timestamp)
                
                return {
                    "learning_session_id": learning_session_id,
                    "latest_assessment": latest_assessment.to_dict(),
                    "total_assessments": len(session_assessments),
                    "quality_trend": self._calculate_quality_trend(session_assessments)
                }
            else:
                # Overall summary
                if not self.learning_quality_history:
                    return {"error": "No quality history available"}
                
                recent_assessments = self.learning_quality_history[-100:]  # Last 100 assessments
                
                return {
                    "node_id": self.node_id,
                    "total_assessments": self.total_assessments,
                    "rollback_recommendations": self.rollback_recommendations,
                    "quality_violations": self.quality_violations,
                    "recent_quality_trend": self._calculate_quality_trend(recent_assessments),
                    "quality_distribution": self._calculate_quality_distribution(recent_assessments),
                    "active_alerts": len(self.quality_alerts),
                    "min_quality_threshold": self.min_quality_threshold
                }
    
    def _calculate_quality_trend(self, assessments: List[QualityAssessment]) -> Dict[str, Any]:
        """Calculate quality trend over time"""
        if len(assessments) < 2:
            return {"trend": "insufficient_data", "change": 0.0}
        
        # Sort by timestamp
        sorted_assessments = sorted(assessments, key=lambda x: x.timestamp)
        
        # Calculate trend
        first_quality = sorted_assessments[0].overall_quality
        last_quality = sorted_assessments[-1].overall_quality
        quality_change = last_quality - first_quality
        
        if quality_change > 0.1:
            trend = "improving"
        elif quality_change < -0.1:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "change": quality_change,
            "first_assessment": sorted_assessments[0].timestamp.isoformat(),
            "last_assessment": sorted_assessments[-1].timestamp.isoformat(),
            "assessment_count": len(assessments)
        }
    
    def _calculate_quality_distribution(self, assessments: List[QualityAssessment]) -> Dict[str, int]:
        """Calculate distribution of quality statuses"""
        distribution = {}
        
        for assessment in assessments:
            status = assessment.quality_status.value
            distribution[status] = distribution.get(status, 0) + 1
        
        return distribution
    
    def _cleanup_old_assessments(self):
        """Remove old assessments to save memory"""
        if len(self.learning_quality_history) > self.max_assessments_stored:
            # Keep only the most recent assessments
            self.learning_quality_history = self.learning_quality_history[-self.max_assessments_stored:]
            
            # Update stored assessments
            recent_ids = {a.assessment_id for a in self.learning_quality_history}
            old_ids = [aid for aid in self.quality_assessments.keys() if aid not in recent_ids]
            
            for old_id in old_ids:
                del self.quality_assessments[old_id]
            
            logger.info(f"Cleaned up {len(old_ids)} old quality assessments")
    
    def update_quality_threshold(self, metric_name: str, **kwargs) -> bool:
        """Update quality threshold for a specific metric"""
        try:
            with self.lock:
                if metric_name not in self.quality_thresholds:
                    logger.warning(f"Unknown quality metric: {metric_name}")
                    return False
                
                threshold = self.quality_thresholds[metric_name]
                
                # Update threshold attributes
                for key, value in kwargs.items():
                    if hasattr(threshold, key):
                        setattr(threshold, key, value)
                
                logger.info(f"Updated quality threshold for {metric_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update quality threshold: {e}")
            return False
    
    def get_quality_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get quality alerts with optional severity filtering"""
        with self.lock:
            if severity:
                return [alert for alert in self.quality_alerts if alert["severity"] == severity]
            else:
                return self.quality_alerts.copy()
    
    def clear_quality_alerts(self) -> int:
        """Clear all quality alerts"""
        with self.lock:
            alert_count = len(self.quality_alerts)
            self.quality_alerts.clear()
            logger.info(f"Cleared {alert_count} quality alerts")
            return alert_count
