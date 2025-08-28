"""
Mesh Health Monitor
==================

Component 10.1: Mesh Degeneration Watchdogs
Monitor civilizational health

Implements comprehensive health monitoring, system vitality assessment,
and long-term health trend analysis.
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


class HealthDimension(Enum):
    """Dimensions of civilizational health"""
    SOCIAL = "social"                    # Social health
    ECONOMIC = "economic"                # Economic health
    ENVIRONMENTAL = "environmental"      # Environmental health
    TECHNOLOGICAL = "technological"      # Technological health
    CULTURAL = "cultural"                # Cultural health
    POLITICAL = "political"              # Political health
    ETHICAL = "ethical"                  # Ethical health
    RESILIENCE = "resilience"            # System resilience


class HealthStatus(Enum):
    """Health status levels"""
    EXCELLENT = "excellent"              # Optimal health
    GOOD = "good"                        # Good health
    FAIR = "fair"                        # Fair health, some concerns
    POOR = "poor"                        # Poor health, significant concerns
    CRITICAL = "critical"                # Critical health, immediate action needed


@dataclass
class HealthMetric:
    """A health metric measurement"""
    metric_id: str
    dimension: HealthDimension
    timestamp: datetime
    
    # Metric data
    metric_name: str
    metric_value: float  # 0.0 to 1.0, higher = better health
    baseline_value: float  # Expected value for this metric
    deviation: float  # Difference from baseline
    
    # Context
    node_id: str
    measurement_context: Dict[str, Any]
    
    # Analysis
    health_status: HealthStatus
    confidence_score: float  # 0.0 to 1.0
    contributing_factors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.metric_id:
            self.metric_id = self._generate_metric_id()
    
    def _generate_metric_id(self) -> str:
        """Generate unique metric ID"""
        content = f"{self.dimension.value}{self.metric_name}{self.node_id}{self.timestamp.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert health metric to dictionary"""
        return {
            "metric_id": self.metric_id,
            "dimension": self.dimension.value,
            "timestamp": self.timestamp.isoformat(),
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "baseline_value": self.baseline_value,
            "deviation": self.deviation,
            "node_id": self.node_id,
            "measurement_context": self.measurement_context,
            "health_status": self.health_status.value,
            "confidence_score": self.confidence_score,
            "contributing_factors": self.contributing_factors
        }


@dataclass
class HealthAssessment:
    """Comprehensive health assessment"""
    assessment_id: str
    timestamp: datetime
    
    # Overall health
    overall_health_score: float  # 0.0 to 1.0
    overall_health_status: HealthStatus
    
    # Dimension scores
    dimension_scores: Dict[HealthDimension, float] = field(default_factory=dict)
    dimension_statuses: Dict[HealthDimension, HealthStatus] = field(default_factory=dict)
    
    # Trend analysis
    health_trend: str = "stable"  # improving, declining, stable, fluctuating
    trend_strength: float = 0.0  # 0.0 to 1.0
    
    # Recommendations
    priority_actions: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.assessment_id:
            self.assessment_id = self._generate_assessment_id()
    
    def _generate_assessment_id(self) -> str:
        """Generate unique assessment ID"""
        content = f"health_assessment{self.timestamp.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert health assessment to dictionary"""
        return {
            "assessment_id": self.assessment_id,
            "timestamp": self.timestamp.isoformat(),
            "overall_health_score": self.overall_health_score,
            "overall_health_status": self.overall_health_status.value,
            "dimension_scores": {dim.value: score for dim, score in self.dimension_scores.items()},
            "dimension_statuses": {dim.value: status.value for dim, status in self.dimension_statuses.items()},
            "health_trend": self.health_trend,
            "trend_strength": self.trend_strength,
            "priority_actions": self.priority_actions,
            "risk_factors": self.risk_factors
        }


class HealthMonitor:
    """
    Monitors civilizational health
    
    Tracks health across multiple dimensions, analyzes trends,
    and provides comprehensive health assessments.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        
        # Storage
        self.health_metrics: Dict[str, HealthMetric] = {}
        self.health_assessments: Dict[str, HealthAssessment] = {}
        
        # Configuration
        self.health_thresholds = {
            HealthDimension.SOCIAL: 0.6,
            HealthDimension.ECONOMIC: 0.5,
            HealthDimension.ENVIRONMENTAL: 0.7,
            HealthDimension.TECHNOLOGICAL: 0.6,
            HealthDimension.CULTURAL: 0.6,
            HealthDimension.POLITICAL: 0.5,
            HealthDimension.ETHICAL: 0.7,
            HealthDimension.RESILIENCE: 0.6
        }
        
        # Baseline values (expected health for healthy civilization)
        self.baseline_values = {
            HealthDimension.SOCIAL: 0.8,      # High social cohesion
            HealthDimension.ECONOMIC: 0.7,    # Stable economy
            HealthDimension.ENVIRONMENTAL: 0.8,  # Good environmental health
            HealthDimension.TECHNOLOGICAL: 0.7,  # Good technological health
            HealthDimension.CULTURAL: 0.8,    # Rich culture
            HealthDimension.POLITICAL: 0.6,   # Stable politics
            HealthDimension.ETHICAL: 0.8,     # Strong ethics
            HealthDimension.RESILIENCE: 0.7   # Good resilience
        }
        
        # Performance metrics
        self.total_metrics = 0
        self.assessments_generated = 0
        self.alerts_generated = 0
        
        logger.info(f"HealthMonitor initialized for node: {self.node_id}")
    
    def measure_health(self, dimension: HealthDimension, metric_name: str,
                      measurement_data: Dict[str, Any],
                      context: Dict[str, Any] = None) -> HealthMetric:
        """Measure health for a specific dimension and metric"""
        try:
            # Calculate health value based on dimension and metric
            health_value = self._calculate_health_value(dimension, metric_name, measurement_data)
            
            # Get baseline and calculate deviation
            baseline_value = self.baseline_values[dimension]
            deviation = health_value - baseline_value
            
            # Determine health status
            health_status = self._determine_health_status(health_value, dimension)
            
            # Analyze contributing factors
            contributing_factors = self._identify_contributing_factors(dimension, metric_name, measurement_data)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(measurement_data, context)
            
            # Create metric
            metric = HealthMetric(
                metric_id="",
                dimension=dimension,
                timestamp=datetime.utcnow(),
                metric_name=metric_name,
                metric_value=health_value,
                baseline_value=baseline_value,
                deviation=deviation,
                node_id=self.node_id,
                measurement_context=context or {},
                health_status=health_status,
                confidence_score=confidence_score,
                contributing_factors=contributing_factors
            )
            
            # Store metric
            self.health_metrics[metric.metric_id] = metric
            
            # Update statistics
            self.total_metrics += 1
            
            # Check for alerts
            if health_status in [HealthStatus.POOR, HealthStatus.CRITICAL]:
                self._generate_health_alert(metric)
            
            logger.info(f"Measured {dimension.value} health for {metric_name}: {health_value:.3f} (status: {health_status.value})")
            return metric
            
        except Exception as e:
            logger.error(f"Failed to measure health: {e}")
            raise
    
    def _calculate_health_value(self, dimension: HealthDimension, metric_name: str, data: Dict[str, Any]) -> float:
        """Calculate health value based on dimension and metric"""
        if dimension == HealthDimension.SOCIAL:
            return self._calculate_social_health(metric_name, data)
        elif dimension == HealthDimension.ECONOMIC:
            return self._calculate_economic_health(metric_name, data)
        elif dimension == HealthDimension.ENVIRONMENTAL:
            return self._calculate_environmental_health(metric_name, data)
        elif dimension == HealthDimension.TECHNOLOGICAL:
            return self._calculate_technological_health(metric_name, data)
        elif dimension == HealthDimension.CULTURAL:
            return self._calculate_cultural_health(metric_name, data)
        elif dimension == HealthDimension.POLITICAL:
            return self._calculate_political_health(metric_name, data)
        elif dimension == HealthDimension.ETHICAL:
            return self._calculate_ethical_health(metric_name, data)
        elif dimension == HealthDimension.RESILIENCE:
            return self._calculate_resilience_health(metric_name, data)
        else:
            return 0.5  # Default health value
    
    def _calculate_social_health(self, metric_name: str, data: Dict[str, Any]) -> float:
        """Calculate social health metrics"""
        if metric_name == "social_cohesion":
            cohesion_score = data.get("cohesion_score", 0.5)
            return min(1.0, max(0.0, cohesion_score))
        elif metric_name == "community_trust":
            trust_score = data.get("trust_score", 0.5)
            return min(1.0, max(0.0, trust_score))
        elif metric_name == "social_mobility":
            mobility_score = data.get("mobility_score", 0.5)
            return min(1.0, max(0.0, mobility_score))
        else:
            return 0.5
    
    def _calculate_economic_health(self, metric_name: str, data: Dict[str, Any]) -> float:
        """Calculate economic health metrics"""
        if metric_name == "economic_stability":
            stability_score = data.get("stability_score", 0.5)
            return min(1.0, max(0.0, stability_score))
        elif metric_name == "wealth_distribution":
            gini_coefficient = data.get("gini_coefficient", 0.5)
            # Lower Gini = better health
            return max(0.0, 1.0 - gini_coefficient)
        elif metric_name == "economic_growth":
            growth_rate = data.get("growth_rate", 0.0)
            # Moderate growth is healthiest
            if 0.02 <= growth_rate <= 0.05:
                return 1.0
            elif growth_rate < 0:
                return 0.0
            else:
                return max(0.0, 1.0 - abs(growth_rate - 0.035) / 0.1)
        else:
            return 0.5
    
    def _calculate_environmental_health(self, metric_name: str, data: Dict[str, Any]) -> float:
        """Calculate environmental health metrics"""
        if metric_name == "air_quality":
            air_quality = data.get("air_quality_index", 50)
            # Lower AQI = better health
            if air_quality <= 50:
                return 1.0
            elif air_quality <= 100:
                return 0.8
            elif air_quality <= 150:
                return 0.6
            else:
                return max(0.0, 1.0 - (air_quality - 150) / 100)
        elif metric_name == "water_quality":
            water_quality = data.get("water_quality_score", 0.5)
            return min(1.0, max(0.0, water_quality))
        elif metric_name == "biodiversity":
            biodiversity_index = data.get("biodiversity_index", 0.5)
            return min(1.0, max(0.0, biodiversity_index))
        else:
            return 0.5
    
    def _calculate_technological_health(self, metric_name: str, data: Dict[str, Any]) -> float:
        """Calculate technological health metrics"""
        if metric_name == "innovation_rate":
            innovation_score = data.get("innovation_score", 0.5)
            return min(1.0, max(0.0, innovation_score))
        elif metric_name == "technology_access":
            access_score = data.get("access_score", 0.5)
            return min(1.0, max(0.0, access_score))
        elif metric_name == "digital_literacy":
            literacy_score = data.get("literacy_score", 0.5)
            return min(1.0, max(0.0, literacy_score))
        else:
            return 0.5
    
    def _calculate_cultural_health(self, metric_name: str, data: Dict[str, Any]) -> float:
        """Calculate cultural health metrics"""
        if metric_name == "cultural_diversity":
            diversity_score = data.get("diversity_score", 0.5)
            return min(1.0, max(0.0, diversity_score))
        elif metric_name == "cultural_preservation":
            preservation_score = data.get("preservation_score", 0.5)
            return min(1.0, max(0.0, preservation_score))
        elif metric_name == "artistic_expression":
            expression_score = data.get("expression_score", 0.5)
            return min(1.0, max(0.0, expression_score))
        else:
            return 0.5
    
    def _calculate_political_health(self, metric_name: str, data: Dict[str, Any]) -> float:
        """Calculate political health metrics"""
        if metric_name == "democratic_health":
            democracy_score = data.get("democracy_score", 0.5)
            return min(1.0, max(0.0, democracy_score))
        elif metric_name == "political_stability":
            stability_score = data.get("stability_score", 0.5)
            return min(1.0, max(0.0, stability_score))
        elif metric_name == "corruption_level":
            corruption_score = data.get("corruption_score", 0.5)
            # Lower corruption = better health
            return max(0.0, 1.0 - corruption_score)
        else:
            return 0.5
    
    def _calculate_ethical_health(self, metric_name: str, data: Dict[str, Any]) -> float:
        """Calculate ethical health metrics"""
        if metric_name == "moral_coherence":
            coherence_score = data.get("coherence_score", 0.5)
            return min(1.0, max(0.0, coherence_score))
        elif metric_name == "ethical_education":
            education_score = data.get("education_score", 0.5)
            return min(1.0, max(0.0, education_score))
        elif metric_name == "justice_fairness":
            justice_score = data.get("justice_score", 0.5)
            return min(1.0, max(0.0, justice_score))
        else:
            return 0.5
    
    def _calculate_resilience_health(self, metric_name: str, data: Dict[str, Any]) -> float:
        """Calculate resilience health metrics"""
        if metric_name == "system_resilience":
            resilience_score = data.get("resilience_score", 0.5)
            return min(1.0, max(0.0, resilience_score))
        elif metric_name == "adaptability":
            adaptability_score = data.get("adaptability_score", 0.5)
            return min(1.0, max(0.0, adaptability_score))
        elif metric_name == "recovery_capacity":
            recovery_score = data.get("recovery_score", 0.5)
            return min(1.0, max(0.0, recovery_score))
        else:
            return 0.5
    
    def _determine_health_status(self, health_value: float, dimension: HealthDimension) -> HealthStatus:
        """Determine health status based on value and dimension"""
        threshold = self.health_thresholds[dimension]
        
        if health_value >= threshold * 1.2:
            return HealthStatus.EXCELLENT
        elif health_value >= threshold:
            return HealthStatus.GOOD
        elif health_value >= threshold * 0.8:
            return HealthStatus.FAIR
        elif health_value >= threshold * 0.6:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL
    
    def _identify_contributing_factors(self, dimension: HealthDimension, metric_name: str, data: Dict[str, Any]) -> List[str]:
        """Identify factors contributing to health status"""
        factors = []
        
        # Generic factors based on data quality
        if not data:
            factors.append("Insufficient data")
        
        if "confidence" in data and data["confidence"] < 0.5:
            factors.append("Low measurement confidence")
        
        # Dimension-specific factors
        if dimension == HealthDimension.SOCIAL:
            if "sample_size" in data and data["sample_size"] < 100:
                factors.append("Small sample size")
        
        elif dimension == HealthDimension.ECONOMIC:
            if "data_freshness" in data:
                days_old = data["data_freshness"]
                if days_old > 30:
                    factors.append("Outdated economic data")
        
        elif dimension == HealthDimension.ENVIRONMENTAL:
            if "measurement_accuracy" in data and data["measurement_accuracy"] < 0.8:
                factors.append("Low measurement accuracy")
        
        return factors
    
    def _calculate_confidence(self, data: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate confidence score for the measurement"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on data quality
        if data and len(data) > 0:
            confidence += 0.2
        
        # Increase confidence based on context
        if context and len(context) > 0:
            confidence += 0.1
        
        # Increase confidence for recent data
        if "timestamp" in data:
            confidence += 0.1
        
        # Increase confidence for comprehensive data
        if "completeness_score" in data:
            completeness = data["completeness_score"]
            confidence += completeness * 0.1
        
        return min(1.0, confidence)
    
    def _generate_health_alert(self, metric: HealthMetric):
        """Generate alert for poor health metric"""
        alert_message = f"Poor {metric.dimension.value} health detected for {metric.metric_name}: {metric.metric_value:.3f}"
        
        if metric.contributing_factors:
            alert_message += f" Contributing factors: {', '.join(metric.contributing_factors)}"
        
        logger.warning(f"HEALTH ALERT: {alert_message}")
        
        # Store alert
        self.alerts_generated += 1
        
        # Could send to alert system, notification system, etc.
    
    def generate_health_assessment(self) -> HealthAssessment:
        """Generate comprehensive health assessment"""
        try:
            # Calculate dimension scores
            dimension_scores = {}
            dimension_statuses = {}
            
            for dimension in HealthDimension:
                metrics = [m for m in self.health_metrics.values() if m.dimension == dimension]
                if metrics:
                    # Calculate average score for dimension
                    avg_score = statistics.mean([m.metric_value for m in metrics])
                    dimension_scores[dimension] = avg_score
                    dimension_statuses[dimension] = self._determine_health_status(avg_score, dimension)
                else:
                    # No metrics for this dimension
                    dimension_scores[dimension] = 0.5
                    dimension_statuses[dimension] = HealthStatus.FAIR
            
            # Calculate overall health score
            overall_health_score = statistics.mean(list(dimension_scores.values()))
            overall_health_status = self._determine_overall_health_status(overall_health_score)
            
            # Analyze health trend
            health_trend, trend_strength = self._analyze_health_trend()
            
            # Generate recommendations
            priority_actions = self._generate_priority_actions(dimension_scores, dimension_statuses)
            risk_factors = self._identify_risk_factors(dimension_scores, dimension_statuses)
            
            # Create assessment
            assessment = HealthAssessment(
                assessment_id="",
                timestamp=datetime.utcnow(),
                overall_health_score=overall_health_score,
                overall_health_status=overall_health_status,
                dimension_scores=dimension_scores,
                dimension_statuses=dimension_statuses,
                health_trend=health_trend,
                trend_strength=trend_strength,
                priority_actions=priority_actions,
                risk_factors=risk_factors
            )
            
            # Store assessment
            self.health_assessments[assessment.assessment_id] = assessment
            self.assessments_generated += 1
            
            logger.info(f"Generated health assessment: {overall_health_status.value} (score: {overall_health_score:.3f})")
            return assessment
            
        except Exception as e:
            logger.error(f"Failed to generate health assessment: {e}")
            raise
    
    def _determine_overall_health_status(self, overall_score: float) -> HealthStatus:
        """Determine overall health status"""
        if overall_score >= 0.8:
            return HealthStatus.EXCELLENT
        elif overall_score >= 0.6:
            return HealthStatus.GOOD
        elif overall_score >= 0.4:
            return HealthStatus.FAIR
        elif overall_score >= 0.2:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL
    
    def _analyze_health_trend(self) -> Tuple[str, float]:
        """Analyze health trend over time"""
        if len(self.health_assessments) < 2:
            return "stable", 0.0
        
        # Get recent assessments
        recent_assessments = sorted(
            self.health_assessments.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )[:5]
        
        if len(recent_assessments) < 2:
            return "stable", 0.0
        
        # Calculate trend
        scores = [a.overall_health_score for a in recent_assessments]
        times = [(a.timestamp - recent_assessments[0].timestamp).total_seconds() for a in recent_assessments]
        
        # Simple trend calculation
        if len(scores) > 1:
            slope = (scores[-1] - scores[0]) / max(times[-1] - times[0], 1)
            
            if slope > 0.001:
                trend = "improving"
                strength = min(1.0, abs(slope) * 1000)
            elif slope < -0.001:
                trend = "declining"
                strength = min(1.0, abs(slope) * 1000)
            else:
                trend = "stable"
                strength = 0.0
        else:
            trend = "stable"
            strength = 0.0
        
        return trend, strength
    
    def _generate_priority_actions(self, dimension_scores: Dict[HealthDimension, float],
                                 dimension_statuses: Dict[HealthDimension, HealthStatus]) -> List[str]:
        """Generate priority actions based on health assessment"""
        actions = []
        
        # Identify critical dimensions
        critical_dimensions = [dim for dim, status in dimension_statuses.items() if status == HealthStatus.CRITICAL]
        poor_dimensions = [dim for dim, status in dimension_statuses.items() if status == HealthStatus.POOR]
        
        # Generate actions for critical dimensions
        for dimension in critical_dimensions:
            if dimension == HealthDimension.SOCIAL:
                actions.append("Immediate social intervention and community support")
            elif dimension == HealthDimension.ECONOMIC:
                actions.append("Emergency economic stabilization measures")
            elif dimension == HealthDimension.ENVIRONMENTAL:
                actions.append("Immediate environmental protection and restoration")
            elif dimension == HealthDimension.TECHNOLOGICAL:
                actions.append("Technology infrastructure emergency response")
            elif dimension == HealthDimension.CULTURAL:
                actions.append("Cultural preservation emergency measures")
            elif dimension == HealthDimension.POLITICAL:
                actions.append("Political crisis intervention and mediation")
            elif dimension == HealthDimension.ETHICAL:
                actions.append("Ethical framework emergency review and restoration")
            elif dimension == HealthDimension.RESILIENCE:
                actions.append("System resilience emergency strengthening")
        
        # Generate actions for poor dimensions
        for dimension in poor_dimensions[:3]:  # Limit to top 3
            if dimension == HealthDimension.SOCIAL:
                actions.append("Social cohesion improvement programs")
            elif dimension == HealthDimension.ECONOMIC:
                actions.append("Economic reform and stabilization programs")
            elif dimension == HealthDimension.ENVIRONMENTAL:
                actions.append("Environmental protection and sustainability programs")
            elif dimension == HealthDimension.TECHNOLOGICAL:
                actions.append("Technology infrastructure improvement programs")
            elif dimension == HealthDimension.CULTURAL:
                actions.append("Cultural preservation and enrichment programs")
            elif dimension == HealthDimension.POLITICAL:
                actions.append("Political reform and stability programs")
            elif dimension == HealthDimension.ETHICAL:
                actions.append("Ethical framework strengthening programs")
            elif dimension == HealthDimension.RESILIENCE:
                actions.append("System resilience improvement programs")
        
        return actions[:5]  # Limit to top 5 actions
    
    def _identify_risk_factors(self, dimension_scores: Dict[HealthDimension, float],
                              dimension_statuses: Dict[HealthDimension, HealthStatus]) -> List[str]:
        """Identify risk factors based on health assessment"""
        risk_factors = []
        
        # Identify low-scoring dimensions
        low_score_dimensions = [dim for dim, score in dimension_scores.items() if score < 0.5]
        
        for dimension in low_score_dimensions:
            if dimension == HealthDimension.SOCIAL:
                risk_factors.append("Social fragmentation and isolation")
            elif dimension == HealthDimension.ECONOMIC:
                risk_factors.append("Economic instability and inequality")
            elif dimension == HealthDimension.ENVIRONMENTAL:
                risk_factors.append("Environmental degradation and resource depletion")
            elif dimension == HealthDimension.TECHNOLOGICAL:
                risk_factors.append("Technology infrastructure vulnerabilities")
            elif dimension == HealthDimension.CULTURAL:
                risk_factors.append("Cultural erosion and homogenization")
            elif dimension == HealthDimension.POLITICAL:
                risk_factors.append("Political instability and polarization")
            elif dimension == HealthDimension.ETHICAL:
                risk_factors.append("Ethical framework weakening")
            elif dimension == HealthDimension.RESILIENCE:
                risk_factors.append("System resilience decline")
        
        return risk_factors[:5]  # Limit to top 5 risk factors
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of health monitoring"""
        summary = {
            "node_id": self.node_id,
            "total_metrics": self.total_metrics,
            "total_assessments": self.assessments_generated,
            "total_alerts": self.alerts_generated,
            "active_metrics": len(self.health_metrics),
            "active_assessments": len(self.health_assessments)
        }
        
        # Add dimension health distribution
        dimension_health = {}
        for dimension in HealthDimension:
            metrics = [m for m in self.health_metrics.values() if m.dimension == dimension]
            if metrics:
                avg_health = statistics.mean([m.metric_value for m in metrics])
                dimension_health[dimension.value] = avg_health
        
        summary["dimension_health"] = dimension_health
        
        # Add health status distribution
        status_counts = {}
        for metric in self.health_metrics.values():
            status = metric.health_status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        summary["health_status_distribution"] = status_counts
        
        return summary
    
    def cleanup_old_data(self, max_age_days: int = 90) -> int:
        """Clean up old health monitoring data"""
        cutoff_time = datetime.utcnow() - timedelta(days=max_age_days)
        cleaned_count = 0
        
        # Remove old metrics
        old_metrics = []
        for metric_id, metric in self.health_metrics.items():
            if metric.timestamp < cutoff_time:
                old_metrics.append(metric_id)
        
        for metric_id in old_metrics:
            del self.health_metrics[metric_id]
            cleaned_count += 1
        
        # Remove old assessments
        old_assessments = []
        for assessment_id, assessment in self.health_assessments.items():
            if assessment.timestamp < cutoff_time:
                old_assessments.append(assessment_id)
        
        for assessment_id in old_assessments:
            del self.health_assessments[assessment_id]
            cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old health monitoring data points")
        
        return cleaned_count

