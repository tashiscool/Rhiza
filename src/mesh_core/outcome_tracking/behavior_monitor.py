"""
Behavior Monitor
================

Advanced behavioral monitoring system for tracking agent behaviors,
patterns, and changes over time within The Mesh network.
"""

import asyncio
import time
import statistics
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class BehaviorType(Enum):
    """Types of behaviors to monitor"""
    COMMUNICATION_PATTERN = "communication_pattern"
    DECISION_MAKING = "decision_making"
    COLLABORATION = "collaboration"
    RESOURCE_USAGE = "resource_usage"
    PERFORMANCE_TREND = "performance_trend"
    LEARNING_BEHAVIOR = "learning_behavior"
    SOCIAL_INTERACTION = "social_interaction"
    ADAPTATION_RESPONSE = "adaptation_response"
    CONFLICT_BEHAVIOR = "conflict_behavior"
    TRUST_BEHAVIOR = "trust_behavior"

class BehaviorPattern(Enum):
    """Recognized behavioral patterns"""
    CONSISTENT = "consistent"
    IMPROVING = "improving"
    DECLINING = "declining"
    ERRATIC = "erratic"
    CYCLICAL = "cyclical"
    ADAPTIVE = "adaptive"
    EMERGENT = "emergent"

class MonitoringLevel(Enum):
    """Levels of behavioral monitoring"""
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    FORENSIC = "forensic"

@dataclass
class BehaviorMetric:
    """Individual behavioral metric"""
    metric_name: str
    value: float
    timestamp: float
    confidence: float
    context: Dict[str, Any]
    anomaly_score: float = 0.0

@dataclass
class BehaviorProfile:
    """Comprehensive behavior profile for an agent"""
    agent_id: str
    profile_start: float
    last_updated: float
    behavior_types: Dict[BehaviorType, List[BehaviorMetric]]
    behavioral_patterns: Dict[str, BehaviorPattern]
    anomaly_history: List[Dict]
    baseline_established: bool
    confidence_score: float
    stability_index: float
    adaptation_rate: float
    social_score: float

@dataclass
class BehaviorAlert:
    """Alert for significant behavioral changes"""
    alert_id: str
    agent_id: str
    behavior_type: BehaviorType
    alert_type: str  # "anomaly", "pattern_change", "degradation", "improvement"
    severity: str    # "low", "medium", "high", "critical"
    description: str
    metrics: Dict[str, float]
    timestamp: float
    investigation_required: bool

@dataclass
class BehaviorChange:
    """Represents a detected behavioral change"""
    change_id: str
    agent_id: str
    behavior_type: BehaviorType
    change_magnitude: float
    change_direction: str  # "positive", "negative", "neutral"
    before_value: float
    after_value: float
    detection_time: float
    confidence: float
    context: Dict[str, Any]

# Additional enums for compatibility
class ChangePattern(Enum):
    """Patterns of behavioral change"""
    GRADUAL = "gradual"
    SUDDEN = "sudden"
    OSCILLATING = "oscillating"
    PROGRESSIVE = "progressive"

class MonitoringScope(Enum):
    """Scope of behavioral monitoring"""
    INDIVIDUAL = "individual"
    GROUP = "group"
    SYSTEM_WIDE = "system_wide"

class BehaviorMonitor:
    """Advanced behavioral monitoring and analysis system"""
    
    def __init__(self, node_id: str, monitoring_config: Optional[Dict] = None):
        self.node_id = node_id
        self.config = monitoring_config or {}
        
        # Monitoring data
        self.agent_profiles: Dict[str, BehaviorProfile] = {}
        self.behavior_baselines: Dict[str, Dict[BehaviorType, Dict]] = {}
        self.monitoring_sessions: Dict[str, Dict] = {}
        
        # Alert system
        self.active_alerts: Dict[str, BehaviorAlert] = {}
        self.alert_history: List[BehaviorAlert] = []
        
        # Pattern recognition
        self.pattern_templates: Dict[str, Dict] = {}
        self.anomaly_thresholds: Dict[BehaviorType, float] = {}
        
        # Performance tracking
        self.monitoring_stats: Dict[str, Any] = {
            "total_agents_monitored": 0,
            "total_behaviors_tracked": 0,
            "alerts_generated": 0,
            "patterns_detected": 0
        }
        
        # Initialize default configurations
        self._initialize_monitoring_config()
        
        logger.info(f"BehaviorMonitor initialized for node {node_id}")

    def _initialize_monitoring_config(self):
        """Initialize default monitoring configurations"""
        # Set default anomaly thresholds
        self.anomaly_thresholds = {
            BehaviorType.COMMUNICATION_PATTERN: 0.3,
            BehaviorType.DECISION_MAKING: 0.4,
            BehaviorType.COLLABORATION: 0.35,
            BehaviorType.RESOURCE_USAGE: 0.5,
            BehaviorType.PERFORMANCE_TREND: 0.25,
            BehaviorType.LEARNING_BEHAVIOR: 0.3,
            BehaviorType.SOCIAL_INTERACTION: 0.4,
            BehaviorType.ADAPTATION_RESPONSE: 0.3,
            BehaviorType.CONFLICT_BEHAVIOR: 0.2,
            BehaviorType.TRUST_BEHAVIOR: 0.15
        }
        
        # Initialize pattern templates
        self._initialize_pattern_templates()

    def _initialize_pattern_templates(self):
        """Initialize behavioral pattern templates"""
        self.pattern_templates = {
            "consistent_performer": {
                "performance_variance": {"max": 0.1},
                "communication_regularity": {"min": 0.8},
                "resource_efficiency": {"min": 0.7}
            },
            "adaptive_learner": {
                "learning_rate": {"min": 0.6},
                "adaptation_speed": {"min": 0.7},
                "performance_improvement": {"min": 0.1}
            },
            "collaborative_agent": {
                "cooperation_score": {"min": 0.8},
                "communication_frequency": {"min": 0.7},
                "conflict_resolution": {"min": 0.6}
            },
            "resource_optimizer": {
                "resource_efficiency": {"min": 0.8},
                "waste_minimization": {"min": 0.7},
                "utilization_rate": {"min": 0.75}
            }
        }

    async def register_agent_for_monitoring(
        self,
        agent_id: str,
        monitoring_level: MonitoringLevel = MonitoringLevel.BASIC,
        behavior_types: Optional[Set[BehaviorType]] = None
    ) -> bool:
        """Register an agent for behavioral monitoring"""
        try:
            if behavior_types is None:
                behavior_types = {
                    BehaviorType.COMMUNICATION_PATTERN,
                    BehaviorType.DECISION_MAKING,
                    BehaviorType.COLLABORATION,
                    BehaviorType.PERFORMANCE_TREND
                }
            
            # Create behavior profile
            profile = BehaviorProfile(
                agent_id=agent_id,
                profile_start=time.time(),
                last_updated=time.time(),
                behavior_types={bt: [] for bt in behavior_types},
                behavioral_patterns={},
                anomaly_history=[],
                baseline_established=False,
                confidence_score=0.5,
                stability_index=0.5,
                adaptation_rate=0.5,
                social_score=0.5
            )
            
            self.agent_profiles[agent_id] = profile
            self.behavior_baselines[agent_id] = {}
            
            self.monitoring_stats["total_agents_monitored"] += 1
            
            logger.info(f"Agent registered for monitoring: {agent_id} ({monitoring_level.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent for monitoring: {e}")
            return False

    async def record_behavior_metric(
        self,
        agent_id: str,
        behavior_type: BehaviorType,
        metric_name: str,
        value: float,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Record a behavioral metric for an agent"""
        try:
            if agent_id not in self.agent_profiles:
                logger.warning(f"Agent not registered for monitoring: {agent_id}")
                return False
            
            profile = self.agent_profiles[agent_id]
            
            if behavior_type not in profile.behavior_types:
                profile.behavior_types[behavior_type] = []
            
            # Calculate anomaly score
            anomaly_score = await self._calculate_anomaly_score(
                agent_id, behavior_type, metric_name, value
            )
            
            # Create behavior metric
            metric = BehaviorMetric(
                metric_name=metric_name,
                value=value,
                timestamp=time.time(),
                confidence=0.8,
                context=context or {},
                anomaly_score=anomaly_score
            )
            
            profile.behavior_types[behavior_type].append(metric)
            profile.last_updated = time.time()
            
            # Update baseline if needed
            await self._update_baseline(agent_id, behavior_type, metric_name, value)
            
            # Check for anomalies
            if anomaly_score > self.anomaly_thresholds.get(behavior_type, 0.5):
                await self._generate_anomaly_alert(agent_id, behavior_type, metric, anomaly_score)
            
            # Update behavioral patterns
            await self._update_behavioral_patterns(agent_id, behavior_type)
            
            self.monitoring_stats["total_behaviors_tracked"] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record behavior metric: {e}")
            return False

    async def analyze_behavior_patterns(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Analyze behavioral patterns for an agent"""
        try:
            if agent_id not in self.agent_profiles:
                return None
            
            profile = self.agent_profiles[agent_id]
            analysis = {
                "agent_id": agent_id,
                "monitoring_duration_hours": (time.time() - profile.profile_start) / 3600,
                "behavioral_patterns": {},
                "stability_metrics": {},
                "anomaly_summary": {},
                "recommendations": []
            }
            
            # Analyze each behavior type
            for behavior_type, metrics in profile.behavior_types.items():
                if not metrics:
                    continue
                
                pattern_analysis = await self._analyze_behavior_type_pattern(
                    agent_id, behavior_type, metrics
                )
                analysis["behavioral_patterns"][behavior_type.value] = pattern_analysis
            
            # Calculate stability metrics
            analysis["stability_metrics"] = await self._calculate_stability_metrics(profile)
            
            # Summarize anomalies
            analysis["anomaly_summary"] = await self._summarize_anomalies(profile)
            
            # Generate recommendations
            analysis["recommendations"] = await self._generate_behavior_recommendations(profile)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze behavior patterns: {e}")
            return None

    async def detect_behavioral_anomalies(
        self,
        agent_id: str,
        time_window_hours: float = 24.0
    ) -> List[Dict[str, Any]]:
        """Detect behavioral anomalies for an agent"""
        try:
            if agent_id not in self.agent_profiles:
                return []
            
            profile = self.agent_profiles[agent_id]
            anomalies = []
            
            cutoff_time = time.time() - (time_window_hours * 3600)
            
            for behavior_type, metrics in profile.behavior_types.items():
                recent_metrics = [
                    m for m in metrics 
                    if m.timestamp > cutoff_time
                ]
                
                for metric in recent_metrics:
                    if metric.anomaly_score > self.anomaly_thresholds.get(behavior_type, 0.5):
                        anomalies.append({
                            "behavior_type": behavior_type.value,
                            "metric_name": metric.metric_name,
                            "value": metric.value,
                            "anomaly_score": metric.anomaly_score,
                            "timestamp": metric.timestamp,
                            "context": metric.context
                        })
            
            return sorted(anomalies, key=lambda x: x["anomaly_score"], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to detect behavioral anomalies: {e}")
            return []

    async def _calculate_anomaly_score(
        self,
        agent_id: str,
        behavior_type: BehaviorType,
        metric_name: str,
        value: float
    ) -> float:
        """Calculate anomaly score for a behavioral metric"""
        try:
            # Get baseline data
            baseline_key = f"{behavior_type.value}_{metric_name}"
            baseline = self.behavior_baselines[agent_id].get(baseline_key)
            
            if not baseline or len(baseline.get("values", [])) < 5:
                return 0.0  # Not enough data for anomaly detection
            
            baseline_mean = baseline["mean"]
            baseline_std = baseline["std"]
            
            if baseline_std == 0:
                return 0.0  # No variation in baseline
            
            # Calculate z-score
            z_score = abs(value - baseline_mean) / baseline_std
            
            # Convert to anomaly score (0-1 scale)
            anomaly_score = min(z_score / 3.0, 1.0)  # 3-sigma rule
            
            return anomaly_score
            
        except Exception as e:
            logger.error(f"Failed to calculate anomaly score: {e}")
            return 0.0

    async def _update_baseline(
        self,
        agent_id: str,
        behavior_type: BehaviorType,
        metric_name: str,
        value: float
    ) -> bool:
        """Update baseline statistics for a metric"""
        try:
            baseline_key = f"{behavior_type.value}_{metric_name}"
            
            if baseline_key not in self.behavior_baselines[agent_id]:
                self.behavior_baselines[agent_id][baseline_key] = {
                    "values": [],
                    "mean": 0.0,
                    "std": 0.0
                }
            
            baseline = self.behavior_baselines[agent_id][baseline_key]
            baseline["values"].append(value)
            
            # Keep only recent values (sliding window)
            max_baseline_samples = 100
            if len(baseline["values"]) > max_baseline_samples:
                baseline["values"] = baseline["values"][-max_baseline_samples:]
            
            # Update statistics
            values = baseline["values"]
            baseline["mean"] = statistics.mean(values)
            baseline["std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            
            # Mark baseline as established if we have enough samples
            if len(values) >= 10:
                self.agent_profiles[agent_id].baseline_established = True
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update baseline: {e}")
            return False

    async def _generate_anomaly_alert(
        self,
        agent_id: str,
        behavior_type: BehaviorType,
        metric: BehaviorMetric,
        anomaly_score: float
    ) -> bool:
        """Generate an alert for behavioral anomaly"""
        try:
            alert_id = self._generate_alert_id(agent_id, behavior_type)
            
            # Determine severity
            if anomaly_score >= 0.8:
                severity = "critical"
            elif anomaly_score >= 0.6:
                severity = "high"
            elif anomaly_score >= 0.4:
                severity = "medium"
            else:
                severity = "low"
            
            alert = BehaviorAlert(
                alert_id=alert_id,
                agent_id=agent_id,
                behavior_type=behavior_type,
                alert_type="anomaly",
                severity=severity,
                description=f"Behavioral anomaly detected in {behavior_type.value}: {metric.metric_name}",
                metrics={
                    "anomaly_score": anomaly_score,
                    "metric_value": metric.value,
                    "metric_name": metric.metric_name
                },
                timestamp=time.time(),
                investigation_required=anomaly_score >= 0.7
            )
            
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            self.monitoring_stats["alerts_generated"] += 1
            
            # Add to agent's anomaly history
            self.agent_profiles[agent_id].anomaly_history.append({
                "timestamp": time.time(),
                "behavior_type": behavior_type.value,
                "anomaly_score": anomaly_score,
                "alert_id": alert_id
            })
            
            logger.warning(f"Behavioral anomaly alert: {alert_id} ({severity})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate anomaly alert: {e}")
            return False

    async def _update_behavioral_patterns(
        self,
        agent_id: str,
        behavior_type: BehaviorType
    ) -> bool:
        """Update detected behavioral patterns for an agent"""
        try:
            profile = self.agent_profiles[agent_id]
            metrics = profile.behavior_types.get(behavior_type, [])
            
            if len(metrics) < 10:  # Need sufficient data
                return True
            
            # Analyze recent trend (last 20 data points)
            recent_metrics = metrics[-20:]
            values = [m.value for m in recent_metrics]
            
            # Detect pattern
            pattern = await self._detect_pattern(values)
            profile.behavioral_patterns[behavior_type.value] = pattern
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update behavioral patterns: {e}")
            return False

    async def _detect_pattern(self, values: List[float]) -> BehaviorPattern:
        """Detect behavioral pattern from a sequence of values"""
        if len(values) < 5:
            return BehaviorPattern.CONSISTENT
        
        # Calculate trend and variance
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0
        
        # Calculate trend using simple linear regression
        n = len(values)
        x_vals = list(range(n))
        
        if n > 1:
            slope = (n * sum(i * v for i, v in enumerate(values)) - sum(x_vals) * sum(values)) / \
                   (n * sum(x * x for x in x_vals) - sum(x_vals) ** 2)
        else:
            slope = 0
        
        # Classify pattern
        if std_val < 0.1 * mean_val:
            return BehaviorPattern.CONSISTENT
        elif slope > 0.1 * mean_val / n:
            return BehaviorPattern.IMPROVING
        elif slope < -0.1 * mean_val / n:
            return BehaviorPattern.DECLINING
        elif std_val > 0.3 * mean_val:
            return BehaviorPattern.ERRATIC
        else:
            return BehaviorPattern.ADAPTIVE

    async def _analyze_behavior_type_pattern(
        self,
        agent_id: str,
        behavior_type: BehaviorType,
        metrics: List[BehaviorMetric]
    ) -> Dict[str, Any]:
        """Analyze patterns for a specific behavior type"""
        if not metrics:
            return {"status": "insufficient_data"}
        
        values = [m.value for m in metrics]
        timestamps = [m.timestamp for m in metrics]
        
        analysis = {
            "total_observations": len(metrics),
            "time_span_hours": (max(timestamps) - min(timestamps)) / 3600 if len(timestamps) > 1 else 0,
            "mean_value": statistics.mean(values),
            "std_deviation": statistics.stdev(values) if len(values) > 1 else 0,
            "min_value": min(values),
            "max_value": max(values),
            "pattern": await self._detect_pattern(values),
            "stability_score": 1.0 - (statistics.stdev(values) / statistics.mean(values)) if statistics.mean(values) > 0 and len(values) > 1 else 0.5,
            "recent_trend": "stable"  # Simplified
        }
        
        return analysis

    async def _calculate_stability_metrics(self, profile: BehaviorProfile) -> Dict[str, float]:
        """Calculate stability metrics for an agent"""
        stability_metrics = {
            "overall_stability": 0.5,
            "consistency_score": 0.5,
            "predictability": 0.5,
            "anomaly_frequency": 0.0
        }
        
        # Calculate anomaly frequency
        recent_anomalies = [
            a for a in profile.anomaly_history
            if time.time() - a["timestamp"] < 86400  # Last 24 hours
        ]
        
        stability_metrics["anomaly_frequency"] = len(recent_anomalies) / 24.0  # Per hour
        
        # Calculate overall stability based on behavioral patterns
        pattern_stability = 0
        stable_patterns = {BehaviorPattern.CONSISTENT, BehaviorPattern.IMPROVING, BehaviorPattern.ADAPTIVE}
        
        for pattern in profile.behavioral_patterns.values():
            if pattern in stable_patterns:
                pattern_stability += 1
        
        if profile.behavioral_patterns:
            stability_metrics["overall_stability"] = pattern_stability / len(profile.behavioral_patterns)
        
        return stability_metrics

    async def _summarize_anomalies(self, profile: BehaviorProfile) -> Dict[str, Any]:
        """Summarize anomaly information for an agent"""
        return {
            "total_anomalies": len(profile.anomaly_history),
            "recent_anomalies_24h": len([
                a for a in profile.anomaly_history
                if time.time() - a["timestamp"] < 86400
            ]),
            "high_severity_anomalies": len([
                a for a in profile.anomaly_history
                if a.get("anomaly_score", 0) > 0.7
            ]),
            "most_common_anomaly_type": "communication_pattern"  # Simplified
        }

    async def _generate_behavior_recommendations(self, profile: BehaviorProfile) -> List[str]:
        """Generate behavioral recommendations for an agent"""
        recommendations = []
        
        # Check for high anomaly frequency
        recent_anomalies = [
            a for a in profile.anomaly_history
            if time.time() - a["timestamp"] < 86400
        ]
        
        if len(recent_anomalies) > 5:
            recommendations.append("Consider investigating recent behavioral changes")
        
        # Check for declining patterns
        for behavior_type, pattern in profile.behavioral_patterns.items():
            if pattern == BehaviorPattern.DECLINING:
                recommendations.append(f"Monitor {behavior_type} performance - declining trend detected")
            elif pattern == BehaviorPattern.ERRATIC:
                recommendations.append(f"Stabilize {behavior_type} behavior - high variability detected")
        
        if not recommendations:
            recommendations.append("Behavioral patterns appear stable")
        
        return recommendations

    def _generate_alert_id(self, agent_id: str, behavior_type: BehaviorType) -> str:
        """Generate unique alert ID"""
        content = f"alert|{agent_id}|{behavior_type.value}|{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        try:
            active_agents = len(self.agent_profiles)
            total_alerts = len(self.alert_history)
            active_alerts = len(self.active_alerts)
            
            return {
                "monitoring_stats": self.monitoring_stats,
                "active_agents": active_agents,
                "total_alerts_generated": total_alerts,
                "active_alerts": active_alerts,
                "monitoring_uptime_hours": 0,  # Would track actual uptime
                "average_metrics_per_agent": (
                    self.monitoring_stats["total_behaviors_tracked"] / active_agents
                ) if active_agents > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get monitoring summary: {e}")
            return {"error": str(e)}