"""
Mesh Entropy Monitor
===================

Component 10.1: Mesh Degeneration Watchdogs
Monitor long-term entropy and system degradation

Implements statistical analysis of long-term trends,
entropy measurement, and degradation detection.
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


class EntropyType(Enum):
    """Types of entropy to monitor"""
    INFORMATION = "information"          # Information entropy
    BEHAVIORAL = "behavioral"            # Behavioral entropy
    STRUCTURAL = "structural"            # Structural entropy
    CULTURAL = "cultural"                # Cultural entropy
    TECHNICAL = "technical"              # Technical entropy


class EntropyLevel(Enum):
    """Entropy levels"""
    LOW = "low"                          # Healthy, organized state
    MODERATE = "moderate"                # Some disorder, manageable
    HIGH = "high"                        # Significant disorder, concerning
    CRITICAL = "critical"                # Severe disorder, immediate action needed


@dataclass
class EntropyMeasurement:
    """A measurement of entropy at a specific time"""
    measurement_id: str
    entropy_type: EntropyType
    timestamp: datetime
    
    # Measurement data
    entropy_value: float  # 0.0 to 1.0, higher = more disorder
    baseline_value: float  # Expected value for this type
    deviation: float  # Difference from baseline
    
    # Context
    node_id: str
    measurement_context: Dict[str, Any]
    
    # Analysis
    entropy_level: EntropyLevel
    confidence_score: float  # 0.0 to 1.0
    contributing_factors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.measurement_id:
            self.measurement_id = self._generate_measurement_id()
    
    def _generate_measurement_id(self) -> str:
        """Generate unique measurement ID"""
        content = f"{self.entropy_type.value}{self.node_id}{self.timestamp.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entropy measurement to dictionary"""
        return {
            "measurement_id": self.measurement_id,
            "entropy_type": self.entropy_type.value,
            "timestamp": self.timestamp.isoformat(),
            "entropy_value": self.entropy_value,
            "baseline_value": self.baseline_value,
            "deviation": self.deviation,
            "node_id": self.node_id,
            "measurement_context": self.measurement_context,
            "entropy_level": self.entropy_level.value,
            "confidence_score": self.confidence_score,
            "contributing_factors": self.contributing_factors
        }


@dataclass
class EntropyTrend:
    """Trend analysis of entropy over time"""
    trend_id: str
    entropy_type: EntropyType
    node_id: str
    start_time: datetime
    end_time: datetime
    
    # Trend data
    measurements: List[EntropyMeasurement] = field(default_factory=list)
    trend_direction: str = "stable"  # increasing, decreasing, stable, fluctuating
    trend_strength: float = 0.0  # 0.0 to 1.0, how strong the trend is
    
    # Analysis results
    slope: float = 0.0  # Rate of change
    correlation: float = 0.0  # Correlation with time
    volatility: float = 0.0  # How much the values fluctuate
    
    # Alerts
    alert_level: str = "none"  # none, warning, alert, critical
    alert_message: str = ""
    
    def __post_init__(self):
        if not self.trend_id:
            self.trend_id = self._generate_trend_id()
    
    def _generate_trend_id(self) -> str:
        """Generate unique trend ID"""
        content = f"{self.entropy_type.value}{self.node_id}{self.start_time.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entropy trend to dictionary"""
        return {
            "trend_id": self.trend_id,
            "entropy_type": self.entropy_type.value,
            "node_id": self.node_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "measurements": [m.to_dict() for m in self.measurements],
            "trend_direction": self.trend_direction,
            "trend_strength": self.trend_strength,
            "slope": self.slope,
            "correlation": self.correlation,
            "volatility": self.volatility,
            "alert_level": self.alert_level,
            "alert_message": self.alert_message
        }


class EntropyMonitor:
    """
    Monitors long-term entropy and system degradation
    
    Tracks entropy across multiple dimensions, analyzes trends,
    and provides early warning of system degradation.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        
        # Storage
        self.entropy_measurements: Dict[str, EntropyMeasurement] = {}
        self.entropy_trends: Dict[str, EntropyTrend] = {}
        
        # Configuration
        self.measurement_interval = timedelta(hours=1)
        self.trend_analysis_window = timedelta(days=30)
        self.alert_thresholds = {
            EntropyType.INFORMATION: 0.7,
            EntropyType.BEHAVIORAL: 0.6,
            EntropyType.STRUCTURAL: 0.8,
            EntropyType.CULTURAL: 0.5,
            EntropyType.TECHNICAL: 0.75
        }
        
        # Baseline values (expected entropy for healthy system)
        self.baseline_values = {
            EntropyType.INFORMATION: 0.3,  # Some information disorder is normal
            EntropyType.BEHAVIORAL: 0.2,   # Behavior should be relatively organized
            EntropyType.STRUCTURAL: 0.1,   # Structure should be very organized
            EntropyType.CULTURAL: 0.2,     # Culture should be relatively stable
            EntropyType.TECHNICAL: 0.2     # Technical systems should be organized
        }
        
        # Performance metrics
        self.total_measurements = 0
        self.alerts_generated = 0
        self.trends_analyzed = 0
        
        logger.info(f"EntropyMonitor initialized for node: {self.node_id}")
    
    def measure_entropy(self, entropy_type: EntropyType, 
                       measurement_data: Dict[str, Any],
                       context: Dict[str, Any] = None) -> EntropyMeasurement:
        """Measure entropy for a specific type"""
        try:
            # Calculate entropy value based on type
            entropy_value = self._calculate_entropy(entropy_type, measurement_data)
            
            # Get baseline and calculate deviation
            baseline_value = self.baseline_values[entropy_type]
            deviation = entropy_value - baseline_value
            
            # Determine entropy level
            entropy_level = self._determine_entropy_level(entropy_value, entropy_type)
            
            # Analyze contributing factors
            contributing_factors = self._identify_contributing_factors(entropy_type, measurement_data)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(measurement_data, context)
            
            # Create measurement
            measurement = EntropyMeasurement(
                measurement_id="",
                entropy_type=entropy_type,
                timestamp=datetime.utcnow(),
                entropy_value=entropy_value,
                baseline_value=baseline_value,
                deviation=deviation,
                node_id=self.node_id,
                measurement_context=context or {},
                entropy_level=entropy_level,
                confidence_score=confidence_score,
                contributing_factors=contributing_factors
            )
            
            # Store measurement
            self.entropy_measurements[measurement.measurement_id] = measurement
            
            # Update statistics
            self.total_measurements += 1
            
            # Check for alerts
            if entropy_level in [EntropyLevel.HIGH, EntropyLevel.CRITICAL]:
                self._generate_alert(measurement)
            
            logger.info(f"Measured {entropy_type.value} entropy: {entropy_value:.3f} (level: {entropy_level.value})")
            return measurement
            
        except Exception as e:
            logger.error(f"Failed to measure entropy: {e}")
            raise
    
    def _calculate_entropy(self, entropy_type: EntropyType, data: Dict[str, Any]) -> float:
        """Calculate entropy value based on type and data"""
        if entropy_type == EntropyType.INFORMATION:
            return self._calculate_information_entropy(data)
        elif entropy_type == EntropyType.BEHAVIORAL:
            return self._calculate_behavioral_entropy(data)
        elif entropy_type == EntropyType.STRUCTURAL:
            return self._calculate_structural_entropy(data)
        elif entropy_type == EntropyType.CULTURAL:
            return self._calculate_cultural_entropy(data)
        elif entropy_type == EntropyType.TECHNICAL:
            return self._calculate_technical_entropy(data)
        else:
            return 0.5  # Default entropy value
    
    def _calculate_information_entropy(self, data: Dict[str, Any]) -> float:
        """Calculate information entropy (Shannon entropy)"""
        if "information_distribution" not in data:
            return 0.5
        
        distribution = data["information_distribution"]
        if not distribution:
            return 0.0
        
        # Calculate Shannon entropy: -sum(p * log2(p))
        total = sum(distribution.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in distribution.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # Normalize to 0-1 range (assuming max entropy is log2(n) where n is number of categories)
        max_entropy = math.log2(len(distribution)) if len(distribution) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return min(1.0, normalized_entropy)
    
    def _calculate_behavioral_entropy(self, data: Dict[str, Any]) -> float:
        """Calculate behavioral entropy based on pattern consistency"""
        if "behavior_patterns" not in data:
            return 0.5
        
        patterns = data["behavior_patterns"]
        if not patterns:
            return 0.0
        
        # Calculate consistency score (lower consistency = higher entropy)
        total_behaviors = sum(patterns.values())
        if total_behaviors == 0:
            return 0.0
        
        # Calculate Gini coefficient (measure of inequality)
        sorted_counts = sorted(patterns.values())
        n = len(sorted_counts)
        cumsum = 0
        for i, count in enumerate(sorted_counts):
            cumsum += (n - i) * count
        
        gini = (n + 1 - 2 * cumsum / total_behaviors) / n if total_behaviors > 0 else 0
        
        # Convert to entropy (higher Gini = lower entropy)
        entropy = 1.0 - gini
        return max(0.0, min(1.0, entropy))
    
    def _calculate_structural_entropy(self, data: Dict[str, Any]) -> float:
        """Calculate structural entropy based on organization metrics"""
        if "structural_metrics" not in data:
            return 0.5
        
        metrics = data["structural_metrics"]
        
        # Calculate based on multiple structural factors
        factors = []
        
        if "connectivity" in metrics:
            # Lower connectivity = higher entropy
            connectivity = metrics["connectivity"]
            factors.append(1.0 - connectivity)
        
        if "coherence" in metrics:
            # Lower coherence = higher entropy
            coherence = metrics["coherence"]
            factors.append(1.0 - coherence)
        
        if "complexity" in metrics:
            # Higher complexity = higher entropy (up to a point)
            complexity = metrics["complexity"]
            factors.append(min(1.0, complexity / 10.0))  # Normalize complexity
        
        if factors:
            return sum(factors) / len(factors)
        else:
            return 0.5
    
    def _calculate_cultural_entropy(self, data: Dict[str, Any]) -> float:
        """Calculate cultural entropy based on value consistency"""
        if "cultural_metrics" not in data:
            return 0.5
        
        metrics = data["cultural_metrics"]
        
        # Calculate based on cultural factors
        factors = []
        
        if "value_consistency" in metrics:
            # Lower consistency = higher entropy
            consistency = metrics["value_consistency"]
            factors.append(1.0 - consistency)
        
        if "norm_adherence" in metrics:
            # Lower adherence = higher entropy
            adherence = metrics["norm_adherence"]
            factors.append(1.0 - adherence)
        
        if "cultural_stability" in metrics:
            # Lower stability = higher entropy
            stability = metrics["cultural_stability"]
            factors.append(1.0 - stability)
        
        if factors:
            return sum(factors) / len(factors)
        else:
            return 0.5
    
    def _calculate_technical_entropy(self, data: Dict[str, Any]) -> float:
        """Calculate technical entropy based on system health"""
        if "technical_metrics" not in data:
            return 0.5
        
        metrics = data["technical_metrics"]
        
        # Calculate based on technical factors
        factors = []
        
        if "error_rate" in metrics:
            # Higher error rate = higher entropy
            error_rate = metrics["error_rate"]
            factors.append(min(1.0, error_rate))
        
        if "response_time" in metrics:
            # Higher response time = higher entropy (normalized)
            response_time = metrics["response_time"]
            factors.append(min(1.0, response_time / 1000.0))  # Normalize to seconds
        
        if "resource_utilization" in metrics:
            # Very high or very low utilization = higher entropy
            utilization = metrics["resource_utilization"]
            if utilization < 0.2 or utilization > 0.8:
                factors.append(0.5)
            else:
                factors.append(0.1)
        
        if factors:
            return sum(factors) / len(factors)
        else:
            return 0.5
    
    def _determine_entropy_level(self, entropy_value: float, entropy_type: EntropyType) -> EntropyLevel:
        """Determine entropy level based on value and type"""
        threshold = self.alert_thresholds[entropy_type]
        
        if entropy_value < threshold * 0.5:
            return EntropyLevel.LOW
        elif entropy_value < threshold * 0.8:
            return EntropyLevel.MODERATE
        elif entropy_value < threshold:
            return EntropyLevel.HIGH
        else:
            return EntropyLevel.CRITICAL
    
    def _identify_contributing_factors(self, entropy_type: EntropyType, data: Dict[str, Any]) -> List[str]:
        """Identify factors contributing to high entropy"""
        factors = []
        
        if entropy_type == EntropyType.INFORMATION:
            if "information_distribution" in data:
                distribution = data["information_distribution"]
                if len(distribution) > 20:
                    factors.append("Information overload")
                if any(count > 1000 for count in distribution.values()):
                    factors.append("Information concentration")
        
        elif entropy_type == EntropyType.BEHAVIORAL:
            if "behavior_patterns" in data:
                patterns = data["behavior_patterns"]
                if len(patterns) < 3:
                    factors.append("Behavioral rigidity")
                if any(count < 5 for count in patterns.values()):
                    factors.append("Inconsistent behavior")
        
        elif entropy_type == EntropyType.STRUCTURAL:
            if "structural_metrics" in data:
                metrics = data["structural_metrics"]
                if metrics.get("connectivity", 1.0) < 0.3:
                    factors.append("Low connectivity")
                if metrics.get("coherence", 1.0) < 0.5:
                    factors.append("Low coherence")
        
        elif entropy_type == EntropyType.CULTURAL:
            if "cultural_metrics" in data:
                metrics = data["cultural_metrics"]
                if metrics.get("value_consistency", 1.0) < 0.6:
                    factors.append("Value inconsistency")
                if metrics.get("cultural_stability", 1.0) < 0.4:
                    factors.append("Cultural instability")
        
        elif entropy_type == EntropyType.TECHNICAL:
            if "technical_metrics" in data:
                metrics = data["technical_metrics"]
                if metrics.get("error_rate", 0.0) > 0.1:
                    factors.append("High error rate")
                if metrics.get("response_time", 0.0) > 500:
                    factors.append("Slow response time")
        
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
    
    def _generate_alert(self, measurement: EntropyMeasurement):
        """Generate alert for high entropy measurement"""
        alert_message = f"High {measurement.entropy_type.value} entropy detected: {measurement.entropy_value:.3f}"
        
        if measurement.contributing_factors:
            alert_message += f" Contributing factors: {', '.join(measurement.contributing_factors)}"
        
        logger.warning(f"ENTROPY ALERT: {alert_message}")
        
        # Store alert
        self.alerts_generated += 1
        
        # Could send to alert system, notification system, etc.
    
    def analyze_trends(self, entropy_type: EntropyType = None, 
                      time_window: timedelta = None) -> List[EntropyTrend]:
        """Analyze entropy trends over time"""
        try:
            if time_window is None:
                time_window = self.trend_analysis_window
            
            end_time = datetime.utcnow()
            start_time = end_time - time_window
            
            # Filter measurements by time and type
            relevant_measurements = []
            for measurement in self.entropy_measurements.values():
                if (measurement.timestamp >= start_time and 
                    measurement.timestamp <= end_time and
                    (entropy_type is None or measurement.entropy_type == entropy_type)):
                    relevant_measurements.append(measurement)
            
            if not relevant_measurements:
                return []
            
            # Group by type and node
            grouped_measurements = {}
            for measurement in relevant_measurements:
                key = (measurement.entropy_type, measurement.node_id)
                if key not in grouped_measurements:
                    grouped_measurements[key] = []
                grouped_measurements[key].append(measurement)
            
            # Analyze trends for each group
            trends = []
            for (ent_type, node_id), measurements in grouped_measurements.items():
                trend = self._analyze_trend_group(ent_type, node_id, measurements, start_time, end_time)
                if trend:
                    trends.append(trend)
                    self.entropy_trends[trend.trend_id] = trend
            
            self.trends_analyzed += len(trends)
            logger.info(f"Analyzed {len(trends)} entropy trends")
            
            return trends
            
        except Exception as e:
            logger.error(f"Failed to analyze trends: {e}")
            return []
    
    def _analyze_trend_group(self, entropy_type: EntropyType, node_id: str,
                            measurements: List[EntropyMeasurement],
                            start_time: datetime, end_time: datetime) -> Optional[EntropyTrend]:
        """Analyze trends for a specific group of measurements"""
        if len(measurements) < 3:
            return None
        
        # Sort measurements by time
        measurements.sort(key=lambda x: x.timestamp)
        
        # Calculate trend statistics
        times = [(m.timestamp - start_time).total_seconds() for m in measurements]
        values = [m.entropy_value for m in measurements]
        
        # Calculate slope (rate of change)
        if len(times) > 1:
            slope = self._calculate_slope(times, values)
        else:
            slope = 0.0
        
        # Calculate correlation with time
        if len(times) > 1:
            correlation = self._calculate_correlation(times, values)
        else:
            correlation = 0.0
        
        # Calculate volatility
        volatility = statistics.stdev(values) if len(values) > 1 else 0.0
        
        # Determine trend direction
        trend_direction = self._determine_trend_direction(slope, correlation, volatility)
        
        # Calculate trend strength
        trend_strength = abs(correlation) * (1.0 - volatility)
        
        # Determine alert level
        alert_level, alert_message = self._determine_trend_alert(
            entropy_type, slope, correlation, volatility
        )
        
        # Create trend
        trend = EntropyTrend(
            trend_id="",
            entropy_type=entropy_type,
            node_id=node_id,
            start_time=start_time,
            end_time=end_time,
            measurements=measurements,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            slope=slope,
            correlation=correlation,
            volatility=volatility,
            alert_level=alert_level,
            alert_message=alert_message
        )
        
        return trend
    
    def _calculate_slope(self, times: List[float], values: List[float]) -> float:
        """Calculate slope using linear regression"""
        if len(times) < 2:
            return 0.0
        
        n = len(times)
        sum_x = sum(times)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(times, values))
        sum_x2 = sum(x * x for x in times)
        
        # Linear regression slope: (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x^2)
        numerator = n * sum_xy - sum_x * sum_y
        denominator = n * sum_x2 - sum_x * sum_x
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _calculate_correlation(self, times: List[float], values: List[float]) -> float:
        """Calculate correlation coefficient"""
        if len(times) < 2:
            return 0.0
        
        n = len(times)
        sum_x = sum(times)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(times, values))
        sum_x2 = sum(x * x for x in times)
        sum_y2 = sum(y * y for y in values)
        
        # Pearson correlation coefficient
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _determine_trend_direction(self, slope: float, correlation: float, volatility: float) -> str:
        """Determine the direction of the trend"""
        if abs(correlation) < 0.3:
            return "stable"
        elif volatility > 0.3:
            return "fluctuating"
        elif slope > 0.001:
            return "increasing"
        elif slope < -0.001:
            return "decreasing"
        else:
            return "stable"
    
    def _determine_trend_alert(self, entropy_type: EntropyType, slope: float, 
                              correlation: float, volatility: float) -> Tuple[str, str]:
        """Determine alert level and message for a trend"""
        if abs(correlation) < 0.3:
            return "none", ""
        
        if entropy_type in [EntropyType.CULTURAL, EntropyType.BEHAVIORAL]:
            # Cultural and behavioral changes are more concerning
            if slope > 0.001 and correlation > 0.5:
                return "critical", f"Rapid increase in {entropy_type.value} entropy detected"
            elif slope > 0.0005 and correlation > 0.4:
                return "alert", f"Gradual increase in {entropy_type.value} entropy detected"
            elif slope > 0.0001 and correlation > 0.3:
                return "warning", f"Slow increase in {entropy_type.value} entropy detected"
        
        elif entropy_type in [EntropyType.TECHNICAL, EntropyType.STRUCTURAL]:
            # Technical and structural changes are moderately concerning
            if slope > 0.002 and correlation > 0.6:
                return "critical", f"Rapid degradation in {entropy_type.value} detected"
            elif slope > 0.001 and correlation > 0.5:
                return "alert", f"Significant degradation in {entropy_type.value} detected"
            elif slope > 0.0005 and correlation > 0.4:
                return "warning", f"Moderate degradation in {entropy_type.value} detected"
        
        elif entropy_type == EntropyType.INFORMATION:
            # Information entropy changes are less concerning
            if slope > 0.003 and correlation > 0.7:
                return "alert", f"Rapid increase in information disorder detected"
            elif slope > 0.001 and correlation > 0.5:
                return "warning", f"Increase in information disorder detected"
        
        return "none", ""
    
    def get_entropy_summary(self, entropy_type: EntropyType = None) -> Dict[str, Any]:
        """Get summary of entropy monitoring"""
        if entropy_type:
            measurements = [m for m in self.entropy_measurements.values() if m.entropy_type == entropy_type]
        else:
            measurements = list(self.entropy_measurements.values())
        
        if not measurements:
            return {"message": "No entropy measurements available"}
        
        # Calculate statistics
        current_values = [m.entropy_value for m in measurements if m.timestamp > datetime.utcnow() - timedelta(hours=24)]
        recent_values = [m.entropy_value for m in measurements if m.timestamp > datetime.utcnow() - timedelta(hours=1)]
        
        summary = {
            "node_id": self.node_id,
            "total_measurements": len(measurements),
            "measurements_last_24h": len(current_values),
            "measurements_last_hour": len(recent_values),
            "current_entropy": statistics.mean(current_values) if current_values else 0.0,
            "recent_entropy": statistics.mean(recent_values) if recent_values else 0.0,
            "total_alerts": self.alerts_generated,
            "trends_analyzed": self.trends_analyzed
        }
        
        # Add entropy level distribution
        level_counts = {}
        for measurement in measurements:
            level = measurement.entropy_level.value
            level_counts[level] = level_counts.get(level, 0) + 1
        
        summary["entropy_level_distribution"] = level_counts
        
        return summary
    
    def cleanup_old_data(self, max_age_days: int = 90) -> int:
        """Clean up old entropy data"""
        cutoff_time = datetime.utcnow() - timedelta(days=max_age_days)
        cleaned_count = 0
        
        # Remove old measurements
        old_measurements = []
        for measurement_id, measurement in self.entropy_measurements.items():
            if measurement.timestamp < cutoff_time:
                old_measurements.append(measurement_id)
        
        for measurement_id in old_measurements:
            del self.entropy_measurements[measurement_id]
            cleaned_count += 1
        
        # Remove old trends
        old_trends = []
        for trend_id, trend in self.entropy_trends.items():
            if trend.end_time < cutoff_time:
                old_trends.append(trend_id)
        
        for trend_id in old_trends:
            del self.entropy_trends[trend_id]
            cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old entropy data points")
        
        return cleaned_count

