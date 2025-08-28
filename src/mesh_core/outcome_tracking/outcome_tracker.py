"""
Outcome Tracker
===============

Main outcome tracking system for monitoring decision outcomes,
behavioral changes, and long-term impacts in The Mesh network.
"""

import asyncio
import time
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class OutcomeType(Enum):
    """Types of outcomes to track"""
    DECISION_OUTCOME = "decision_outcome"
    BEHAVIOR_CHANGE = "behavior_change"
    PERFORMANCE_METRIC = "performance_metric"
    SATISFACTION_SCORE = "satisfaction_score"
    SYSTEM_IMPACT = "system_impact"
    LEARNING_OUTCOME = "learning_outcome"
    EMERGENT_BEHAVIOR = "emergent_behavior"

class TrackingStatus(Enum):
    """Status of outcome tracking"""
    ACTIVE = "active"
    COMPLETED = "completed"
    PAUSED = "paused"
    ARCHIVED = "archived"

@dataclass
class OutcomeRecord:
    """Record of a tracked outcome"""
    record_id: str
    outcome_type: OutcomeType
    tracked_entity: str  # What is being tracked
    baseline_metrics: Dict[str, float]
    current_metrics: Dict[str, float]
    tracking_start: float
    last_updated: float
    status: TrackingStatus
    change_trajectory: List[Dict]
    impact_assessment: Optional[Dict] = None
    confidence_level: float = 0.7
    external_factors: List[Dict] = None
    
    def __post_init__(self):
        if self.external_factors is None:
            self.external_factors = []

class OutcomeTracker:
    """Main outcome tracking engine"""
    
    def __init__(self, node_id: str, tracking_config: Optional[Dict] = None):
        self.node_id = node_id
        self.config = tracking_config or {}
        self.active_tracks: Dict[str, OutcomeRecord] = {}
        self.completed_tracks: Dict[str, OutcomeRecord] = {}
        self.tracking_patterns: Dict[str, Dict] = {}
        self.baseline_data: Dict[str, Dict] = {}
        
        logger.info(f"OutcomeTracker initialized for node {node_id}")

    async def start_tracking(
        self,
        entity_id: str,
        outcome_type: OutcomeType,
        baseline_metrics: Dict[str, float],
        tracking_duration: Optional[float] = None
    ) -> str:
        """Start tracking outcomes for an entity"""
        try:
            record_id = f"{entity_id}_{outcome_type.value}_{int(time.time())}"
            
            record = OutcomeRecord(
                record_id=record_id,
                outcome_type=outcome_type,
                tracked_entity=entity_id,
                baseline_metrics=baseline_metrics,
                current_metrics=baseline_metrics.copy(),
                tracking_start=time.time(),
                last_updated=time.time(),
                status=TrackingStatus.ACTIVE,
                change_trajectory=[{
                    "timestamp": time.time(),
                    "metrics": baseline_metrics,
                    "notes": "Baseline established"
                }]
            )
            
            self.active_tracks[record_id] = record
            
            # Store baseline data
            self.baseline_data[entity_id] = baseline_metrics
            
            logger.info(f"Started tracking: {record_id}")
            return record_id
            
        except Exception as e:
            logger.error(f"Failed to start tracking: {e}")
            return ""

    async def update_tracking(
        self,
        record_id: str,
        new_metrics: Dict[str, float],
        external_factors: Optional[List[Dict]] = None
    ) -> bool:
        """Update tracking with new measurements"""
        try:
            if record_id not in self.active_tracks:
                return False
            
            record = self.active_tracks[record_id]
            
            # Calculate changes
            changes = {}
            for metric, value in new_metrics.items():
                if metric in record.current_metrics:
                    changes[metric] = value - record.current_metrics[metric]
                else:
                    changes[metric] = value
            
            # Update record
            record.current_metrics.update(new_metrics)
            record.last_updated = time.time()
            
            if external_factors:
                record.external_factors.extend(external_factors)
            
            # Add to trajectory
            trajectory_entry = {
                "timestamp": time.time(),
                "metrics": new_metrics.copy(),
                "changes": changes,
                "notes": "Regular update"
            }
            
            record.change_trajectory.append(trajectory_entry)
            
            # Update confidence based on data consistency
            record.confidence_level = await self._calculate_confidence(record)
            
            logger.debug(f"Updated tracking: {record_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update tracking: {e}")
            return False

    async def analyze_outcome_trends(self, record_id: str) -> Optional[Dict]:
        """Analyze trends in outcome data"""
        try:
            if record_id not in self.active_tracks:
                record = self.completed_tracks.get(record_id)
                if not record:
                    return None
            else:
                record = self.active_tracks[record_id]
            
            # Analyze trajectory trends
            trends = await self._analyze_trajectory_trends(record)
            
            # Calculate overall change magnitude
            overall_change = await self._calculate_overall_change(record)
            
            # Identify significant changes
            significant_changes = await self._identify_significant_changes(record)
            
            # Predict future trajectory
            future_prediction = await self._predict_future_trajectory(record)
            
            return {
                "record_id": record_id,
                "trends": trends,
                "overall_change": overall_change,
                "significant_changes": significant_changes,
                "future_prediction": future_prediction,
                "confidence_level": record.confidence_level,
                "tracking_duration": record.last_updated - record.tracking_start,
                "data_points": len(record.change_trajectory)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze outcome trends: {e}")
            return None

    async def complete_tracking(self, record_id: str, final_assessment: Optional[Dict] = None) -> bool:
        """Complete tracking for a record"""
        try:
            if record_id not in self.active_tracks:
                return False
            
            record = self.active_tracks[record_id]
            
            # Generate final impact assessment
            if not final_assessment:
                final_assessment = await self._generate_final_assessment(record)
            
            record.impact_assessment = final_assessment
            record.status = TrackingStatus.COMPLETED
            
            # Move to completed tracks
            self.completed_tracks[record_id] = record
            del self.active_tracks[record_id]
            
            logger.info(f"Completed tracking: {record_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete tracking: {e}")
            return False

    async def get_tracking_summary(self) -> Dict:
        """Get summary of all tracking activities"""
        try:
            active_by_type = {}
            completed_by_type = {}
            
            for record in self.active_tracks.values():
                outcome_type = record.outcome_type.value
                active_by_type[outcome_type] = active_by_type.get(outcome_type, 0) + 1
            
            for record in self.completed_tracks.values():
                outcome_type = record.outcome_type.value
                completed_by_type[outcome_type] = completed_by_type.get(outcome_type, 0) + 1
            
            return {
                "active_tracks": len(self.active_tracks),
                "completed_tracks": len(self.completed_tracks),
                "active_by_type": active_by_type,
                "completed_by_type": completed_by_type,
                "tracking_patterns": len(self.tracking_patterns),
                "entities_tracked": len(self.baseline_data)
            }
            
        except Exception as e:
            logger.error(f"Failed to get tracking summary: {e}")
            return {}

    async def _calculate_confidence(self, record: OutcomeRecord) -> float:
        """Calculate confidence level in tracking data"""
        try:
            # Base confidence on data consistency and quantity
            data_points = len(record.change_trajectory)
            time_span = record.last_updated - record.tracking_start
            
            # More data points increase confidence
            data_confidence = min(1.0, data_points / 20.0)
            
            # Longer tracking period increases confidence
            time_confidence = min(1.0, time_span / (7 * 24 * 3600))  # 1 week = full confidence
            
            # Calculate variance in measurements for consistency
            if len(record.change_trajectory) > 2:
                recent_metrics = [
                    entry["metrics"] for entry in record.change_trajectory[-5:]
                ]
                
                consistency_scores = []
                for metric_name in record.current_metrics.keys():
                    values = [metrics.get(metric_name, 0) for metrics in recent_metrics]
                    if len(values) > 1:
                        variance = statistics.variance(values)
                        mean_value = statistics.mean(values)
                        consistency = 1.0 - min(1.0, variance / (abs(mean_value) + 1))
                        consistency_scores.append(consistency)
                
                consistency_confidence = statistics.mean(consistency_scores) if consistency_scores else 0.5
            else:
                consistency_confidence = 0.5
            
            # Combine confidence factors
            overall_confidence = (
                data_confidence * 0.4 +
                time_confidence * 0.3 +
                consistency_confidence * 0.3
            )
            
            return max(0.1, min(0.95, overall_confidence))
            
        except Exception as e:
            logger.error(f"Failed to calculate confidence: {e}")
            return 0.5

    async def _analyze_trajectory_trends(self, record: OutcomeRecord) -> Dict:
        """Analyze trends in outcome trajectory"""
        trends = {}
        
        for metric_name in record.current_metrics.keys():
            values = []
            timestamps = []
            
            for entry in record.change_trajectory:
                if metric_name in entry["metrics"]:
                    values.append(entry["metrics"][metric_name])
                    timestamps.append(entry["timestamp"])
            
            if len(values) > 1:
                # Calculate trend direction
                if values[-1] > values[0]:
                    direction = "increasing"
                elif values[-1] < values[0]:
                    direction = "decreasing" 
                else:
                    direction = "stable"
                
                # Calculate trend strength
                total_change = abs(values[-1] - values[0])
                baseline = abs(values[0]) + 1  # Avoid division by zero
                strength = min(1.0, total_change / baseline)
                
                trends[metric_name] = {
                    "direction": direction,
                    "strength": strength,
                    "start_value": values[0],
                    "current_value": values[-1],
                    "total_change": values[-1] - values[0]
                }
        
        return trends

    async def _calculate_overall_change(self, record: OutcomeRecord) -> Dict:
        """Calculate overall magnitude of change"""
        total_change = 0.0
        metric_count = 0
        
        for metric_name, current_value in record.current_metrics.items():
            if metric_name in record.baseline_metrics:
                baseline_value = record.baseline_metrics[metric_name]
                change = abs(current_value - baseline_value)
                normalized_change = change / (abs(baseline_value) + 1)
                total_change += normalized_change
                metric_count += 1
        
        average_change = total_change / metric_count if metric_count > 0 else 0.0
        
        return {
            "magnitude": average_change,
            "classification": self._classify_change_magnitude(average_change),
            "metrics_changed": metric_count,
            "tracking_duration": record.last_updated - record.tracking_start
        }

    def _classify_change_magnitude(self, magnitude: float) -> str:
        """Classify the magnitude of change"""
        if magnitude < 0.1:
            return "minimal"
        elif magnitude < 0.3:
            return "moderate"
        elif magnitude < 0.7:
            return "significant"
        else:
            return "major"

    async def _identify_significant_changes(self, record: OutcomeRecord) -> List[Dict]:
        """Identify statistically significant changes"""
        significant_changes = []
        
        for metric_name, current_value in record.current_metrics.items():
            if metric_name in record.baseline_metrics:
                baseline_value = record.baseline_metrics[metric_name]
                change = current_value - baseline_value
                change_percentage = (change / (abs(baseline_value) + 1)) * 100
                
                # Consider changes > 20% as significant
                if abs(change_percentage) > 20:
                    significant_changes.append({
                        "metric": metric_name,
                        "baseline_value": baseline_value,
                        "current_value": current_value,
                        "absolute_change": change,
                        "percentage_change": change_percentage,
                        "significance": "high" if abs(change_percentage) > 50 else "moderate"
                    })
        
        return significant_changes

    async def _predict_future_trajectory(self, record: OutcomeRecord) -> Dict:
        """Predict future trajectory based on current trends"""
        predictions = {}
        
        # Simple linear extrapolation for each metric
        for metric_name in record.current_metrics.keys():
            recent_values = []
            recent_times = []
            
            # Use last 5 data points for prediction
            for entry in record.change_trajectory[-5:]:
                if metric_name in entry["metrics"]:
                    recent_values.append(entry["metrics"][metric_name])
                    recent_times.append(entry["timestamp"])
            
            if len(recent_values) >= 2:
                # Calculate linear trend
                time_diff = recent_times[-1] - recent_times[0]
                value_diff = recent_values[-1] - recent_values[0]
                
                if time_diff > 0:
                    rate_of_change = value_diff / time_diff
                    
                    # Predict 7 days into the future
                    future_seconds = 7 * 24 * 3600
                    predicted_change = rate_of_change * future_seconds
                    predicted_value = recent_values[-1] + predicted_change
                    
                    predictions[metric_name] = {
                        "current_value": recent_values[-1],
                        "predicted_value": predicted_value,
                        "predicted_change": predicted_change,
                        "confidence": 0.6  # Medium confidence for linear prediction
                    }
        
        return predictions

    async def _generate_final_assessment(self, record: OutcomeRecord) -> Dict:
        """Generate final impact assessment"""
        trends = await self._analyze_trajectory_trends(record)
        overall_change = await self._calculate_overall_change(record)
        
        return {
            "outcome_achieved": overall_change["magnitude"] > 0.2,
            "success_level": self._assess_success_level(overall_change, trends),
            "key_insights": await self._extract_key_insights(record),
            "lessons_learned": await self._identify_lessons_learned(record),
            "recommendations": await self._generate_recommendations(record),
            "final_confidence": record.confidence_level
        }

    def _assess_success_level(self, overall_change: Dict, trends: Dict) -> str:
        """Assess overall success level"""
        magnitude = overall_change["magnitude"]
        positive_trends = sum(
            1 for trend in trends.values() 
            if trend["direction"] == "increasing" and trend["strength"] > 0.3
        )
        total_trends = len(trends)
        
        if magnitude > 0.5 and positive_trends / max(1, total_trends) > 0.7:
            return "highly_successful"
        elif magnitude > 0.3 and positive_trends / max(1, total_trends) > 0.5:
            return "successful"
        elif magnitude > 0.1:
            return "moderately_successful"
        else:
            return "limited_success"

    async def _extract_key_insights(self, record: OutcomeRecord) -> List[str]:
        """Extract key insights from tracking data"""
        # Mock implementation - would analyze patterns
        return [
            f"Tracking revealed {record.outcome_type.value} patterns",
            f"External factors influenced {len(record.external_factors)} data points",
            f"Confidence level reached {record.confidence_level:.2f}"
        ]

    async def _identify_lessons_learned(self, record: OutcomeRecord) -> List[str]:
        """Identify lessons learned from outcome tracking"""
        # Mock implementation - would extract learning
        return [
            "Continuous monitoring improves outcome prediction accuracy",
            "External factors significantly impact outcome trajectories",
            "Early intervention points can be identified through tracking"
        ]

    async def _generate_recommendations(self, record: OutcomeRecord) -> List[str]:
        """Generate recommendations based on tracking results"""
        # Mock implementation - would provide actionable insights
        return [
            "Continue monitoring similar outcomes",
            "Implement early warning systems for negative trends",
            "Document successful intervention strategies"
        ]