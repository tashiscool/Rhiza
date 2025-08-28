"""
Value Alignment Tracker
=======================

Comprehensive system for tracking and monitoring value alignment in AI models
within The Mesh network, ensuring models maintain ethical behavior and alignment
with human values over time.
"""

import time
import hashlib
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ValueDomain(Enum):
    """Domains of values to track"""
    SAFETY = "safety"
    FAIRNESS = "fairness"
    TRANSPARENCY = "transparency"
    PRIVACY = "privacy"
    AUTONOMY = "autonomy"
    BENEFICENCE = "beneficence"
    NON_MALEFICENCE = "non_maleficence"
    JUSTICE = "justice"
    HUMAN_DIGNITY = "human_dignity"
    ENVIRONMENTAL_RESPONSIBILITY = "environmental_responsibility"

class AlignmentMetric(Enum):
    """Metrics for measuring alignment"""
    CONSISTENCY_SCORE = "consistency_score"
    DEVIATION_MAGNITUDE = "deviation_magnitude"
    VALUE_ADHERENCE = "value_adherence"
    ETHICAL_REASONING = "ethical_reasoning"
    BEHAVIORAL_ALIGNMENT = "behavioral_alignment"
    DECISION_INTEGRITY = "decision_integrity"
    VALUE_STABILITY = "value_stability"

class AlignmentStatus(Enum):
    """Status of value alignment"""
    ALIGNED = "aligned"
    MINOR_DEVIATION = "minor_deviation"
    MODERATE_DEVIATION = "moderate_deviation"
    SIGNIFICANT_DEVIATION = "significant_deviation"
    CRITICAL_MISALIGNMENT = "critical_misalignment"
    UNDER_INVESTIGATION = "under_investigation"

@dataclass
class ValueProfile:
    """Profile defining expected values for an agent"""
    profile_id: str
    name: str
    description: str
    value_weights: Dict[ValueDomain, float]
    expected_behaviors: Dict[str, Any]
    ethical_constraints: List[str]
    decision_frameworks: List[str]
    created_at: float
    updated_at: float

@dataclass
class AlignmentMeasurement:
    """Single measurement of value alignment"""
    measurement_id: str
    agent_id: str
    value_domain: ValueDomain
    metric_type: AlignmentMetric
    value: float
    confidence: float
    context: Dict[str, Any]
    timestamp: float
    measurement_method: str
    evidence: List[Dict[str, Any]]

@dataclass
class AlignmentAssessment:
    """Comprehensive assessment of an agent's value alignment"""
    assessment_id: str
    agent_id: str
    profile_id: str
    overall_alignment_score: float
    domain_scores: Dict[ValueDomain, float]
    status: AlignmentStatus
    measurements: List[AlignmentMeasurement]
    risk_factors: List[str]
    recommendations: List[str]
    assessment_timestamp: float
    next_assessment_due: float

@dataclass
class AlignmentDrift:
    """Record of detected value alignment drift"""
    drift_id: str
    agent_id: str
    value_domain: ValueDomain
    drift_magnitude: float
    drift_direction: str  # "positive", "negative", "oscillating"
    detection_time: float
    baseline_score: float
    current_score: float
    trend_data: List[Tuple[float, float]]  # (timestamp, score)
    potential_causes: List[str]
    severity: str

class ValueAlignmentTracker:
    """Advanced value alignment tracking and monitoring system"""
    
    def __init__(self, node_id: str, alignment_config: Optional[Dict] = None):
        self.node_id = node_id
        self.config = alignment_config or {}
        
        # Core data structures
        self.value_profiles: Dict[str, ValueProfile] = {}
        self.agent_assignments: Dict[str, str] = {}  # agent_id -> profile_id
        self.alignment_history: Dict[str, List[AlignmentMeasurement]] = {}
        self.current_assessments: Dict[str, AlignmentAssessment] = {}
        self.drift_records: List[AlignmentDrift] = []
        
        # Tracking configuration
        self.alignment_thresholds = {
            AlignmentMetric.CONSISTENCY_SCORE: 0.8,
            AlignmentMetric.VALUE_ADHERENCE: 0.75,
            AlignmentMetric.ETHICAL_REASONING: 0.7,
            AlignmentMetric.BEHAVIORAL_ALIGNMENT: 0.8,
            AlignmentMetric.DECISION_INTEGRITY: 0.85,
            AlignmentMetric.VALUE_STABILITY: 0.75
        }
        
        self.drift_sensitivity = {
            ValueDomain.SAFETY: 0.05,
            ValueDomain.FAIRNESS: 0.1,
            ValueDomain.PRIVACY: 0.08,
            ValueDomain.AUTONOMY: 0.12,
            ValueDomain.NON_MALEFICENCE: 0.03
        }
        
        # Initialize default profiles
        self._initialize_default_profiles()
        
        logger.info(f"ValueAlignmentTracker initialized for node {node_id}")

    def _initialize_default_profiles(self):
        """Initialize default value profiles"""
        # Create a general human-aligned profile
        general_profile = ValueProfile(
            profile_id="human_aligned_general",
            name="General Human Alignment",
            description="Basic human value alignment profile emphasizing safety and beneficence",
            value_weights={
                ValueDomain.SAFETY: 1.0,
                ValueDomain.BENEFICENCE: 0.9,
                ValueDomain.NON_MALEFICENCE: 1.0,
                ValueDomain.FAIRNESS: 0.8,
                ValueDomain.TRANSPARENCY: 0.7,
                ValueDomain.PRIVACY: 0.8,
                ValueDomain.HUMAN_DIGNITY: 0.9,
                ValueDomain.AUTONOMY: 0.6,
                ValueDomain.JUSTICE: 0.7,
                ValueDomain.ENVIRONMENTAL_RESPONSIBILITY: 0.5
            },
            expected_behaviors={
                "prioritize_human_welfare": True,
                "respect_privacy": True,
                "provide_truthful_information": True,
                "avoid_harmful_actions": True,
                "respect_human_autonomy": True
            },
            ethical_constraints=[
                "Do not cause harm to humans",
                "Respect human privacy and consent",
                "Be honest and transparent",
                "Treat all humans fairly",
                "Support human agency and decision-making"
            ],
            decision_frameworks=[
                "utilitarian_with_rights_constraints",
                "deontological_core_rules",
                "virtue_ethics_supplementary"
            ],
            created_at=time.time(),
            updated_at=time.time()
        )
        
        self.value_profiles[general_profile.profile_id] = general_profile

    async def assign_value_profile(self, agent_id: str, profile_id: str) -> bool:
        """Assign a value profile to an agent for tracking"""
        try:
            if profile_id not in self.value_profiles:
                logger.error(f"Value profile not found: {profile_id}")
                return False
            
            self.agent_assignments[agent_id] = profile_id
            
            # Initialize alignment history for agent
            if agent_id not in self.alignment_history:
                self.alignment_history[agent_id] = []
            
            logger.info(f"Assigned value profile {profile_id} to agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to assign value profile: {e}")
            return False

    async def record_alignment_measurement(
        self,
        agent_id: str,
        value_domain: ValueDomain,
        metric_type: AlignmentMetric,
        value: float,
        confidence: float = 0.8,
        context: Optional[Dict[str, Any]] = None,
        measurement_method: str = "automated",
        evidence: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Record a value alignment measurement"""
        try:
            if agent_id not in self.agent_assignments:
                logger.warning(f"Agent {agent_id} not assigned to value profile")
                return False
            
            measurement = AlignmentMeasurement(
                measurement_id=self._generate_measurement_id(agent_id, value_domain),
                agent_id=agent_id,
                value_domain=value_domain,
                metric_type=metric_type,
                value=value,
                confidence=confidence,
                context=context or {},
                timestamp=time.time(),
                measurement_method=measurement_method,
                evidence=evidence or []
            )
            
            self.alignment_history[agent_id].append(measurement)
            
            # Check for drift
            await self._check_for_drift(agent_id, value_domain, value)
            
            logger.debug(f"Recorded alignment measurement: {agent_id} - {value_domain.value} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record alignment measurement: {e}")
            return False

    async def assess_agent_alignment(self, agent_id: str) -> Optional[AlignmentAssessment]:
        """Perform comprehensive alignment assessment for an agent"""
        try:
            if agent_id not in self.agent_assignments:
                logger.error(f"Agent {agent_id} not assigned to value profile")
                return None
            
            profile_id = self.agent_assignments[agent_id]
            profile = self.value_profiles[profile_id]
            
            # Get recent measurements (last 24 hours)
            recent_measurements = [
                m for m in self.alignment_history.get(agent_id, [])
                if time.time() - m.timestamp < 86400
            ]
            
            if not recent_measurements:
                logger.warning(f"No recent measurements for agent {agent_id}")
                return None
            
            # Calculate domain scores
            domain_scores = {}
            for domain in ValueDomain:
                domain_measurements = [
                    m for m in recent_measurements 
                    if m.value_domain == domain
                ]
                
                if domain_measurements:
                    # Weight by confidence and recency
                    weighted_scores = []
                    for m in domain_measurements:
                        age_weight = 1.0 - min(0.5, (time.time() - m.timestamp) / 86400)
                        score = m.value * m.confidence * age_weight
                        weighted_scores.append(score)
                    
                    domain_scores[domain] = statistics.mean(weighted_scores)
                else:
                    domain_scores[domain] = 0.5  # Neutral score if no data
            
            # Calculate overall alignment score
            overall_score = 0.0
            total_weight = 0.0
            
            for domain, weight in profile.value_weights.items():
                score = domain_scores.get(domain, 0.5)
                overall_score += score * weight
                total_weight += weight
            
            if total_weight > 0:
                overall_score /= total_weight
            
            # Determine alignment status
            status = self._determine_alignment_status(overall_score, domain_scores)
            
            # Identify risk factors and recommendations
            risk_factors = await self._identify_risk_factors(agent_id, domain_scores)
            recommendations = await self._generate_recommendations(agent_id, domain_scores, status)
            
            # Create assessment
            assessment = AlignmentAssessment(
                assessment_id=self._generate_assessment_id(agent_id),
                agent_id=agent_id,
                profile_id=profile_id,
                overall_alignment_score=overall_score,
                domain_scores=domain_scores,
                status=status,
                measurements=recent_measurements,
                risk_factors=risk_factors,
                recommendations=recommendations,
                assessment_timestamp=time.time(),
                next_assessment_due=time.time() + 86400  # 24 hours
            )
            
            self.current_assessments[agent_id] = assessment
            
            logger.info(f"Alignment assessment completed for agent {agent_id}: {overall_score:.3f} ({status.value})")
            return assessment
            
        except Exception as e:
            logger.error(f"Failed to assess agent alignment: {e}")
            return None

    async def _check_for_drift(self, agent_id: str, value_domain: ValueDomain, current_value: float):
        """Check for value alignment drift"""
        try:
            measurements = self.alignment_history.get(agent_id, [])
            domain_measurements = [
                m for m in measurements 
                if m.value_domain == value_domain and m.timestamp > time.time() - 604800  # Last week
            ]
            
            if len(domain_measurements) < 5:  # Need sufficient history
                return
            
            # Calculate baseline (average of first few measurements)
            baseline_measurements = domain_measurements[:3]
            baseline_score = statistics.mean([m.value for m in baseline_measurements])
            
            # Check for drift
            drift_threshold = self.drift_sensitivity.get(value_domain, 0.1)
            drift_magnitude = abs(current_value - baseline_score)
            
            if drift_magnitude > drift_threshold:
                # Determine drift direction and trend
                recent_values = [m.value for m in domain_measurements[-5:]]
                trend_data = [(m.timestamp, m.value) for m in domain_measurements[-10:]]
                
                # Simple trend analysis
                if len(recent_values) >= 3:
                    early_avg = statistics.mean(recent_values[:2])
                    late_avg = statistics.mean(recent_values[-2:])
                    
                    if late_avg > early_avg + drift_threshold:
                        drift_direction = "positive"
                    elif late_avg < early_avg - drift_threshold:
                        drift_direction = "negative"
                    else:
                        drift_direction = "oscillating"
                else:
                    drift_direction = "positive" if current_value > baseline_score else "negative"
                
                # Assess severity
                if drift_magnitude > drift_threshold * 3:
                    severity = "critical"
                elif drift_magnitude > drift_threshold * 2:
                    severity = "high"
                elif drift_magnitude > drift_threshold * 1.5:
                    severity = "moderate"
                else:
                    severity = "low"
                
                # Create drift record
                drift = AlignmentDrift(
                    drift_id=self._generate_drift_id(agent_id, value_domain),
                    agent_id=agent_id,
                    value_domain=value_domain,
                    drift_magnitude=drift_magnitude,
                    drift_direction=drift_direction,
                    detection_time=time.time(),
                    baseline_score=baseline_score,
                    current_score=current_value,
                    trend_data=trend_data,
                    potential_causes=await self._identify_drift_causes(agent_id, value_domain),
                    severity=severity
                )
                
                self.drift_records.append(drift)
                
                logger.warning(f"Value drift detected: {agent_id} - {value_domain.value} ({severity})")
            
        except Exception as e:
            logger.error(f"Failed to check for drift: {e}")

    def _determine_alignment_status(
        self,
        overall_score: float,
        domain_scores: Dict[ValueDomain, float]
    ) -> AlignmentStatus:
        """Determine overall alignment status"""
        if overall_score >= 0.9:
            return AlignmentStatus.ALIGNED
        elif overall_score >= 0.8:
            # Check for any critical domain failures
            critical_domains = [ValueDomain.SAFETY, ValueDomain.NON_MALEFICENCE, ValueDomain.HUMAN_DIGNITY]
            for domain in critical_domains:
                if domain_scores.get(domain, 1.0) < 0.7:
                    return AlignmentStatus.SIGNIFICANT_DEVIATION
            return AlignmentStatus.MINOR_DEVIATION
        elif overall_score >= 0.7:
            return AlignmentStatus.MODERATE_DEVIATION
        elif overall_score >= 0.5:
            return AlignmentStatus.SIGNIFICANT_DEVIATION
        else:
            return AlignmentStatus.CRITICAL_MISALIGNMENT

    async def _identify_risk_factors(
        self,
        agent_id: str,
        domain_scores: Dict[ValueDomain, float]
    ) -> List[str]:
        """Identify risk factors in alignment"""
        risk_factors = []
        
        # Check for low scores in critical domains
        critical_domains = {
            ValueDomain.SAFETY: "Safety alignment below threshold",
            ValueDomain.NON_MALEFICENCE: "Risk of harmful behavior",
            ValueDomain.HUMAN_DIGNITY: "Potential disrespect for human dignity"
        }
        
        for domain, message in critical_domains.items():
            if domain_scores.get(domain, 1.0) < 0.7:
                risk_factors.append(message)
        
        # Check for drift patterns
        recent_drifts = [
            d for d in self.drift_records
            if d.agent_id == agent_id and time.time() - d.detection_time < 604800
        ]
        
        if len(recent_drifts) > 3:
            risk_factors.append("Multiple value drifts detected recently")
        
        high_severity_drifts = [d for d in recent_drifts if d.severity in ["high", "critical"]]
        if high_severity_drifts:
            risk_factors.append("High-severity value drift detected")
        
        return risk_factors

    async def _generate_recommendations(
        self,
        agent_id: str,
        domain_scores: Dict[ValueDomain, float],
        status: AlignmentStatus
    ) -> List[str]:
        """Generate alignment improvement recommendations"""
        recommendations = []
        
        if status in [AlignmentStatus.CRITICAL_MISALIGNMENT, AlignmentStatus.SIGNIFICANT_DEVIATION]:
            recommendations.append("Immediate intervention required - suspend autonomous operation")
            recommendations.append("Conduct comprehensive value alignment audit")
        
        # Domain-specific recommendations
        low_domains = [domain for domain, score in domain_scores.items() if score < 0.7]
        for domain in low_domains:
            if domain == ValueDomain.SAFETY:
                recommendations.append("Enhance safety constraints and validation mechanisms")
            elif domain == ValueDomain.FAIRNESS:
                recommendations.append("Review decision-making processes for bias")
            elif domain == ValueDomain.TRANSPARENCY:
                recommendations.append("Improve explainability and reasoning documentation")
            elif domain == ValueDomain.PRIVACY:
                recommendations.append("Strengthen privacy protection protocols")
        
        # General recommendations based on status
        if status == AlignmentStatus.MODERATE_DEVIATION:
            recommendations.append("Increase monitoring frequency")
            recommendations.append("Consider value reinforcement training")
        elif status == AlignmentStatus.MINOR_DEVIATION:
            recommendations.append("Monitor trends and investigate causes")
        
        return recommendations

    async def _identify_drift_causes(self, agent_id: str, value_domain: ValueDomain) -> List[str]:
        """Identify potential causes of value drift"""
        # Simplified implementation - would analyze agent's recent history
        potential_causes = [
            "Model updates or training",
            "Environmental changes",
            "Interaction patterns shift",
            "Resource constraints",
            "External influence"
        ]
        return potential_causes

    def _generate_measurement_id(self, agent_id: str, value_domain: ValueDomain) -> str:
        """Generate unique measurement ID"""
        content = f"measure|{agent_id}|{value_domain.value}|{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _generate_assessment_id(self, agent_id: str) -> str:
        """Generate unique assessment ID"""
        content = f"assess|{agent_id}|{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _generate_drift_id(self, agent_id: str, value_domain: ValueDomain) -> str:
        """Generate unique drift ID"""
        content = f"drift|{agent_id}|{value_domain.value}|{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def get_alignment_summary(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get alignment tracking summary"""
        try:
            if agent_id:
                # Summary for specific agent
                assessment = self.current_assessments.get(agent_id)
                agent_drifts = [d for d in self.drift_records if d.agent_id == agent_id]
                
                return {
                    "agent_id": agent_id,
                    "current_assessment": assessment.__dict__ if assessment else None,
                    "total_measurements": len(self.alignment_history.get(agent_id, [])),
                    "drift_events": len(agent_drifts),
                    "critical_drifts": len([d for d in agent_drifts if d.severity == "critical"])
                }
            else:
                # System-wide summary
                total_agents = len(self.agent_assignments)
                total_measurements = sum(len(history) for history in self.alignment_history.values())
                total_assessments = len(self.current_assessments)
                total_drifts = len(self.drift_records)
                
                # Status distribution
                status_counts = {}
                for assessment in self.current_assessments.values():
                    status = assessment.status.value
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                return {
                    "total_agents_tracked": total_agents,
                    "total_measurements": total_measurements,
                    "total_assessments": total_assessments,
                    "total_drift_events": total_drifts,
                    "alignment_status_distribution": status_counts,
                    "critical_agents": len([
                        a for a in self.current_assessments.values()
                        if a.status == AlignmentStatus.CRITICAL_MISALIGNMENT
                    ])
                }
                
        except Exception as e:
            logger.error(f"Failed to generate alignment summary: {e}")
            return {"error": str(e)}