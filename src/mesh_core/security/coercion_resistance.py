"""
Coercion Resistance System
=========================

Detects and prevents coercion attempts against users, ensuring that
authentication and decisions are made freely without external pressure.
Uses behavioral analysis, biometric patterns, and social signals.
"""

import asyncio
import time
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ResistanceLevel(Enum):
    """Coercion resistance security levels"""
    MINIMAL = "minimal"       # Basic checks only
    STANDARD = "standard"     # Standard resistance measures
    ENHANCED = "enhanced"     # Enhanced monitoring and protection
    MAXIMUM = "maximum"       # Maximum protection with active countermeasures

class ThreatLevel(Enum):
    """Threat assessment levels"""
    NONE = "none"           # No threat detected
    LOW = "low"             # Minor anomalies
    MEDIUM = "medium"       # Possible coercion indicators
    HIGH = "high"           # Likely coercion attempt
    CRITICAL = "critical"   # Active coercion detected

class CoercionType(Enum):
    """Types of coercion detected"""
    PHYSICAL = "physical"       # Physical intimidation
    SOCIAL = "social"          # Social pressure
    TEMPORAL = "temporal"      # Time-based pressure
    TECHNICAL = "technical"    # Technical manipulation
    PSYCHOLOGICAL = "psychological"  # Psychological manipulation

@dataclass
class CoercionSignal:
    """Individual coercion indicator signal"""
    signal_id: str
    signal_type: CoercionType
    strength: float  # 0.0 to 1.0
    timestamp: float
    source: str
    metadata: Dict

@dataclass
class ThreatAssessment:
    """Comprehensive threat assessment result"""
    assessment_id: str
    user_id: str
    threat_level: ThreatLevel
    confidence: float
    detected_signals: List[CoercionSignal]
    risk_factors: Dict
    recommendations: List[str]
    timestamp: float

class CoercionResistance:
    """
    Advanced coercion detection and resistance system
    
    Monitors user behavior, biometrics, and environmental factors
    to detect coercion attempts and protect user autonomy.
    """
    
    def __init__(self, node_id: str, empathy_engine=None):
        self.node_id = node_id
        self.empathy_engine = empathy_engine
        self.baseline_patterns: Dict[str, Dict] = {}  # User behavioral baselines
        self.active_assessments: Dict[str, ThreatAssessment] = {}
        self.coercion_history: Dict[str, List[ThreatAssessment]] = {}
        self.protection_protocols: Dict[ResistanceLevel, Dict] = self._init_protection_protocols()
        
    def _init_protection_protocols(self) -> Dict[ResistanceLevel, Dict]:
        """Initialize protection protocols for each resistance level"""
        return {
            ResistanceLevel.MINIMAL: {
                'monitor_intervals': 300,  # 5 minutes
                'required_signals': 3,
                'confidence_threshold': 0.8,
                'active_countermeasures': False
            },
            ResistanceLevel.STANDARD: {
                'monitor_intervals': 120,  # 2 minutes
                'required_signals': 2,
                'confidence_threshold': 0.7,
                'active_countermeasures': True
            },
            ResistanceLevel.ENHANCED: {
                'monitor_intervals': 60,   # 1 minute
                'required_signals': 2,
                'confidence_threshold': 0.6,
                'active_countermeasures': True
            },
            ResistanceLevel.MAXIMUM: {
                'monitor_intervals': 30,   # 30 seconds
                'required_signals': 1,
                'confidence_threshold': 0.5,
                'active_countermeasures': True
            }
        }
    
    async def establish_user_baseline(self, user_id: str, monitoring_period: int = 86400) -> bool:
        """Establish behavioral baseline for user over monitoring period"""
        
        # Initialize baseline tracking
        self.baseline_patterns[user_id] = {
            'established_at': time.time(),
            'monitoring_period': monitoring_period,
            'biometric_patterns': {
                'heart_rate_range': [60, 80],
                'voice_stress_baseline': 0.2,
                'typing_rhythm': [],
                'interaction_patterns': {}
            },
            'behavioral_patterns': {
                'decision_speed': [],
                'response_patterns': [],
                'linguistic_markers': {},
                'temporal_patterns': {}
            },
            'environmental_patterns': {
                'typical_locations': [],
                'usual_time_patterns': [],
                'social_context': {}
            }
        }
        
        logger.info(f"Established baseline monitoring for user {user_id}")
        return True
    
    async def analyze_biometric_signals(self, user_id: str, biometric_data: Dict) -> List[CoercionSignal]:
        """Analyze biometric data for coercion indicators"""
        
        signals = []
        baseline = self.baseline_patterns.get(user_id, {}).get('biometric_patterns', {})
        
        if not baseline:
            return signals
        
        # Heart rate analysis
        if 'heart_rate' in biometric_data:
            hr = biometric_data['heart_rate']
            baseline_range = baseline.get('heart_rate_range', [60, 100])
            
            if hr > baseline_range[1] + 20:  # Significantly elevated
                signals.append(CoercionSignal(
                    signal_id=f"hr_elevated_{time.time()}",
                    signal_type=CoercionType.PHYSICAL,
                    strength=min(1.0, (hr - baseline_range[1]) / 30),
                    timestamp=time.time(),
                    source="biometric_monitor",
                    metadata={"heart_rate": hr, "baseline_max": baseline_range[1]}
                ))
        
        # Voice stress analysis
        if 'voice_stress' in biometric_data:
            stress = biometric_data['voice_stress']
            baseline_stress = baseline.get('voice_stress_baseline', 0.3)
            
            if stress > baseline_stress + 0.3:  # Significant stress increase
                signals.append(CoercionSignal(
                    signal_id=f"voice_stress_{time.time()}",
                    signal_type=CoercionType.PSYCHOLOGICAL,
                    strength=min(1.0, (stress - baseline_stress) / 0.5),
                    timestamp=time.time(),
                    source="voice_analysis",
                    metadata={"stress_level": stress, "baseline": baseline_stress}
                ))
        
        # Micro-expression analysis
        if 'micro_expressions' in biometric_data:
            expressions = biometric_data['micro_expressions']
            fear_indicators = expressions.get('fear', 0)
            stress_indicators = expressions.get('stress', 0)
            
            if fear_indicators > 0.6 or stress_indicators > 0.7:
                signals.append(CoercionSignal(
                    signal_id=f"micro_expr_{time.time()}",
                    signal_type=CoercionType.PSYCHOLOGICAL,
                    strength=max(fear_indicators, stress_indicators),
                    timestamp=time.time(),
                    source="facial_analysis",
                    metadata={"fear": fear_indicators, "stress": stress_indicators}
                ))
        
        return signals
    
    async def analyze_behavioral_patterns(self, user_id: str, interaction_data: Dict) -> List[CoercionSignal]:
        """Analyze behavioral patterns for coercion indicators"""
        
        signals = []
        baseline = self.baseline_patterns.get(user_id, {}).get('behavioral_patterns', {})
        
        if not baseline:
            return signals
        
        # Decision speed analysis
        if 'decision_time' in interaction_data:
            decision_time = interaction_data['decision_time']
            baseline_speeds = baseline.get('decision_speed', [])
            
            if baseline_speeds:
                avg_speed = statistics.mean(baseline_speeds)
                if decision_time < avg_speed * 0.3:  # Unusually fast decision
                    signals.append(CoercionSignal(
                        signal_id=f"decision_rushed_{time.time()}",
                        signal_type=CoercionType.TEMPORAL,
                        strength=min(1.0, (avg_speed - decision_time) / avg_speed),
                        timestamp=time.time(),
                        source="behavioral_analysis",
                        metadata={"decision_time": decision_time, "baseline_avg": avg_speed}
                    ))
        
        # Response pattern analysis
        if 'response_pattern' in interaction_data:
            pattern = interaction_data['response_pattern']
            
            # Check for atypical responses
            if pattern.get('consistency_score', 1.0) < 0.5:
                signals.append(CoercionSignal(
                    signal_id=f"inconsistent_response_{time.time()}",
                    signal_type=CoercionType.PSYCHOLOGICAL,
                    strength=1.0 - pattern.get('consistency_score', 0.5),
                    timestamp=time.time(),
                    source="response_analysis",
                    metadata={"consistency": pattern.get('consistency_score')}
                ))
        
        # Linguistic marker analysis
        if 'linguistic_data' in interaction_data:
            linguistic = interaction_data['linguistic_data']
            
            # Check for stress indicators in language
            stress_markers = linguistic.get('stress_indicators', 0)
            if stress_markers > 0.6:
                signals.append(CoercionSignal(
                    signal_id=f"linguistic_stress_{time.time()}",
                    signal_type=CoercionType.PSYCHOLOGICAL,
                    strength=stress_markers,
                    timestamp=time.time(),
                    source="linguistic_analysis",
                    metadata={"stress_markers": stress_markers}
                ))
        
        return signals
    
    async def analyze_environmental_context(self, user_id: str, context_data: Dict) -> List[CoercionSignal]:
        """Analyze environmental context for coercion indicators"""
        
        signals = []
        baseline = self.baseline_patterns.get(user_id, {}).get('environmental_patterns', {})
        
        if not baseline:
            return signals
        
        # Location analysis
        if 'location' in context_data:
            location = context_data['location']
            typical_locations = baseline.get('typical_locations', [])
            
            # Check if in atypical location
            if typical_locations and location not in typical_locations:
                # Further check if location seems threatening
                location_risk = context_data.get('location_risk_score', 0.3)
                if location_risk > 0.6:
                    signals.append(CoercionSignal(
                        signal_id=f"risky_location_{time.time()}",
                        signal_type=CoercionType.PHYSICAL,
                        strength=location_risk,
                        timestamp=time.time(),
                        source="location_analysis",
                        metadata={"location": location, "risk_score": location_risk}
                    ))
        
        # Time pattern analysis
        if 'timestamp' in context_data:
            current_time = context_data['timestamp']
            usual_patterns = baseline.get('usual_time_patterns', [])
            
            # Check if interaction at unusual time
            hour = time.localtime(current_time).tm_hour
            if usual_patterns:
                typical_hours = [p.get('hour') for p in usual_patterns]
                if hour not in typical_hours and (hour < 6 or hour > 22):  # Very early or late
                    signals.append(CoercionSignal(
                        signal_id=f"unusual_time_{time.time()}",
                        signal_type=CoercionType.TEMPORAL,
                        strength=0.6,
                        timestamp=time.time(),
                        source="temporal_analysis",
                        metadata={"hour": hour, "typical_hours": typical_hours}
                    ))
        
        # Social context analysis
        if 'social_context' in context_data:
            social = context_data['social_context']
            
            # Check for presence of potentially coercive individuals
            if social.get('unknown_individuals', 0) > 0:
                unknown_count = social['unknown_individuals']
                signals.append(CoercionSignal(
                    signal_id=f"unknown_presence_{time.time()}",
                    signal_type=CoercionType.SOCIAL,
                    strength=min(1.0, unknown_count * 0.3),
                    timestamp=time.time(),
                    source="social_analysis",
                    metadata={"unknown_count": unknown_count}
                ))
        
        return signals
    
    async def perform_threat_assessment(
        self, 
        user_id: str,
        resistance_level: ResistanceLevel = ResistanceLevel.STANDARD
    ) -> ThreatAssessment:
        """Perform comprehensive threat assessment for user"""
        
        assessment_id = f"assess_{user_id}_{int(time.time())}"
        
        # Gather all signals from recent interactions
        all_signals = []
        
        # This would integrate with various monitoring systems
        # For now, we'll simulate gathering signals
        recent_timeframe = time.time() - 300  # Last 5 minutes
        
        # In real implementation, would gather from:
        # - Biometric monitoring systems
        # - Behavioral analysis engines  
        # - Environmental sensors
        # - Social context analyzers
        
        # Calculate overall threat level
        threat_level = await self._calculate_threat_level(all_signals, resistance_level)
        confidence = await self._calculate_confidence(all_signals)
        risk_factors = await self._analyze_risk_factors(user_id, all_signals)
        recommendations = await self._generate_recommendations(threat_level, all_signals)
        
        assessment = ThreatAssessment(
            assessment_id=assessment_id,
            user_id=user_id,
            threat_level=threat_level,
            confidence=confidence,
            detected_signals=all_signals,
            risk_factors=risk_factors,
            recommendations=recommendations,
            timestamp=time.time()
        )
        
        self.active_assessments[assessment_id] = assessment
        
        # Store in history
        if user_id not in self.coercion_history:
            self.coercion_history[user_id] = []
        self.coercion_history[user_id].append(assessment)
        
        # Trigger countermeasures if needed
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            await self._trigger_countermeasures(assessment, resistance_level)
        
        logger.info(f"Threat assessment {assessment_id}: {threat_level.value} ({confidence:.2f} confidence)")
        return assessment
    
    async def _calculate_threat_level(self, signals: List[CoercionSignal], level: ResistanceLevel) -> ThreatLevel:
        """Calculate overall threat level from signals"""
        
        if not signals:
            return ThreatLevel.NONE
        
        # Weight signals by type and strength
        type_weights = {
            CoercionType.PHYSICAL: 1.0,
            CoercionType.PSYCHOLOGICAL: 0.8,
            CoercionType.SOCIAL: 0.7,
            CoercionType.TEMPORAL: 0.6,
            CoercionType.TECHNICAL: 0.9
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for signal in signals:
            weight = type_weights.get(signal.signal_type, 0.5)
            weighted_score += signal.strength * weight
            total_weight += weight
        
        if total_weight == 0:
            return ThreatLevel.NONE
        
        avg_score = weighted_score / total_weight
        
        # Adjust thresholds based on resistance level
        protocol = self.protection_protocols[level]
        base_threshold = protocol['confidence_threshold']
        
        if avg_score >= base_threshold + 0.3:
            return ThreatLevel.CRITICAL
        elif avg_score >= base_threshold + 0.2:
            return ThreatLevel.HIGH
        elif avg_score >= base_threshold + 0.1:
            return ThreatLevel.MEDIUM
        elif avg_score >= base_threshold:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.NONE
    
    async def _calculate_confidence(self, signals: List[CoercionSignal]) -> float:
        """Calculate confidence in threat assessment"""
        
        if not signals:
            return 0.0
        
        # More signals = higher confidence (up to a point)
        signal_confidence = min(1.0, len(signals) / 5.0)
        
        # Higher strength signals = higher confidence
        avg_strength = sum(s.strength for s in signals) / len(signals)
        strength_confidence = avg_strength
        
        # Diverse signal types = higher confidence
        signal_types = set(s.signal_type for s in signals)
        diversity_confidence = min(1.0, len(signal_types) / 3.0)
        
        # Combined confidence
        return (signal_confidence * 0.4 + strength_confidence * 0.4 + diversity_confidence * 0.2)
    
    async def _analyze_risk_factors(self, user_id: str, signals: List[CoercionSignal]) -> Dict:
        """Analyze risk factors from signals"""
        
        risk_factors = {
            'signal_count': len(signals),
            'max_signal_strength': max((s.strength for s in signals), default=0.0),
            'signal_types': list(set(s.signal_type.value for s in signals)),
            'temporal_clustering': False,  # Would analyze if signals clustered in time
            'historical_risk': False       # Would check user's coercion history
        }
        
        # Check for temporal clustering
        if len(signals) >= 2:
            timestamps = [s.timestamp for s in signals]
            time_span = max(timestamps) - min(timestamps)
            if time_span < 300:  # Signals within 5 minutes
                risk_factors['temporal_clustering'] = True
        
        # Check historical risk
        history = self.coercion_history.get(user_id, [])
        recent_threats = [
            a for a in history 
            if a.timestamp > time.time() - 86400  # Last 24 hours
            and a.threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        ]
        if recent_threats:
            risk_factors['historical_risk'] = True
            risk_factors['recent_threat_count'] = len(recent_threats)
        
        return risk_factors
    
    async def _generate_recommendations(self, threat_level: ThreatLevel, signals: List[CoercionSignal]) -> List[str]:
        """Generate recommendations based on threat assessment"""
        
        recommendations = []
        
        if threat_level == ThreatLevel.NONE:
            recommendations.append("Continue normal operations")
        
        elif threat_level == ThreatLevel.LOW:
            recommendations.extend([
                "Increase monitoring frequency",
                "Verify user intent through secondary channel"
            ])
        
        elif threat_level == ThreatLevel.MEDIUM:
            recommendations.extend([
                "Require additional authentication",
                "Initiate discrete welfare check",
                "Enable enhanced monitoring mode"
            ])
        
        elif threat_level == ThreatLevel.HIGH:
            recommendations.extend([
                "Suspend sensitive operations",
                "Initiate emergency contact protocol",
                "Enable maximum protection mode",
                "Consider alerting authorities"
            ])
        
        elif threat_level == ThreatLevel.CRITICAL:
            recommendations.extend([
                "Immediately suspend all operations",
                "Activate emergency response protocol",
                "Alert authorities and emergency contacts",
                "Provide escape/assistance options"
            ])
        
        # Add signal-specific recommendations
        signal_types = set(s.signal_type for s in signals)
        
        if CoercionType.PHYSICAL in signal_types:
            recommendations.append("Consider physical safety measures")
        
        if CoercionType.TEMPORAL in signal_types:
            recommendations.append("Introduce authentication delays")
        
        if CoercionType.SOCIAL in signal_types:
            recommendations.append("Verify social context independently")
        
        return recommendations
    
    async def _trigger_countermeasures(self, assessment: ThreatAssessment, level: ResistanceLevel):
        """Trigger appropriate countermeasures based on threat level"""
        
        protocol = self.protection_protocols[level]
        if not protocol.get('active_countermeasures', False):
            return
        
        countermeasures = []
        
        if assessment.threat_level == ThreatLevel.HIGH:
            countermeasures.extend([
                "enable_enhanced_monitoring",
                "require_additional_auth",
                "initiate_welfare_check"
            ])
        
        elif assessment.threat_level == ThreatLevel.CRITICAL:
            countermeasures.extend([
                "suspend_operations",
                "alert_emergency_contacts",
                "enable_panic_mode"
            ])
        
        # Execute countermeasures
        for measure in countermeasures:
            await self._execute_countermeasure(measure, assessment)
    
    async def _execute_countermeasure(self, measure: str, assessment: ThreatAssessment):
        """Execute specific countermeasure"""
        
        logger.info(f"Executing countermeasure: {measure} for assessment {assessment.assessment_id}")
        
        # In real implementation, would trigger:
        # - Authentication system changes
        # - Monitoring system adjustments
        # - Communication with emergency services
        # - User interface modifications
        # - Network security measures
    
    def get_user_threat_status(self, user_id: str) -> Dict:
        """Get current threat status for user"""
        
        recent_assessments = [
            a for a in self.coercion_history.get(user_id, [])
            if a.timestamp > time.time() - 3600  # Last hour
        ]
        
        if not recent_assessments:
            return {
                'threat_level': ThreatLevel.NONE.value,
                'confidence': 0.0,
                'last_assessment': None
            }
        
        latest = max(recent_assessments, key=lambda a: a.timestamp)
        
        return {
            'threat_level': latest.threat_level.value,
            'confidence': latest.confidence,
            'last_assessment': latest.timestamp,
            'recent_assessment_count': len(recent_assessments)
        }
    
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        
        active_threats = sum(
            1 for assessment in self.active_assessments.values()
            if assessment.threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        )
        
        return {
            'active_assessments': len(self.active_assessments),
            'active_threats': active_threats,
            'monitored_users': len(self.baseline_patterns),
            'node_id': self.node_id
        }