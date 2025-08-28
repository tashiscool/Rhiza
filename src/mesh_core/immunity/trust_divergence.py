"""
Trust Divergence Monitor - Tracks trust pattern deviations

Monitors trust relationships and scoring patterns to detect when nodes
deviate from expected behavior, indicating potential manipulation or corruption.
"""

import asyncio
import logging
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

class DivergenceType(Enum):
    """Types of trust divergence that can be detected"""
    SUDDEN_TRUST_DROP = "sudden_trust_drop"
    TRUST_INFLATION = "trust_inflation"
    UNUSUAL_TRUST_PATTERNS = "unusual_trust_patterns"
    TRUST_MANIPULATION = "trust_manipulation"
    SYBIL_ATTACK_INDICATOR = "sybil_attack_indicator"
    ECLIPSE_ATTACK_INDICATOR = "eclipse_attack_indicator"

class DivergenceSeverity(Enum):
    """Severity levels of trust divergence"""
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"

@dataclass
class TrustDivergence:
    """Result of trust divergence analysis"""
    node_id: str
    divergence_type: DivergenceType
    severity: DivergenceSeverity
    confidence_score: float
    divergence_metrics: Dict[str, Any]
    timestamp: datetime
    affected_relationships: List[str]
    recommended_action: str
    false_positive_risk: float

@dataclass
class TrustPattern:
    """Normal trust pattern for a node"""
    node_id: str
    trust_score_mean: float
    trust_score_std: float
    trust_change_rate: float
    relationship_count: int
    trust_reciprocity: float
    last_updated: datetime

class TrustDivergenceMonitor:
    """Monitors trust patterns for signs of divergence and manipulation"""
    
    def __init__(self, trust_ledger: TrustLedger, network_health: NetworkHealthMonitor):
        self.trust_ledger = trust_ledger
        self.network_health = network_health
        # Try to get config, fall back to defaults if not available
        try:
            self.config = get_component_config("mesh_immunity")
            self.trust_drop_threshold = self.config.get("trust_drop_threshold", 0.2)
            self.trust_inflation_threshold = self.config.get("trust_inflation_threshold", 0.3)
            self.pattern_anomaly_threshold = self.config.get("pattern_anomaly_threshold", 2.5)
        except Exception:
            # Use default values if config is not available
            self.config = {}
            self.trust_drop_threshold = 0.2
            self.trust_inflation_threshold = 0.3
            self.pattern_anomaly_threshold = 2.5
        
        # Trust patterns for each node
        self.trust_patterns: Dict[str, TrustPattern] = {}
        
        # Divergence history
        self.divergence_history: List[TrustDivergence] = []
        
        # Statistical models
        self.trust_distribution_model = None
        
        logger.info("Trust divergence monitor initialized")
    
    async def monitor_trust_divergence(self, node_id: str) -> Optional[TrustDivergence]:
        """Monitor a specific node for trust divergence"""
        try:
            # Get current trust state
            current_state = await self._get_current_trust_state(node_id)
            if not current_state:
                return None
            
            # Get or create trust pattern baseline
            baseline = self.trust_patterns.get(node_id)
            if not baseline:
                baseline = await self._create_trust_pattern(node_id)
                self.trust_patterns[node_id] = baseline
            
            # Check for various types of divergence
            divergences = []
            
            # Check for sudden trust drops
            trust_drop = await self._detect_sudden_trust_drop(node_id, current_state, baseline)
            if trust_drop:
                divergences.append(trust_drop)
            
            # Check for trust inflation
            trust_inflation = await self._detect_trust_inflation(node_id, current_state, baseline)
            if trust_inflation:
                divergences.append(trust_inflation)
            
            # Check for unusual trust patterns
            pattern_anomaly = await self._detect_unusual_patterns(node_id, current_state, baseline)
            if pattern_anomaly:
                divergences.append(pattern_anomaly)
            
            # Check for manipulation indicators
            manipulation = await self._detect_trust_manipulation(node_id, current_state, baseline)
            if manipulation:
                divergences.append(manipulation)
            
            # Check for attack indicators
            attack_indicators = await self._detect_attack_indicators(node_id, current_state, baseline)
            if attack_indicators:
                divergences.extend(attack_indicators)
            
            # Combine divergences if multiple types found
            if divergences:
                return await self._combine_divergences(node_id, divergences)
            
            return None
            
        except Exception as e:
            logger.error(f"Error monitoring trust divergence for node {node_id}: {e}")
            return None
    
    async def _detect_sudden_trust_drop(self, node_id: str, current_state: Dict[str, Any], baseline: TrustPattern) -> Optional[TrustDivergence]:
        """Detect sudden drops in trust scores"""
        try:
            # Get recent trust changes
            trust_history = await self.trust_ledger.get_trust_history(node_id, hours=6)
            if len(trust_history) < 5:
                return None
            
            # Calculate recent trust trend
            recent_scores = [entry.trust_score for entry in trust_history[-5:]]
            historical_scores = [entry.trust_score for entry in trust_history[:-5]]
            
            if len(historical_scores) < 3:
                return None
            
            recent_mean = np.mean(recent_scores)
            historical_mean = np.mean(historical_scores)
            
            trust_drop = historical_mean - recent_mean
            
            if trust_drop > self.trust_drop_threshold:
                # Calculate confidence based on drop magnitude
                confidence = min(0.9, trust_drop / (self.trust_drop_threshold * 2))
                
                # Determine severity
                if trust_drop > self.trust_drop_threshold * 2:
                    severity = DivergenceSeverity.CRITICAL
                elif trust_drop > self.trust_drop_threshold * 1.5:
                    severity = DivergenceSeverity.MAJOR
                else:
                    severity = DivergenceSeverity.MODERATE
                
                return TrustDivergence(
                    node_id=node_id,
                    divergence_type=DivergenceType.SUDDEN_TRUST_DROP,
                    severity=severity,
                    confidence_score=confidence,
                    divergence_metrics={
                        "trust_drop": trust_drop,
                        "historical_mean": historical_mean,
                        "recent_mean": recent_mean,
                        "drop_percentage": (trust_drop / historical_mean) * 100
                    },
                    timestamp=datetime.now(),
                    affected_relationships=await self._get_affected_relationships(node_id),
                    recommended_action="Investigate recent interactions and validate trust changes",
                    false_positive_risk=0.2
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting trust drop: {e}")
            return None
    
    async def _detect_trust_inflation(self, node_id: str, current_state: Dict[str, Any], baseline: TrustPattern) -> Optional[TrustDivergence]:
        """Detect suspicious trust score inflation"""
        try:
            # Get recent trust changes
            trust_history = await self.trust_ledger.get_trust_history(node_id, hours=12)
            if len(trust_history) < 10:
                return None
            
            # Calculate trust growth rate
            trust_scores = [entry.trust_score for entry in trust_history]
            trust_changes = np.diff(trust_scores)
            
            # Check for unusual growth patterns
            positive_changes = trust_changes[trust_changes > 0]
            if len(positive_changes) > 0:
                growth_rate = np.mean(positive_changes)
                growth_variance = np.var(positive_changes)
                
                # Check if growth is suspiciously high
                if growth_rate > self.trust_inflation_threshold:
                    # Calculate confidence based on growth pattern
                    confidence = min(0.9, growth_rate / (self.trust_inflation_threshold * 2))
                    
                    # Determine severity
                    if growth_rate > self.trust_inflation_threshold * 2:
                        severity = DivergenceSeverity.MAJOR
                    else:
                        severity = DivergenceSeverity.MODERATE
                    
                    return TrustDivergence(
                        node_id=node_id,
                        divergence_type=DivergenceType.TRUST_INFLATION,
                        severity=severity,
                        confidence_score=confidence,
                        divergence_metrics={
                            "growth_rate": growth_rate,
                            "growth_variance": growth_variance,
                            "positive_changes_count": len(positive_changes),
                            "total_changes": len(trust_changes)
                        },
                        timestamp=datetime.now(),
                        affected_relationships=await self._get_affected_relationships(node_id),
                        recommended_action="Validate trust score increases and investigate relationships",
                        false_positive_risk=0.3
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting trust inflation: {e}")
            return None
    
    async def _detect_unusual_patterns(self, node_id: str, current_state: Dict[str, Any], baseline: TrustPattern) -> Optional[TrustDivergence]:
        """Detect unusual trust patterns that don't fit normal behavior"""
        try:
            # Get recent trust activity
            recent_activity = await self.trust_ledger.get_recent_activity(node_id, hours=24)
            if len(recent_activity) < 10:
                return None
            
            # Calculate pattern metrics
            trust_scores = [entry.trust_score for entry in recent_activity]
            trust_changes = np.diff(trust_scores)
            
            # Check for statistical anomalies
            if len(trust_changes) > 0:
                change_mean = np.mean(trust_changes)
                change_std = np.std(trust_changes)
                
                # Calculate z-scores for recent changes
                if change_std > 0:
                    z_scores = np.abs((trust_changes - change_mean) / change_std)
                    anomaly_count = np.sum(z_scores > self.pattern_anomaly_threshold)
                    
                    if anomaly_count > len(trust_changes) * 0.3:  # More than 30% anomalies
                        confidence = min(0.9, anomaly_count / len(trust_changes))
                        
                        return TrustDivergence(
                            node_id=node_id,
                            divergence_type=DivergenceType.UNUSUAL_TRUST_PATTERNS,
                            severity=DivergenceSeverity.MODERATE,
                            confidence_score=confidence,
                            divergence_metrics={
                                "anomaly_count": anomaly_count,
                                "total_changes": len(trust_changes),
                                "anomaly_percentage": (anomaly_count / len(trust_changes)) * 100,
                                "change_mean": change_mean,
                                "change_std": change_std
                            },
                            timestamp=datetime.now(),
                            affected_relationships=await self._get_affected_relationships(node_id),
                            recommended_action="Investigate unusual trust patterns and validate node behavior",
                            false_positive_risk=0.25
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting unusual patterns: {e}")
            return None
    
    async def _detect_trust_manipulation(self, node_id: str, current_state: Dict[str, Any], baseline: TrustPattern) -> Optional[TrustDivergence]:
        """Detect signs of trust manipulation"""
        try:
            # Get trust relationships
            relationships = await self.trust_ledger.get_node_relationships(node_id)
            if not relationships:
                return None
            
            # Check for suspicious relationship patterns
            manipulation_indicators = []
            
            # Check for circular trust relationships
            circular_trust = await self._detect_circular_trust(node_id, relationships)
            if circular_trust:
                manipulation_indicators.append(circular_trust)
            
            # Check for trust reciprocity anomalies
            reciprocity_anomaly = await self._detect_reciprocity_anomaly(node_id, relationships)
            if reciprocity_anomaly:
                manipulation_indicators.append(reciprocity_anomaly)
            
            # Check for trust score manipulation
            score_manipulation = await self._detect_score_manipulation(node_id, relationships)
            if score_manipulation:
                manipulation_indicators.append(score_manipulation)
            
            if manipulation_indicators:
                # Combine indicators
                total_confidence = sum(indicator["confidence"] for indicator in manipulation_indicators)
                avg_confidence = total_confidence / len(manipulation_indicators)
                
                return TrustDivergence(
                    node_id=node_id,
                    divergence_type=DivergenceType.TRUST_MANIPULATION,
                    severity=DivergenceSeverity.MAJOR if avg_confidence > 0.7 else DivergenceSeverity.MODERATE,
                    confidence_score=avg_confidence,
                    divergence_metrics={
                        "manipulation_indicators": manipulation_indicators,
                        "indicator_count": len(manipulation_indicators)
                    },
                    timestamp=datetime.now(),
                    affected_relationships=await self._get_affected_relationships(node_id),
                    recommended_action="Investigate trust manipulation patterns and validate relationships",
                    false_positive_risk=0.15
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting trust manipulation: {e}")
            return None
    
    async def _detect_attack_indicators(self, node_id: str, current_state: Dict[str, Any], baseline: TrustPattern) -> List[Dict[str, Any]]:
        """Detect indicators of various attack types"""
        indicators = []
        
        try:
            # Check for Sybil attack indicators
            sybil_indicator = await self._detect_sybil_attack(node_id, current_state, baseline)
            if sybil_indicator:
                indicators.append(sybil_indicator)
            
            # Check for Eclipse attack indicators
            eclipse_indicator = await self._detect_eclipse_attack(node_id, current_state, baseline)
            if eclipse_indicator:
                indicators.append(eclipse_indicator)
            
        except Exception as e:
            logger.error(f"Error detecting attack indicators: {e}")
        
        return indicators
    
    async def _detect_sybil_attack(self, node_id: str, current_state: Dict[str, Any], baseline: TrustPattern) -> Optional[Dict[str, Any]]:
        """Detect indicators of Sybil attacks"""
        try:
            # Get network topology around this node
            neighbors = await self.network_health.get_node_neighbors(node_id)
            if not neighbors:
                return None
            
            # Check for suspicious neighbor patterns
            suspicious_patterns = []
            
            # Check for nodes with very similar behavior patterns
            similar_nodes = await self._find_similar_nodes(node_id, neighbors)
            if len(similar_nodes) > 3:  # More than 3 similar nodes is suspicious
                suspicious_patterns.append({
                    "type": "similar_behavior",
                    "similar_node_count": len(similar_nodes),
                    "confidence": min(0.8, len(similar_nodes) / 10)
                })
            
            # Check for trust score clustering
            trust_clustering = await self._detect_trust_clustering(node_id, neighbors)
            if trust_clustering:
                suspicious_patterns.append(trust_clustering)
            
            if suspicious_patterns:
                return {
                    "attack_type": "sybil",
                    "indicators": suspicious_patterns,
                    "confidence": np.mean([p["confidence"] for p in suspicious_patterns])
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting Sybil attack: {e}")
            return None
    
    async def _detect_eclipse_attack(self, node_id: str, current_state: Dict[str, Any], baseline: TrustPattern) -> Optional[Dict[str, Any]]:
        """Detect indicators of Eclipse attacks"""
        try:
            # Get network connectivity
            connectivity = await self.network_health.get_node_connectivity(node_id)
            if not connectivity:
                return None
            
            # Check for isolation patterns
            isolation_indicators = []
            
            # Check if node is becoming isolated
            if connectivity.get("isolation_score", 0) > 0.7:
                isolation_indicators.append({
                    "type": "high_isolation",
                    "isolation_score": connectivity["isolation_score"],
                    "confidence": connectivity["isolation_score"]
                })
            
            # Check for unusual routing patterns
            routing_anomaly = await self._detect_routing_anomaly(node_id, connectivity)
            if routing_anomaly:
                isolation_indicators.append(routing_anomaly)
            
            if isolation_indicators:
                return {
                    "attack_type": "eclipse",
                    "indicators": isolation_indicators,
                    "confidence": np.mean([i["confidence"] for i in isolation_indicators])
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting Eclipse attack: {e}")
            return None
    
    async def _combine_divergences(self, node_id: str, divergences: List[Dict[str, Any]]) -> TrustDivergence:
        """Combine multiple divergence indicators into a single result"""
        if len(divergences) == 1:
            div = divergences[0]
            return TrustDivergence(
                node_id=node_id,
                divergence_type=div["type"],
                severity=div["severity"],
                confidence_score=div["confidence"],
                divergence_metrics=div["metrics"],
                timestamp=datetime.now(),
                affected_relationships=div["affected_relationships"],
                recommended_action=div["recommended_action"],
                false_positive_risk=div["false_positive_risk"]
            )
        
        # Combine multiple divergences
        total_confidence = sum(d["confidence"] for d in divergences)
        avg_confidence = total_confidence / len(divergences)
        
        # Determine overall severity
        max_severity = max(d["severity"] for d in divergences)
        if len(divergences) >= 3 and avg_confidence > 0.7:
            if max_severity == DivergenceSeverity.MODERATE:
                max_severity = DivergenceSeverity.MAJOR
            elif max_severity == DivergenceSeverity.MAJOR:
                max_severity = DivergenceSeverity.CRITICAL
        
        # Combine metrics
        combined_metrics = {
            "divergence_count": len(divergences),
            "divergence_types": [d["type"].value for d in divergences],
            "individual_divergences": [
                {
                    "type": d["type"].value,
                    "severity": d["severity"].value,
                    "confidence": d["confidence"],
                    "metrics": d["metrics"]
                }
                for d in divergences
            ]
        }
        
        # Determine recommended action
        if max_severity == DivergenceSeverity.CRITICAL:
            recommended_action = "Immediate investigation and potential isolation required"
        elif max_severity == DivergenceSeverity.MAJOR:
            recommended_action = "Comprehensive investigation and enhanced monitoring"
        elif max_severity == DivergenceSeverity.MODERATE:
            recommended_action = "Investigate divergence patterns and validate behavior"
        else:
            recommended_action = "Continue monitoring for additional evidence"
        
        # Calculate false positive risk
        false_positive_risk = max(0.05, min(0.4, 0.4 / len(divergences)))
        
        return TrustDivergence(
            node_id=node_id,
            divergence_type=DivergenceType.TRUST_MANIPULATION,  # Generic type for combined
            severity=max_severity,
            confidence_score=avg_confidence,
            divergence_metrics=combined_metrics,
            timestamp=datetime.now(),
            affected_relationships=await self._get_affected_relationships(node_id),
            recommended_action=recommended_action,
            false_positive_risk=false_positive_risk
        )
    
    async def _get_current_trust_state(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get current trust state for a node"""
        try:
            # Get trust information
            trust_info = await self.trust_ledger.get_node_trust_info(node_id)
            
            # Get recent trust activity
            recent_activity = await self.trust_ledger.get_recent_activity(node_id, hours=24)
            
            # Get trust relationships
            relationships = await self.trust_ledger.get_node_relationships(node_id)
            
            return {
                "trust_info": trust_info,
                "recent_activity": recent_activity,
                "relationships": relationships
            }
        except Exception as e:
            logger.error(f"Error getting trust state: {e}")
            return None
    
    async def _create_trust_pattern(self, node_id: str) -> TrustPattern:
        """Create trust pattern baseline for a node"""
        try:
            # Get historical trust data
            historical_data = await self.trust_ledger.get_behavioral_history(node_id, days=14)
            
            if not historical_data:
                # Create default pattern
                return TrustPattern(
                    node_id=node_id,
                    trust_score_mean=0.5,
                    trust_score_std=0.1,
                    trust_change_rate=0.0,
                    relationship_count=0,
                    trust_reciprocity=0.5,
                    last_updated=datetime.now()
                )
            
            # Calculate pattern metrics
            trust_scores = [entry.trust_score for entry in historical_data]
            trust_changes = np.diff(trust_scores)
            
            return TrustPattern(
                node_id=node_id,
                trust_score_mean=np.mean(trust_scores),
                trust_score_std=np.std(trust_scores),
                trust_change_rate=np.mean(trust_changes) if len(trust_changes) > 0 else 0.0,
                relationship_count=len(await self.trust_ledger.get_node_relationships(node_id)),
                trust_reciprocity=0.5,  # Placeholder - would calculate actual reciprocity
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error creating trust pattern: {e}")
            # Return default pattern
            return TrustPattern(
                node_id=node_id,
                trust_score_mean=0.5,
                trust_score_std=0.1,
                trust_change_rate=0.0,
                relationship_count=0,
                trust_reciprocity=0.5,
                last_updated=datetime.now()
            )
    
    async def _get_affected_relationships(self, node_id: str) -> List[str]:
        """Get list of relationships affected by trust divergence"""
        try:
            relationships = await self.trust_ledger.get_node_relationships(node_id)
            return [rel.target_node_id for rel in relationships]
        except Exception as e:
            logger.error(f"Error getting affected relationships: {e}")
            return []
    
    # Placeholder methods for advanced detection
    async def _detect_circular_trust(self, node_id: str, relationships: List[Any]) -> Optional[Dict[str, Any]]:
        """Detect circular trust relationships"""
        # Implementation would check for trust cycles
        return None
    
    async def _detect_reciprocity_anomaly(self, node_id: str, relationships: List[Any]) -> Optional[Dict[str, Any]]:
        """Detect anomalies in trust reciprocity"""
        # Implementation would analyze trust reciprocity patterns
        return None
    
    async def _detect_score_manipulation(self, node_id: str, relationships: List[Any]) -> Optional[Dict[str, Any]]:
        """Detect manipulation of trust scores"""
        # Implementation would check for suspicious score changes
        return None
    
    async def _find_similar_nodes(self, node_id: str, neighbors: List[str]) -> List[str]:
        """Find nodes with similar behavior patterns"""
        # Implementation would compare behavioral patterns
        return []
    
    async def _detect_trust_clustering(self, node_id: str, neighbors: List[str]) -> Optional[Dict[str, Any]]:
        """Detect clustering in trust scores"""
        # Implementation would analyze trust score distributions
        return None
    
    async def _detect_routing_anomaly(self, node_id: str, connectivity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect anomalies in network routing"""
        # Implementation would analyze routing patterns
        return None
    
    async def update_trust_pattern(self, node_id: str):
        """Update trust pattern for a node"""
        try:
            pattern = await self._create_trust_pattern(node_id)
            self.trust_patterns[node_id] = pattern
            logger.info(f"Updated trust pattern for node {node_id}")
        except Exception as e:
            logger.error(f"Error updating trust pattern: {e}")
    
    def get_divergence_history(self, node_id: Optional[str] = None) -> List[TrustDivergence]:
        """Get trust divergence history"""
        if node_id:
            return [d for d in self.divergence_history if d.node_id == node_id]
        return self.divergence_history
    
    def get_divergence_summary(self) -> Dict[str, Any]:
        """Get summary of trust divergence activity"""
        if not self.divergence_history:
            return {"total_divergences": 0, "active_divergences": 0}
        
        # Count by severity
        severity_counts = {}
        for divergence in self.divergence_history:
            severity = divergence.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count active divergences (last 24 hours)
        recent_divergences = [
            d for d in self.divergence_history 
            if d.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        return {
            "total_divergences": len(self.divergence_history),
            "active_divergences": len(recent_divergences),
            "severity_breakdown": severity_counts,
            "recent_divergences": len(recent_divergences)
        }
