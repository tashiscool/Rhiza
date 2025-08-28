"""
Mesh Feedback Analyzer
=====================

Component 10.1: Mesh Degeneration Watchdogs
Analyze feedback loops for distortion

Implements feedback loop analysis, echo chamber detection,
and filter bubble identification.
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


class FeedbackType(Enum):
    """Types of feedback loops"""
    POSITIVE = "positive"                # Amplifying feedback
    NEGATIVE = "negative"                # Dampening feedback
    REINFORCING = "reinforcing"          # Self-reinforcing
    BALANCING = "balancing"              # Self-correcting
    ECHO_CHAMBER = "echo_chamber"        # Echo chamber effect
    FILTER_BUBBLE = "filter_bubble"      # Filter bubble effect


class DistortionLevel(Enum):
    """Levels of feedback distortion"""
    NONE = "none"                        # No distortion detected
    LOW = "low"                          # Minor distortion, monitor
    MODERATE = "moderate"                # Moderate distortion, investigate
    HIGH = "high"                        # Significant distortion, alert
    CRITICAL = "critical"                # Severe distortion, immediate action


@dataclass
class FeedbackLoop:
    """A detected feedback loop"""
    loop_id: str
    feedback_type: FeedbackType
    timestamp: datetime
    
    # Loop characteristics
    nodes_involved: List[str] = field(default_factory=list)
    loop_strength: float = 0.0  # 0.0 to 1.0
    distortion_level: DistortionLevel = DistortionLevel.NONE
    
    # Analysis results
    amplification_factor: float = 1.0  # How much the loop amplifies
    stability_score: float = 0.0  # How stable the loop is
    health_impact: float = 0.0  # Impact on system health
    
    # Context
    loop_context: Dict[str, Any] = field(default_factory=dict)
    detection_method: str = ""
    
    def __post_init__(self):
        if not self.loop_id:
            self.loop_id = self._generate_loop_id()
    
    def _generate_loop_id(self) -> str:
        """Generate unique loop ID"""
        content = f"{self.feedback_type.value}{','.join(self.nodes_involved)}{self.timestamp.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert feedback loop to dictionary"""
        return {
            "loop_id": self.loop_id,
            "feedback_type": self.feedback_type.value,
            "timestamp": self.timestamp.isoformat(),
            "nodes_involved": self.nodes_involved,
            "loop_strength": self.loop_strength,
            "distortion_level": self.distortion_level.value,
            "amplification_factor": self.amplification_factor,
            "stability_score": self.stability_score,
            "health_impact": self.health_impact,
            "loop_context": self.loop_context,
            "detection_method": self.detection_method
        }


@dataclass
class EchoChamber:
    """A detected echo chamber"""
    chamber_id: str
    timestamp: datetime
    
    # Chamber characteristics
    core_nodes: List[str] = field(default_factory=list)
    peripheral_nodes: List[str] = field(default_factory=list)
    isolation_score: float = 0.0  # How isolated the chamber is
    
    # Content analysis
    content_diversity: float = 0.0  # Diversity of content
    opinion_polarization: float = 0.0  # How polarized opinions are
    confirmation_bias: float = 0.0  # Strength of confirmation bias
    
    # Impact metrics
    influence_radius: float = 0.0  # How far influence spreads
    distortion_impact: float = 0.0  # Impact on information quality
    
    def __post_init__(self):
        if not self.chamber_id:
            self.chamber_id = self._generate_chamber_id()
    
    def _generate_chamber_id(self) -> str:
        """Generate unique chamber ID"""
        content = f"{','.join(self.core_nodes)}{self.timestamp.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert echo chamber to dictionary"""
        return {
            "chamber_id": self.chamber_id,
            "timestamp": self.timestamp.isoformat(),
            "core_nodes": self.core_nodes,
            "peripheral_nodes": self.peripheral_nodes,
            "isolation_score": self.isolation_score,
            "content_diversity": self.content_diversity,
            "opinion_polarization": self.opinion_polarization,
            "confirmation_bias": self.confirmation_bias,
            "influence_radius": self.influence_radius,
            "distortion_impact": self.distortion_impact
        }


class FeedbackAnalyzer:
    """
    Analyzes feedback loops for distortion
    
    Identifies echo chambers, filter bubbles, and unhealthy
    feedback patterns that can lead to system degradation.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        
        # Storage
        self.feedback_loops: Dict[str, FeedbackLoop] = {}
        self.echo_chambers: Dict[str, EchoChamber] = {}
        
        # Configuration
        self.detection_thresholds = {
            FeedbackType.POSITIVE: 0.7,
            FeedbackType.NEGATIVE: 0.6,
            FeedbackType.REINFORCING: 0.8,
            FeedbackType.BALANCING: 0.5,
            FeedbackType.ECHO_CHAMBER: 0.6,
            FeedbackType.FILTER_BUBBLE: 0.65
        }
        
        # Analysis parameters
        self.min_loop_size = 3
        self.max_loop_size = 10
        self.echo_chamber_threshold = 0.7
        
        # Performance metrics
        self.loops_detected = 0
        self.chambers_detected = 0
        self.distortions_identified = 0
        
        logger.info(f"FeedbackAnalyzer initialized for node: {self.node_id}")
    
    def analyze_network_feedback(self, network_data: Dict[str, Any]) -> List[FeedbackLoop]:
        """Analyze network for feedback loops"""
        try:
            # Simple feedback loop detection
            loops = self._detect_simple_feedback_loops(network_data)
            
            # Analyze each loop
            analyzed_loops = []
            for loop in loops:
                analyzed_loop = self._analyze_feedback_loop(loop, network_data)
                if analyzed_loop:
                    analyzed_loops.append(analyzed_loop)
                    self.feedback_loops[analyzed_loop.loop_id] = analyzed_loop
                    self.loops_detected += 1
            
            logger.info(f"Detected {len(analyzed_loops)} feedback loops")
            return analyzed_loops
            
        except Exception as e:
            logger.error(f"Failed to analyze network feedback: {e}")
            return []
    
    def _detect_simple_feedback_loops(self, network_data: Dict[str, Any]) -> List[List[str]]:
        """Detect simple feedback loops in the network"""
        loops = []
        
        # Look for circular connections
        if "connections" in network_data:
            connections = network_data["connections"]
            
            # Simple circular detection
            for i, conn1 in enumerate(connections):
                source1 = conn1.get("source", "")
                target1 = conn1.get("target", "")
                
                for j, conn2 in enumerate(connections):
                    if i != j:
                        source2 = conn2.get("source", "")
                        target2 = conn2.get("target", "")
                        
                        # Check for circular connection
                        if target1 == source2 and target2 == source1:
                            loop = [source1, target1, source2]
                            if len(loop) >= self.min_loop_size:
                                loops.append(loop)
        
        return loops
    
    def _analyze_feedback_loop(self, loop_nodes: List[str], network_data: Dict[str, Any]) -> Optional[FeedbackLoop]:
        """Analyze a specific feedback loop"""
        try:
            # Determine feedback type
            feedback_type = self._determine_feedback_type(loop_nodes, network_data)
            
            # Calculate loop strength
            loop_strength = self._calculate_loop_strength(loop_nodes, network_data)
            
            # Calculate amplification factor
            amplification_factor = self._calculate_amplification_factor(loop_nodes, network_data)
            
            # Calculate stability score
            stability_score = self._calculate_stability_score(loop_nodes, network_data)
            
            # Calculate health impact
            health_impact = self._calculate_health_impact(loop_nodes, network_data)
            
            # Determine distortion level
            distortion_level = self._determine_distortion_level(
                loop_strength, amplification_factor, stability_score, health_impact
            )
            
            # Create feedback loop
            loop = FeedbackLoop(
                loop_id="",
                feedback_type=feedback_type,
                timestamp=datetime.utcnow(),
                nodes_involved=loop_nodes,
                loop_strength=loop_strength,
                distortion_level=distortion_level,
                amplification_factor=amplification_factor,
                stability_score=stability_score,
                health_impact=health_impact,
                loop_context={"network_data": network_data},
                detection_method="simple_cycle_detection"
            )
            
            return loop
            
        except Exception as e:
            logger.error(f"Failed to analyze feedback loop: {e}")
            return None
    
    def _determine_feedback_type(self, loop_nodes: List[str], network_data: Dict[str, Any]) -> FeedbackType:
        """Determine the type of feedback loop"""
        # Simple heuristic: assume reinforcing for demo
        return FeedbackType.REINFORCING
    
    def _calculate_loop_strength(self, loop_nodes: List[str], network_data: Dict[str, Any]) -> float:
        """Calculate the strength of the feedback loop"""
        # Simple calculation based on loop size
        if len(loop_nodes) < 2:
            return 0.0
        
        # Larger loops are generally stronger
        strength = min(1.0, len(loop_nodes) / 10.0)
        return strength
    
    def _calculate_amplification_factor(self, loop_nodes: List[str], network_data: Dict[str, Any]) -> float:
        """Calculate how much the loop amplifies signals"""
        # Simple calculation
        return min(5.0, len(loop_nodes) * 0.5)
    
    def _calculate_stability_score(self, loop_nodes: List[str], network_data: Dict[str, Any]) -> float:
        """Calculate stability score of the loop"""
        # Simple calculation
        if len(loop_nodes) < 3:
            return 0.5
        
        # Smaller loops are more stable
        stability = max(0.1, 1.0 - (len(loop_nodes) - 3) / 7.0)
        return stability
    
    def _calculate_health_impact(self, loop_nodes: List[str], network_data: Dict[str, Any]) -> float:
        """Calculate impact on system health"""
        # Simple calculation
        loop_strength = self._calculate_loop_strength(loop_nodes, network_data)
        return loop_strength
    
    def _determine_distortion_level(self, loop_strength: float, amplification_factor: float,
                                  stability_score: float, health_impact: float) -> DistortionLevel:
        """Determine the level of distortion caused by the feedback loop"""
        # Simple calculation
        distortion_score = (loop_strength + (amplification_factor - 1) / 4 + (1 - stability_score) + health_impact) / 4
        
        if distortion_score > 0.8:
            return DistortionLevel.CRITICAL
        elif distortion_score > 0.6:
            return DistortionLevel.HIGH
        elif distortion_score > 0.4:
            return DistortionLevel.MODERATE
        elif distortion_score > 0.2:
            return DistortionLevel.LOW
        else:
            return DistortionLevel.NONE
    
    def detect_echo_chambers(self, communication_data: Dict[str, Any]) -> List[EchoChamber]:
        """Detect echo chambers in communication patterns"""
        try:
            chambers = []
            
            # Simple echo chamber detection
            if "communications" in communication_data:
                comms = communication_data["communications"]
                
                # Group by sender for simple analysis
                sender_groups = {}
                for comm in comms:
                    sender = comm.get("sender", "")
                    if sender:
                        if sender not in sender_groups:
                            sender_groups[sender] = []
                        sender_groups[sender].append(comm)
                
                # Create simple echo chambers
                for sender, group_comms in sender_groups.items():
                    if len(group_comms) >= 3:  # Minimum size
                        chamber = self._create_simple_echo_chamber(sender, group_comms)
                        if chamber:
                            chambers.append(chamber)
                            self.echo_chambers[chamber.chamber_id] = chamber
                            self.chambers_detected += 1
            
            logger.info(f"Detected {len(chambers)} echo chambers")
            return chambers
            
        except Exception as e:
            logger.error(f"Failed to detect echo chambers: {e}")
            return []
    
    def _create_simple_echo_chamber(self, sender: str, communications: List[Dict[str, Any]]) -> Optional[EchoChamber]:
        """Create a simple echo chamber for demo purposes"""
        try:
            # Simple calculations
            isolation_score = 0.7  # Assume some isolation
            content_diversity = 0.3  # Assume low diversity
            opinion_polarization = 0.6  # Assume some polarization
            confirmation_bias = 0.5  # Assume moderate bias
            influence_radius = 0.4  # Assume moderate influence
            distortion_impact = 0.5  # Assume moderate distortion
            
            # Create chamber
            chamber = EchoChamber(
                chamber_id="",
                timestamp=datetime.utcnow(),
                core_nodes=[sender],
                peripheral_nodes=[],
                isolation_score=isolation_score,
                content_diversity=content_diversity,
                opinion_polarization=opinion_polarization,
                confirmation_bias=confirmation_bias,
                influence_radius=influence_radius,
                distortion_impact=distortion_impact
            )
            
            return chamber
            
        except Exception as e:
            logger.error(f"Failed to create echo chamber: {e}")
            return None
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of feedback analysis"""
        summary = {
            "node_id": self.node_id,
            "total_loops_detected": self.loops_detected,
            "total_chambers_detected": self.chambers_detected,
            "total_distortions_identified": self.distortions_identified,
            "active_loops": len(self.feedback_loops),
            "active_chambers": len(self.echo_chambers)
        }
        
        # Add feedback type distribution
        feedback_types = {}
        for loop in self.feedback_loops.values():
            feedback_type = loop.feedback_type.value
            feedback_types[feedback_type] = feedback_types.get(feedback_type, 0) + 1
        
        summary["feedback_type_distribution"] = feedback_types
        
        # Add distortion level distribution
        distortion_levels = {}
        for loop in self.feedback_loops.values():
            distortion = loop.distortion_level.value
            distortion_levels[distortion] = distortion_levels.get(distortion, 0) + 1
        
        summary["distortion_level_distribution"] = distortion_levels
        
        return summary
    
    def cleanup_old_data(self, max_age_days: int = 90) -> int:
        """Clean up old feedback analysis data"""
        cutoff_time = datetime.utcnow() - timedelta(days=max_age_days)
        cleaned_count = 0
        
        # Remove old feedback loops
        old_loops = []
        for loop_id, loop in self.feedback_loops.items():
            if loop.timestamp < cutoff_time:
                old_loops.append(loop_id)
        
        for loop_id in old_loops:
            del self.feedback_loops[loop_id]
            cleaned_count += 1
        
        # Remove old echo chambers
        old_chambers = []
        for chamber_id, chamber in self.echo_chambers.items():
            if chamber.timestamp < cutoff_time:
                old_chambers.append(chamber_id)
        
        for chamber_id in old_chambers:
            del self.echo_chambers[chamber_id]
            cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old feedback analysis data points")
        
        return cleaned_count
