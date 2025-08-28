"""
Mesh Interaction Learner
=======================

Component 8.1: Local Learning from Interactions
Learn from user interactions and local experiences

Implements learning from user interactions, behavior patterns,
feedback loops, and local knowledge accumulation.
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


class InteractionType(Enum):
    """Types of user interactions"""
    QUERY = "query"                    # User asking questions
    FEEDBACK = "feedback"              # User providing feedback
    CORRECTION = "correction"          # User correcting mistakes
    PREFERENCE = "preference"          # User expressing preferences
    BEHAVIOR = "behavior"              # User behavior patterns
    CONVERSATION = "conversation"      # Conversational interactions


class PatternType(Enum):
    """Types of learning patterns"""
    FREQUENCY = "frequency"            # Learn from frequency patterns
    CORRELATION = "correlation"        # Learn from correlations
    SEQUENCE = "sequence"              # Learn from sequences
    CONTEXT = "context"                # Learn from context
    FEEDBACK = "feedback"              # Learn from feedback


@dataclass
class UserInteraction:
    """A user interaction for learning"""
    interaction_id: str
    user_id: str
    interaction_type: InteractionType
    timestamp: datetime
    
    # Interaction content
    content: str
    context: Dict[str, Any]
    
    # Learning signals
    learning_value: float = 0.0  # 0.0 to 1.0
    confidence: float = 0.0      # 0.0 to 1.0
    feedback_score: Optional[float] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    
    def __post_init__(self):
        if not self.interaction_id:
            self.interaction_id = self._generate_interaction_id()
    
    def _generate_interaction_id(self) -> str:
        """Generate unique interaction ID"""
        content = f"{self.user_id}{self.interaction_type.value}{self.timestamp.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert interaction to dictionary"""
        return {
            "interaction_id": self.interaction_id,
            "user_id": self.user_id,
            "interaction_type": self.interaction_type.value,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "context": self.context,
            "learning_value": self.learning_value,
            "confidence": self.confidence,
            "feedback_score": self.feedback_score,
            "tags": self.tags,
            "notes": self.notes
        }


@dataclass
class LearningPattern:
    """A learned pattern from interactions"""
    pattern_id: str
    pattern_type: PatternType
    created_at: datetime
    
    # Pattern data
    pattern_data: Dict[str, Any]
    confidence: float = 0.0  # 0.0 to 1.0
    support_count: int = 0
    
    # Learning metadata
    learning_source: str = "interaction"
    last_updated: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    
    # Quality metrics
    quality_score: float = 0.0  # 0.0 to 1.0
    stability_score: float = 0.0  # 0.0 to 1.0
    
    def __post_init__(self):
        if not self.pattern_id:
            self.pattern_id = self._generate_pattern_id()
    
    def _generate_pattern_id(self) -> str:
        """Generate unique pattern ID"""
        content = f"{self.pattern_type.value}{self.created_at.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary"""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type.value,
            "created_at": self.created_at.isoformat(),
            "pattern_data": self.pattern_data,
            "confidence": self.confidence,
            "support_count": self.support_count,
            "learning_source": self.learning_source,
            "last_updated": self.last_updated.isoformat(),
            "version": self.version,
            "quality_score": self.quality_score,
            "stability_score": self.stability_score
        }


class InteractionLearner:
    """
    Learns from user interactions and local experiences
    
    Analyzes interaction patterns, extracts learning signals,
    and builds local knowledge for continuous improvement.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        
        # Interaction storage
        self.user_interactions: Dict[str, UserInteraction] = {}
        self.learned_patterns: Dict[str, LearningPattern] = {}
        
        # Learning configuration
        self.min_interactions_for_pattern = 3  # Lower for demo/testing
        self.confidence_threshold = 0.5  # Lowered for demo/testing
        self.learning_decay_factor = 0.95
        
        # Pattern tracking
        self.pattern_versions: Dict[str, int] = {}
        self.pattern_stability: Dict[str, float] = {}
        
        # Performance metrics
        self.total_interactions = 0
        self.patterns_learned = 0
        self.patterns_updated = 0
        
        # Threading
        self.lock = threading.RLock()
        
        logger.info(f"InteractionLearner initialized for node: {self.node_id}")
    
    def record_interaction(self, user_id: str, interaction_type: InteractionType,
                          content: str, context: Dict[str, Any], **kwargs) -> str:
        """Record a new user interaction"""
        try:
            with self.lock:
                # Create interaction
                interaction = UserInteraction(
                    interaction_id="",
                    user_id=user_id,
                    interaction_type=interaction_type,
                    timestamp=datetime.utcnow(),
                    content=content,
                    context=context,
                    **kwargs
                )
                
                # Calculate learning value
                interaction.learning_value = self._calculate_learning_value(interaction)
                
                # Store interaction
                self.user_interactions[interaction.interaction_id] = interaction
                self.total_interactions += 1
                
                # Trigger pattern learning
                self._trigger_pattern_learning(interaction)
                
                logger.info(f"Recorded interaction: {interaction_type.value} from {user_id}")
                return interaction.interaction_id
                
        except Exception as e:
            logger.error(f"Failed to record interaction: {e}")
            raise
    
    def _calculate_learning_value(self, interaction: UserInteraction) -> float:
        """Calculate the learning value of an interaction"""
        base_value = 0.5
        
        # Boost for feedback and corrections
        if interaction.interaction_type == InteractionType.FEEDBACK:
            base_value += 0.3
        elif interaction.interaction_type == InteractionType.CORRECTION:
            base_value += 0.4
        
        # Boost for high confidence
        if interaction.confidence > 0.8:
            base_value += 0.2
        
        # Boost for feedback score
        if interaction.feedback_score is not None:
            if interaction.feedback_score > 0.7:
                base_value += 0.2
            elif interaction.feedback_score < 0.3:
                base_value += 0.1  # Learn from negative feedback too
        
        return min(1.0, base_value)
    
    def _trigger_pattern_learning(self, interaction: UserInteraction):
        """Trigger pattern learning from new interaction"""
        try:
            # Check if we have enough interactions for pattern learning
            if len(self.user_interactions) < self.min_interactions_for_pattern:
                return
            
            # Analyze for different pattern types
            self._analyze_frequency_patterns(interaction)
            self._analyze_correlation_patterns(interaction)
            self._analyze_sequence_patterns(interaction)
            self._analyze_context_patterns(interaction)
            
        except Exception as e:
            logger.error(f"Error in pattern learning: {e}")
    
    def _analyze_frequency_patterns(self, new_interaction: UserInteraction):
        """Analyze frequency-based patterns"""
        try:
            # Group interactions by type and content similarity
            similar_interactions = []
            
            for interaction in self.user_interactions.values():
                if (interaction.interaction_type == new_interaction.interaction_type and
                    self._calculate_similarity(interaction.content, new_interaction.content) > 0.7):
                    similar_interactions.append(interaction)
            
            # If we have enough similar interactions, create a frequency pattern
            if len(similar_interactions) >= 2:  # Lowered from 3 to 2 for demo
                pattern_data = {
                    "interaction_type": new_interaction.interaction_type.value,
                    "content_pattern": new_interaction.content,
                    "frequency": len(similar_interactions),
                    "user_ids": list(set(i.user_id for i in similar_interactions)),
                    "time_range": {
                        "start": min(i.timestamp for i in similar_interactions).isoformat(),
                        "end": max(i.timestamp for i in similar_interactions).isoformat()
                    }
                }
                
                self._create_or_update_pattern(
                    PatternType.FREQUENCY,
                    pattern_data,
                    confidence=min(1.0, len(similar_interactions) / 10.0)
                )
                
        except Exception as e:
            logger.error(f"Error analyzing frequency patterns: {e}")
    
    def _analyze_correlation_patterns(self, new_interaction: UserInteraction):
        """Analyze correlation-based patterns"""
        try:
            # Look for correlations between interaction types and contexts
            correlations = {}
            
            for interaction in self.user_interactions.values():
                if interaction.interaction_type != new_interaction.interaction_type:
                    continue
                
                # Analyze context correlations
                for key, value in interaction.context.items():
                    if key in new_interaction.context:
                        if key not in correlations:
                            correlations[key] = {"values": [], "counts": {}}
                        
                        correlations[key]["values"].append(value)
                        
                        if value not in correlations[key]["counts"]:
                            correlations[key]["counts"][value] = 0
                        correlations[key]["counts"][value] += 1
            
            # Create patterns for strong correlations
            for key, data in correlations.items():
                if len(data["values"]) >= 2:  # Lowered from 3 to 2 for demo
                    # Find most common value
                    most_common = max(data["counts"].items(), key=lambda x: x[1])
                    confidence = most_common[1] / len(data["values"])
                    
                    if confidence > 0.6:
                        pattern_data = {
                            "interaction_type": new_interaction.interaction_type.value,
                            "correlated_key": key,
                            "correlated_value": most_common[0],
                            "confidence": confidence,
                            "total_occurrences": len(data["values"])
                        }
                        
                        self._create_or_update_pattern(
                            PatternType.CORRELATION,
                            pattern_data,
                            confidence=confidence
                        )
                        
        except Exception as e:
            logger.error(f"Error analyzing correlation patterns: {e}")
    
    def _analyze_sequence_patterns(self, new_interaction: UserInteraction):
        """Analyze sequence-based patterns"""
        try:
            # Look for sequential patterns in user interactions
            user_interactions = [i for i in self.user_interactions.values() 
                               if i.user_id == new_interaction.user_id]
            
            if len(user_interactions) < 3:
                return
            
            # Sort by timestamp
            user_interactions.sort(key=lambda x: x.timestamp)
            
            # Look for repeating sequences
            sequences = []
            for i in range(len(user_interactions) - 2):
                seq = [
                    user_interactions[i].interaction_type.value,
                    user_interactions[i+1].interaction_type.value,
                    user_interactions[i+2].interaction_type.value
                ]
                sequences.append(seq)
            
            # Count sequence occurrences
            sequence_counts = {}
            for seq in sequences:
                seq_key = "->".join(seq)
                sequence_counts[seq_key] = sequence_counts.get(seq_key, 0) + 1
            
            # Create patterns for frequent sequences
            for seq_key, count in sequence_counts.items():
                if count >= 1:  # Lowered from 2 to 1 for demo
                    pattern_data = {
                        "sequence": seq_key.split("->"),
                        "frequency": count,
                        "user_id": new_interaction.user_id,
                        "total_sequences": len(sequences)
                    }
                    
                    confidence = min(1.0, count / len(sequences))
                    
                    self._create_or_update_pattern(
                        PatternType.SEQUENCE,
                        pattern_data,
                        confidence=confidence
                    )
                    
        except Exception as e:
            logger.error(f"Error analyzing sequence patterns: {e}")
    
    def _analyze_context_patterns(self, new_interaction: UserInteraction):
        """Analyze context-based patterns"""
        try:
            # Look for patterns in interaction contexts
            context_patterns = {}
            
            for interaction in self.user_interactions.values():
                if interaction.interaction_type != new_interaction.interaction_type:
                    continue
                
                # Analyze context keys
                for key in interaction.context.keys():
                    if key not in context_patterns:
                        context_patterns[key] = 0
                    context_patterns[key] += 1
            
            # Create patterns for common context keys
            for key, count in context_patterns.items():
                if count >= 2:  # Lowered from 3 to 2 for demo
                    pattern_data = {
                        "interaction_type": new_interaction.interaction_type.value,
                        "context_key": key,
                        "frequency": count,
                        "total_interactions": len([i for i in self.user_interactions.values() 
                                                if i.interaction_type == new_interaction.interaction_type])
                    }
                    
                    confidence = min(1.0, count / pattern_data["total_interactions"])
                    
                    self._create_or_update_pattern(
                        PatternType.CONTEXT,
                        pattern_data,
                        confidence=confidence
                    )
                    
        except Exception as e:
            logger.error(f"Error analyzing context patterns: {e}")
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        # Simple similarity based on common words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _create_or_update_pattern(self, pattern_type: LearningPattern, 
                                 pattern_data: Dict[str, Any], confidence: float):
        """Create a new pattern or update existing one"""
        try:
            # Check if similar pattern exists
            existing_pattern = None
            for pattern in self.learned_patterns.values():
                if (pattern.pattern_type == pattern_type and
                    self._patterns_are_similar(pattern.pattern_data, pattern_data)):
                    existing_pattern = pattern
                    break
            
            if existing_pattern:
                # Update existing pattern
                existing_pattern.pattern_data.update(pattern_data)
                existing_pattern.confidence = (existing_pattern.confidence + confidence) / 2
                existing_pattern.support_count += 1
                existing_pattern.last_updated = datetime.utcnow()
                existing_pattern.version += 1
                
                self.patterns_updated += 1
                logger.info(f"Updated pattern: {existing_pattern.pattern_id}")
                
            else:
                # Create new pattern
                pattern = LearningPattern(
                    pattern_id="",
                    pattern_type=pattern_type,
                    created_at=datetime.utcnow(),
                    pattern_data=pattern_data,
                    confidence=confidence,
                    support_count=1
                )
                
                self.learned_patterns[pattern.pattern_id] = pattern
                self.patterns_learned += 1
                
                logger.info(f"Created new pattern: {pattern.pattern_id}")
                
        except Exception as e:
            logger.error(f"Error creating/updating pattern: {e}")
    
    def _patterns_are_similar(self, data1: Dict[str, Any], data2: Dict[str, Any]) -> bool:
        """Check if two patterns are similar enough to merge"""
        # Simple similarity check - can be enhanced
        common_keys = set(data1.keys()) & set(data2.keys())
        if len(common_keys) < 2:
            return False
        
        # Check if key values are similar
        similar_count = 0
        for key in common_keys:
            if data1[key] == data2[key]:
                similar_count += 1
        
        return similar_count / len(common_keys) > 0.7
    
    def get_learned_patterns(self, pattern_type: Optional[PatternType] = None,
                            min_confidence: float = 0.0) -> List[LearningPattern]:
        """Get learned patterns with optional filtering"""
        with self.lock:
            patterns = []
            
            for pattern in self.learned_patterns.values():
                if pattern_type and pattern.pattern_type != pattern_type:
                    continue
                
                if pattern.confidence < min_confidence:
                    continue
                
                patterns.append(pattern)
            
            # Sort by confidence
            patterns.sort(key=lambda x: x.confidence, reverse=True)
            return patterns
    
    def get_interaction_summary(self) -> Dict[str, Any]:
        """Get comprehensive interaction learning summary"""
        with self.lock:
            # Count interactions by type
            interaction_counts = {}
            for interaction in self.user_interactions.values():
                it_type = interaction.interaction_type.value
                interaction_counts[it_type] = interaction_counts.get(it_type, 0) + 1
            
            # Count patterns by type
            pattern_counts = {}
            for pattern in self.learned_patterns.values():
                p_type = pattern.pattern_type.value
                pattern_counts[p_type] = pattern_counts.get(p_type, 0) + 1
            
            return {
                "node_id": self.node_id,
                "total_interactions": self.total_interactions,
                "interaction_counts": interaction_counts,
                "total_patterns": len(self.learned_patterns),
                "pattern_counts": pattern_counts,
                "patterns_learned": self.patterns_learned,
                "patterns_updated": self.patterns_updated,
                "min_interactions_for_pattern": self.min_interactions_for_pattern,
                "confidence_threshold": self.confidence_threshold
            }
    
    def _analyze_patterns_for_user(self, user_id: str):
        """Force pattern analysis for a specific user's interactions"""
        try:
            with self.lock:
                # Get user interactions
                user_interactions = [i for i in self.user_interactions.values() if i.user_id == user_id]
                
                if len(user_interactions) < self.min_interactions_for_pattern:
                    return
                
                # Analyze patterns for most recent interaction
                if user_interactions:
                    latest_interaction = max(user_interactions, key=lambda x: x.timestamp)
                    self._analyze_frequency_patterns(latest_interaction)
                    self._analyze_correlation_patterns(latest_interaction)
                    self._analyze_sequence_patterns(latest_interaction)
                    self._analyze_context_patterns(latest_interaction)
                    
                    logger.info(f"Forced pattern analysis for user: {user_id}")
                    
        except Exception as e:
            logger.error(f"Error in forced pattern analysis: {e}")
    
    def cleanup_old_interactions(self, days_old: int = 30) -> int:
        """Remove old interactions to save memory"""
        try:
            with self.lock:
                cutoff_date = datetime.utcnow() - timedelta(days=days_old)
                old_interactions = [iid for iid, interaction in self.user_interactions.items()
                                  if interaction.timestamp < cutoff_date]
                
                for iid in old_interactions:
                    del self.user_interactions[iid]
                
                logger.info(f"Cleaned up {len(old_interactions)} old interactions")
                return len(old_interactions)
                
        except Exception as e:
            logger.error(f"Failed to cleanup old interactions: {e}")
            return 0
