#!/usr/bin/env python3
"""
Context Learner - Enhanced with Sentient's Context-Aware Learning Concepts

This module integrates Sentient's learning capabilities into The Mesh,
providing continuous improvement, pattern recognition, and adaptive behavior.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import sys
from collections import defaultdict, Counter

# Try to import Sentient's learning concepts
try:
    sys.path.append('/Users/admin/AI/Sentient/src/server/main')
    from memories.constants import TOPICS
    SENTIENT_LEARNING_AVAILABLE = True
except ImportError:
    SENTIENT_LEARNING_AVAILABLE = False
    # Mock constants for development/testing
    TOPICS = [
        {"name": "Personal Identity", "description": "Core traits, personality, beliefs, values, ethics, and preferences"},
        {"name": "Interests & Lifestyle", "description": "Hobbies, recreational activities, habits, routines, daily behavior"},
        {"name": "Work & Learning", "description": "Career, jobs, professional achievements, academic background, skills, certifications"},
        {"name": "Health & Wellbeing", "description": "Mental and physical health, self-care practices"},
        {"name": "Relationships & Social Life", "description": "family, friends, romantic connections, social interactions, social media"},
        {"name": "Financial", "description": "Income, expenses, investments, financial goals"},
        {"name": "Goals & Challenges", "description": "Aspirations, objectives, obstacles, and difficulties faced"},
        {"name": "Miscellaneous", "description": "Anything that doesn't clearly fit into the above"}
    ]

# Enums for context learning
class LearningPatternType(Enum):
    """Types of learning patterns"""
    INTERACTION_PATTERN = "interaction_pattern"
    TASK_PATTERN = "task_pattern"
    TIME_PATTERN = "time_pattern"
    TOPIC_PATTERN = "topic_pattern"
    BEHAVIOR_PATTERN = "behavior_pattern"
    PREFERENCE_PATTERN = "preference_pattern"

class LearningQuality(Enum):
    """Quality indicators for learning"""
    HIGH = "high"      # Strong pattern, high confidence
    MEDIUM = "medium"  # Moderate pattern, good confidence
    LOW = "low"        # Weak pattern, low confidence
    UNCERTAIN = "uncertain"  # Insufficient data

class AdaptationType(Enum):
    """Types of adaptations based on learning"""
    RESPONSE_STYLE = "response_style"
    INTERACTION_FREQUENCY = "interaction_frequency"
    SUGGESTION_TIMING = "suggestion_timing"
    CONTENT_PERSONALIZATION = "content_personalization"
    TOOL_RECOMMENDATION = "tool_recommendation"

# Dataclasses for context learning
@dataclass
class LearningPattern:
    """A learned pattern from user interactions"""
    pattern_id: str
    user_id: str
    pattern_type: LearningPatternType
    pattern_data: Dict[str, Any]
    confidence_score: float
    sample_size: int
    first_observed: float
    last_observed: float
    frequency: float  # Occurrences per day
    quality: LearningQuality
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContextSnapshot:
    """Snapshot of context for learning analysis"""
    timestamp: float
    user_id: str
    context_type: str
    context_data: Dict[str, Any]
    interaction_data: Optional[Dict[str, Any]] = None
    outcome_data: Optional[Dict[str, Any]] = None

@dataclass
class LearningInsight:
    """Insight derived from learning patterns"""
    insight_id: str
    user_id: str
    insight_type: str
    description: str
    confidence: float
    supporting_patterns: List[str]
    suggested_actions: List[str]
    created_at: float = field(default_factory=time.time)

@dataclass
class AdaptationRecommendation:
    """Recommendation for system adaptation"""
    adaptation_id: str
    user_id: str
    adaptation_type: AdaptationType
    current_value: Any
    recommended_value: Any
    reasoning: str
    confidence: float
    priority: int  # 0=high, 1=medium, 2=low
    created_at: float = field(default_factory=time.time)

@dataclass
class ContextLearnerConfig:
    """Configuration for context learning"""
    enabled: bool = True
    learning_rate: float = 0.1  # How quickly to adapt
    min_sample_size: int = 5    # Minimum samples for pattern recognition
    confidence_threshold: float = 0.7  # Minimum confidence for adaptations
    pattern_lifetime_days: int = 30    # How long to keep patterns
    max_patterns_per_user: int = 100  # Maximum patterns to maintain
    learning_interval: int = 60        # Learning cycle interval in seconds

@dataclass
class LearningResult:
    """Result of a learning cycle"""
    patterns_identified: int
    insights_generated: int
    adaptations_recommended: int
    learning_quality_score: float
    user_improvement_score: float
    system_adaptation_score: float

class ContextLearner:
    """
    Enhanced Context Learner with Sentient's Context-Aware Learning Concepts
    
    Provides continuous improvement, pattern recognition, and adaptive behavior
    while maintaining Mesh's local-first architecture.
    """
    
    def __init__(self, config: ContextLearnerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Learning state
        self.learning_patterns: Dict[str, LearningPattern] = {}
        self.context_history: List[ContextSnapshot] = []
        self.learning_insights: Dict[str, LearningInsight] = {}
        self.adaptation_recommendations: Dict[str, AdaptationRecommendation] = {}
        
        # Pattern recognition state
        self.pattern_matchers: Dict[LearningPatternType, Any] = {}
        self.learning_models: Dict[str, Any] = {}
        
        # Sentient integration status
        self.sentient_available = SENTIENT_LEARNING_AVAILABLE
        
        # Learning engine state
        self._is_running = False
        self._engine_task: Optional[asyncio.Task] = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize context learning components"""
        
        if self.sentient_available:
            self.logger.info("Sentient learning concepts available")
        else:
            self.logger.info("Using mock learning concepts for development")
        
        # Initialize pattern matchers
        self._initialize_pattern_matchers()
    
    def _initialize_pattern_matchers(self):
        """Initialize pattern matching components"""
        
        # Initialize pattern matchers for different types
        self.pattern_matchers[LearningPatternType.INTERACTION_PATTERN] = InteractionPatternMatcher()
        self.pattern_matchers[LearningPatternType.TASK_PATTERN] = TaskPatternMatcher()
        self.pattern_matchers[LearningPatternType.TIME_PATTERN] = TimePatternMatcher()
        self.pattern_matchers[LearningPatternType.TOPIC_PATTERN] = TopicPatternMatcher()
        self.pattern_matchers[LearningPatternType.BEHAVIOR_PATTERN] = BehaviorPatternMatcher()
        self.pattern_matchers[LearningPatternType.PREFERENCE_PATTERN] = PreferencePatternMatcher()
    
    async def start_learning_engine(self):
        """Start the context learning engine"""
        
        if self._is_running:
            self.logger.warning("Learning engine is already running")
            return
        
        self.logger.info("Starting context learning engine...")
        self._is_running = True
        self._engine_task = asyncio.create_task(self._run_learning_loop())
    
    async def stop_learning_engine(self):
        """Stop the context learning engine"""
        
        if not self._is_running:
            self.logger.warning("Learning engine is not running")
            return
        
        self.logger.info("Stopping context learning engine...")
        self._is_running = False
        
        if self._engine_task:
            self._engine_task.cancel()
            try:
                await self._engine_task
            except asyncio.CancelledError:
                self.logger.info("Learning engine task cancelled successfully")
            except Exception as e:
                self.logger.error(f"Error while stopping learning engine: {e}")
        
        self._engine_task = None
    
    async def _run_learning_loop(self):
        """Main loop for context learning"""
        
        self.logger.info("Context learning engine started")
        
        while self._is_running:
            try:
                # Analyze context history for patterns
                await self._analyze_context_for_patterns()
                
                # Generate learning insights
                await self._generate_learning_insights()
                
                # Generate adaptation recommendations
                await self._generate_adaptation_recommendations()
                
                # Clean up old patterns and insights
                await self._cleanup_old_learning_data()
                
                # Wait for next learning cycle
                await asyncio.sleep(self.config.learning_interval)
                
            except Exception as e:
                self.logger.error(f"Error in learning loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _analyze_context_for_patterns(self):
        """Analyze context history to identify learning patterns"""
        
        if not self.context_history:
            return
        
        # Group context by user for analysis
        user_contexts = defaultdict(list)
        for context in self.context_history:
            user_contexts[context.user_id].append(context)
        
        # Analyze patterns for each user
        for user_id, contexts in user_contexts.items():
            try:
                await self._analyze_user_context_patterns(user_id, contexts)
            except Exception as e:
                self.logger.error(f"Error analyzing patterns for user {user_id}: {e}")
    
    async def _analyze_user_context_patterns(
        self, 
        user_id: str, 
        contexts: List[ContextSnapshot]
    ):
        """Analyze patterns for a specific user"""
        
        # Analyze different types of patterns
        for pattern_type, matcher in self.pattern_matchers.items():
            try:
                patterns = await matcher.identify_patterns(user_id, contexts)
                
                # Add new patterns to the system
                for pattern in patterns:
                    if self._should_add_pattern(pattern):
                        self.learning_patterns[pattern.pattern_id] = pattern
                        
                        self.logger.debug(
                            f"Identified {pattern_type.value} pattern for user {user_id}: "
                            f"{pattern.pattern_id}"
                        )
                        
            except Exception as e:
                self.logger.warning(f"Error analyzing {pattern_type.value} patterns: {e}")
    
    def _should_add_pattern(self, pattern: LearningPattern) -> bool:
        """Determine if a pattern should be added to the system"""
        
        # Check if pattern already exists
        if pattern.pattern_id in self.learning_patterns:
            return False
        
        # Check minimum sample size
        if pattern.sample_size < self.config.min_sample_size:
            return False
        
        # Check confidence threshold
        if pattern.confidence_score < self.config.confidence_threshold:
            return False
        
        # Check user pattern limit
        user_patterns = [
            p for p in self.learning_patterns.values()
            if p.user_id == pattern.user_id
        ]
        
        if len(user_patterns) >= self.config.max_patterns_per_user:
            # Remove lowest quality pattern
            if user_patterns:
                lowest_quality = min(user_patterns, key=lambda p: p.confidence_score)
                del self.learning_patterns[lowest_quality.pattern_id]
        
        return True
    
    async def _generate_learning_insights(self):
        """Generate insights from identified patterns"""
        
        # Group patterns by user for insight generation
        user_patterns = defaultdict(list)
        for pattern in self.learning_patterns.values():
            user_patterns[pattern.user_id].append(pattern)
        
        # Generate insights for each user
        for user_id, patterns in user_patterns.items():
            try:
                insights = await self._generate_user_insights(user_id, patterns)
                
                # Add new insights to the system
                for insight in insights:
                    self.learning_insights[insight.insight_id] = insight
                    
                    self.logger.info(
                        f"Generated insight for user {user_id}: {insight.description[:50]}..."
                    )
                    
            except Exception as e:
                self.logger.error(f"Error generating insights for user {user_id}: {e}")
    
    async def _generate_user_insights(
        self, 
        user_id: str, 
        patterns: List[LearningPattern]
    ) -> List[LearningInsight]:
        """Generate insights for a specific user"""
        
        insights = []
        
        # Use Sentient's approach if available
        if self.sentient_available:
            sentient_insights = await self._generate_sentient_insights(user_id, patterns)
            insights.extend(sentient_insights)
        else:
            # Generate basic insights
            basic_insights = self._generate_basic_insights(user_id, patterns)
            insights.extend(basic_insights)
        
        return insights
    
    async def _generate_sentient_insights(
        self, 
        user_id: str, 
        patterns: List[LearningPattern]
    ) -> List[LearningInsight]:
        """Generate insights using Sentient's learning concepts"""
        
        insights = []
        
        try:
            # Analyze interaction patterns
            interaction_patterns = [
                p for p in patterns 
                if p.pattern_type == LearningPatternType.INTERACTION_PATTERN
            ]
            
            if interaction_patterns:
                insight = await self._create_interaction_insight(user_id, interaction_patterns)
                if insight:
                    insights.append(insight)
            
            # Analyze task patterns
            task_patterns = [
                p for p in patterns 
                if p.pattern_type == LearningPatternType.TASK_PATTERN
            ]
            
            if task_patterns:
                insight = await self._create_task_insight(user_id, task_patterns)
                if insight:
                    insights.append(insight)
            
            # Analyze topic patterns
            topic_patterns = [
                p for p in patterns 
                if p.pattern_type == LearningPatternType.TOPIC_PATTERN
            ]
            
            if topic_patterns:
                insight = await self._create_topic_insight(user_id, topic_patterns)
                if insight:
                    insights.append(insight)
            
            # Analyze time patterns
            time_patterns = [
                p for p in patterns 
                if p.pattern_type == LearningPatternType.TIME_PATTERN
            ]
            
            if time_patterns:
                insight = await self._create_time_insight(user_id, time_patterns)
                if insight:
                    insights.append(insight)
            
        except Exception as e:
            self.logger.warning(f"Sentient insight generation failed: {e}")
        
        return insights
    
    async def _create_interaction_insight(
        self, 
        user_id: str, 
        patterns: List[LearningPattern]
    ) -> Optional[LearningInsight]:
        """Create insight from interaction patterns"""
        
        insight_id = f"interaction_insight_{user_id}_{int(time.time())}"
        
        # Analyze interaction frequency and timing
        avg_frequency = sum(p.frequency for p in patterns) / len(patterns)
        most_active_time = self._find_most_active_time(patterns)
        
        description = f"You tend to interact with the system {avg_frequency:.1f} times per day, "
        description += f"with peak activity around {most_active_time}."
        
        suggested_actions = [
            "Schedule important tasks during your peak activity times",
            "Set up automated reminders for low-activity periods",
            "Optimize notification timing based on your interaction patterns"
        ]
        
        insight = LearningInsight(
            insight_id=insight_id,
            user_id=user_id,
            insight_type="interaction_pattern",
            description=description,
            confidence=min(0.9, max(0.5, avg_frequency / 10)),  # Normalize confidence
            supporting_patterns=[p.pattern_id for p in patterns],
            suggested_actions=suggested_actions
        )
        
        return insight
    
    async def _create_task_insight(
        self, 
        user_id: str, 
        patterns: List[LearningPattern]
    ) -> Optional[LearningInsight]:
        """Create insight from task patterns"""
        
        insight_id = f"task_insight_{user_id}_{int(time.time())}"
        
        # Analyze task completion patterns
        task_data = [p.pattern_data for p in patterns]
        avg_completion_time = sum(
            data.get("completion_time", 0) for data in task_data
        ) / len(task_data)
        
        most_common_task_type = self._find_most_common_task_type(patterns)
        
        description = f"Your tasks typically take {avg_completion_time:.1f} minutes to complete. "
        description += f"You most frequently work on {most_common_task_type} tasks."
        
        suggested_actions = [
            "Break down complex tasks into smaller, more manageable pieces",
            "Schedule similar task types together for efficiency",
            "Set realistic time estimates based on your completion patterns"
        ]
        
        insight = LearningInsight(
            insight_id=insight_id,
            user_id=user_id,
            insight_type="task_pattern",
            description=description,
            confidence=min(0.9, max(0.5, len(patterns) / 10)),  # Normalize confidence
            supporting_patterns=[p.pattern_id for p in patterns],
            suggested_actions=suggested_actions
        )
        
        return insight
    
    async def _create_topic_insight(
        self, 
        user_id: str, 
        patterns: List[LearningPattern]
    ) -> Optional[LearningInsight]:
        """Create insight from topic patterns"""
        
        insight_id = f"topic_insight_{user_id}_{int(time.time())}"
        
        # Analyze topic preferences
        topic_data = [p.pattern_data for p in patterns]
        topic_counts = Counter(
            data.get("topic", "unknown") for data in topic_data
        )
        
        if topic_counts:
            most_common_topic = topic_counts.most_common(1)[0][0]
            topic_frequency = topic_counts[most_common_topic]
            
            description = f"You frequently engage with topics related to '{most_common_topic}' "
            description += f"({topic_frequency} interactions)."
            
            suggested_actions = [
                "Explore related topics to expand your knowledge",
                "Set up automated information gathering for your interests",
                "Connect with others who share similar interests"
            ]
            
            insight = LearningInsight(
                insight_id=insight_id,
                user_id=user_id,
                insight_type="topic_pattern",
                description=description,
                confidence=min(0.9, max(0.5, topic_frequency / 20)),  # Normalize confidence
                supporting_patterns=[p.pattern_id for p in patterns],
                suggested_actions=suggested_actions
            )
            
            return insight
        
        return None
    
    async def _create_time_insight(
        self, 
        user_id: str, 
        patterns: List[LearningPattern]
    ) -> Optional[LearningInsight]:
        """Create insight from time patterns"""
        
        insight_id = f"time_insight_{user_id}_{int(time.time())}"
        
        # Analyze time-based patterns
        time_data = [p.pattern_data for p in patterns]
        peak_hours = self._find_peak_hours(time_data)
        
        description = f"Your peak productivity hours are {', '.join(peak_hours)}. "
        description += "Consider scheduling important work during these times."
        
        suggested_actions = [
            "Schedule high-priority tasks during your peak hours",
            "Use low-energy periods for routine or administrative tasks",
            "Set up automated workflows to take advantage of your productive times"
        ]
        
        insight = LearningInsight(
            insight_id=insight_id,
            user_id=user_id,
            insight_type="time_pattern",
            description=description,
            confidence=min(0.9, max(0.5, len(patterns) / 10)),  # Normalize confidence
            supporting_patterns=[p.pattern_id for p in patterns],
            suggested_actions=suggested_actions
        )
        
        return insight
    
    def _find_most_active_time(self, patterns: List[LearningPattern]) -> str:
        """Find the most active time from patterns"""
        
        # Mock implementation - replace with real time analysis
        return "10:00 AM - 2:00 PM"
    
    def _find_most_common_task_type(self, patterns: List[LearningPattern]) -> str:
        """Find the most common task type from patterns"""
        
        # Mock implementation - replace with real task analysis
        return "project planning"
    
    def _find_peak_hours(self, time_data: List[Dict[str, Any]]) -> List[str]:
        """Find peak productivity hours from time data"""
        
        # Mock implementation - replace with real time analysis
        return ["9:00 AM", "2:00 PM"]
    
    def _generate_basic_insights(
        self, 
        user_id: str, 
        patterns: List[LearningPattern]
    ) -> List[LearningInsight]:
        """Generate basic insights when Sentient is not available"""
        
        insights = []
        
        if patterns:
            # Basic pattern summary insight
            insight = LearningInsight(
                insight_id=f"basic_insight_{user_id}_{int(time.time())}",
                user_id=user_id,
                insight_type="pattern_summary",
                description=f"I've identified {len(patterns)} learning patterns from your interactions.",
                confidence=0.6,
                supporting_patterns=[p.pattern_id for p in patterns],
                suggested_actions=["Review your interaction patterns", "Consider optimizing your workflow"]
            )
            insights.append(insight)
        
        return insights
    
    async def _generate_adaptation_recommendations(self):
        """Generate recommendations for system adaptation"""
        
        # Generate recommendations based on insights
        for insight in self.learning_insights.values():
            try:
                recommendations = await self._generate_insight_recommendations(insight)
                
                # Add new recommendations to the system
                for recommendation in recommendations:
                    self.adaptation_recommendations[recommendation.adaptation_id] = recommendation
                    
                    self.logger.debug(
                        f"Generated adaptation recommendation: {recommendation.adaptation_type.value}"
                    )
                    
            except Exception as e:
                self.logger.error(f"Error generating recommendations for insight {insight.insight_id}: {e}")
    
    async def _generate_insight_recommendations(
        self, 
        insight: LearningInsight
    ) -> List[AdaptationRecommendation]:
        """Generate adaptation recommendations from an insight"""
        
        recommendations = []
        
        # Generate recommendations based on insight type
        if insight.insight_type == "interaction_pattern":
            rec = await self._create_interaction_adaptation(insight)
            if rec:
                recommendations.append(rec)
        
        elif insight.insight_type == "task_pattern":
            rec = await self._create_task_adaptation(insight)
            if rec:
                recommendations.append(rec)
        
        elif insight.insight_type == "time_pattern":
            rec = await self._create_time_adaptation(insight)
            if rec:
                recommendations.append(rec)
        
        return recommendations
    
    async def _create_interaction_adaptation(
        self, 
        insight: LearningInsight
    ) -> Optional[AdaptationRecommendation]:
        """Create interaction adaptation recommendation"""
        
        adaptation_id = f"interaction_adapt_{insight.user_id}_{int(time.time())}"
        
        recommendation = AdaptationRecommendation(
            adaptation_id=adaptation_id,
            user_id=insight.user_id,
            adaptation_type=AdaptationType.INTERACTION_FREQUENCY,
            current_value="default",
            recommended_value="optimized",
            reasoning=f"Based on your interaction patterns: {insight.description}",
            confidence=insight.confidence,
            priority=1
        )
        
        return recommendation
    
    async def _create_task_adaptation(
        self, 
        insight: LearningInsight
    ) -> Optional[AdaptationRecommendation]:
        """Create task adaptation recommendation"""
        
        adaptation_id = f"task_adapt_{insight.user_id}_{int(time.time())}"
        
        recommendation = AdaptationRecommendation(
            adaptation_id=adaptation_id,
            user_id=insight.user_id,
            adaptation_type=AdaptationType.TOOL_RECOMMENDATION,
            current_value="standard_tools",
            recommended_value="optimized_tools",
            reasoning=f"Based on your task patterns: {insight.description}",
            confidence=insight.confidence,
            priority=1
        )
        
        return recommendation
    
    async def _create_time_adaptation(
        self, 
        insight: LearningInsight
    ) -> Optional[AdaptationRecommendation]:
        """Create time adaptation recommendation"""
        
        adaptation_id = f"time_adapt_{insight.user_id}_{int(time.time())}"
        
        recommendation = AdaptationRecommendation(
            adaptation_id=adaptation_id,
            user_id=insight.user_id,
            adaptation_type=AdaptationType.SUGGESTION_TIMING,
            current_value="standard_timing",
            recommended_value="optimized_timing",
            reasoning=f"Based on your time patterns: {insight.description}",
            confidence=insight.confidence,
            priority=0  # High priority for time optimizations
        )
        
        return recommendation
    
    async def _cleanup_old_learning_data(self):
        """Clean up old patterns and insights"""
        
        current_time = time.time()
        lifetime_seconds = self.config.pattern_lifetime_days * 24 * 3600
        
        # Clean up old patterns
        old_patterns = [
            p_id for p_id, pattern in self.learning_patterns.items()
            if current_time - pattern.last_observed > lifetime_seconds
        ]
        
        for pattern_id in old_patterns:
            del self.learning_patterns[pattern_id]
        
        if old_patterns:
            self.logger.info(f"Cleaned up {len(old_patterns)} old patterns")
        
        # Clean up old insights (keep for longer)
        insight_lifetime = lifetime_seconds * 2  # Keep insights twice as long
        old_insights = [
            i_id for i_id, insight in self.learning_insights.items()
            if current_time - insight.created_at > insight_lifetime
        ]
        
        for insight_id in old_insights:
            del self.learning_insights[insight_id]
        
        if old_insights:
            self.logger.info(f"Cleaned up {len(old_insights)} old insights")
    
    # Public API methods
    async def record_context(
        self, 
        user_id: str, 
        context_type: str, 
        context_data: Dict[str, Any],
        interaction_data: Optional[Dict[str, Any]] = None,
        outcome_data: Optional[Dict[str, Any]] = None
    ):
        """Record a context snapshot for learning"""
        
        snapshot = ContextSnapshot(
            timestamp=time.time(),
            user_id=user_id,
            context_type=context_type,
            context_data=context_data,
            interaction_data=interaction_data,
            outcome_data=outcome_data
        )
        
        self.context_history.append(snapshot)
        
        # Maintain history size
        if len(self.context_history) > 1000:
            self.context_history = self.context_history[-1000:]
    
    async def get_user_patterns(
        self, 
        user_id: str, 
        pattern_type: Optional[LearningPatternType] = None
    ) -> List[LearningPattern]:
        """Get learning patterns for a specific user"""
        
        user_patterns = [
            p for p in self.learning_patterns.values()
            if p.user_id == user_id
        ]
        
        if pattern_type:
            user_patterns = [p for p in user_patterns if p.pattern_type == pattern_type]
        
        return user_patterns
    
    async def get_user_insights(
        self, 
        user_id: str, 
        insight_type: Optional[str] = None
    ) -> List[LearningInsight]:
        """Get learning insights for a specific user"""
        
        user_insights = [
            i for i in self.learning_insights.values()
            if i.user_id == user_id
        ]
        
        if insight_type:
            user_insights = [i for i in user_insights if i.insight_type == insight_type]
        
        return user_insights
    
    async def get_adaptation_recommendations(
        self, 
        user_id: str, 
        adaptation_type: Optional[AdaptationType] = None
    ) -> List[AdaptationRecommendation]:
        """Get adaptation recommendations for a specific user"""
        
        user_recommendations = [
            r for r in self.adaptation_recommendations.values()
            if r.user_id == user_id
        ]
        
        if adaptation_type:
            user_recommendations = [
                r for r in user_recommendations 
                if r.adaptation_type == adaptation_type
            ]
        
        return user_recommendations
    
    async def get_learning_summary(self, user_id: str) -> Dict[str, Any]:
        """Get a summary of learning progress for a user"""
        
        patterns = await self.get_user_patterns(user_id)
        insights = await self.get_user_insights(user_id)
        recommendations = await self.get_adaptation_recommendations(user_id)
        
        summary = {
            "total_patterns": len(patterns),
            "total_insights": len(insights),
            "total_recommendations": len(recommendations),
            "pattern_types": list(set(p.pattern_type.value for p in patterns)),
            "insight_types": list(set(i.insight_type for i in insights)),
            "adaptation_types": list(set(r.adaptation_type.value for r in recommendations)),
            "learning_quality": self._calculate_learning_quality(patterns),
            "recent_activity": self._get_recent_learning_activity(user_id)
        }
        
        return summary
    
    def _calculate_learning_quality(self, patterns: List[LearningPattern]) -> float:
        """Calculate overall learning quality score"""
        
        if not patterns:
            return 0.0
        
        # Average confidence weighted by sample size
        total_weight = sum(p.sample_size for p in patterns)
        weighted_confidence = sum(
            p.confidence_score * p.sample_size for p in patterns
        )
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _get_recent_learning_activity(self, user_id: str) -> Dict[str, Any]:
        """Get recent learning activity for a user"""
        
        current_time = time.time()
        recent_threshold = 24 * 3600  # 24 hours
        
        recent_patterns = [
            p for p in self.learning_patterns.values()
            if p.user_id == user_id and 
            current_time - p.last_observed < recent_threshold
        ]
        
        recent_insights = [
            i for i in self.learning_insights.values()
            if i.user_id == user_id and 
            current_time - i.created_at < recent_threshold
        ]
        
        return {
            "patterns_updated": len(recent_patterns),
            "insights_generated": len(recent_insights),
            "last_activity": max(
                [p.last_observed for p in recent_patterns] + 
                [i.created_at for i in recent_insights],
                default=0
            )
        }
    
    async def cleanup(self):
        """Clean up resources"""
        
        # Stop the learning engine
        await self.stop_learning_engine()
        
        # Save learning data
        await self._save_learning_data()
    
    async def _save_learning_data(self):
        """Save learning data to persistent storage"""
        
        # This would integrate with persistent storage
        self.logger.info("Saving context learning data")


# Pattern matcher classes
class InteractionPatternMatcher:
    """Matches interaction patterns from context data"""
    
    async def identify_patterns(
        self, 
        user_id: str, 
        contexts: List[ContextSnapshot]
    ) -> List[LearningPattern]:
        """Identify interaction patterns"""
        
        patterns = []
        
        # Mock pattern identification - replace with real analysis
        if len(contexts) >= 5:
            pattern = LearningPattern(
                pattern_id=f"interaction_{user_id}_{int(time.time())}",
                user_id=user_id,
                pattern_type=LearningPatternType.INTERACTION_PATTERN,
                pattern_data={
                    "interaction_frequency": len(contexts) / 7,  # per day
                    "common_contexts": list(set(c.context_type for c in contexts))
                },
                confidence_score=0.8,
                sample_size=len(contexts),
                first_observed=contexts[0].timestamp,
                last_observed=contexts[-1].timestamp,
                frequency=len(contexts) / 7,
                quality=LearningQuality.HIGH
            )
            patterns.append(pattern)
        
        return patterns


class TaskPatternMatcher:
    """Matches task patterns from context data"""
    
    async def identify_patterns(
        self, 
        user_id: str, 
        contexts: List[ContextSnapshot]
    ) -> List[LearningPattern]:
        """Identify task patterns"""
        
        patterns = []
        
        # Mock pattern identification - replace with real analysis
        task_contexts = [c for c in contexts if c.context_type == "task"]
        
        if len(task_contexts) >= 3:
            pattern = LearningPattern(
                pattern_id=f"task_{user_id}_{int(time.time())}",
                user_id=user_id,
                pattern_type=LearningPatternType.TASK_PATTERN,
                pattern_data={
                    "task_count": len(task_contexts),
                    "completion_time": 45,  # Mock average completion time
                    "task_types": ["planning", "execution", "review"]
                },
                confidence_score=0.7,
                sample_size=len(task_contexts),
                first_observed=task_contexts[0].timestamp,
                last_observed=task_contexts[-1].timestamp,
                frequency=len(task_contexts) / 7,
                quality=LearningQuality.MEDIUM
            )
            patterns.append(pattern)
        
        return patterns


class TimePatternMatcher:
    """Matches time patterns from context data"""
    
    async def identify_patterns(
        self, 
        user_id: str, 
        contexts: List[ContextSnapshot]
    ) -> List[LearningPattern]:
        """Identify time patterns"""
        
        patterns = []
        
        # Mock pattern identification - replace with real analysis
        if len(contexts) >= 10:
            pattern = LearningPattern(
                pattern_id=f"time_{user_id}_{int(time.time())}",
                user_id=user_id,
                pattern_type=LearningPatternType.TIME_PATTERN,
                pattern_data={
                    "peak_hours": ["9:00", "14:00"],
                    "activity_distribution": "morning_heavy"
                },
                confidence_score=0.6,
                sample_size=len(contexts),
                first_observed=contexts[0].timestamp,
                last_observed=contexts[-1].timestamp,
                frequency=len(contexts) / 7,
                quality=LearningQuality.LOW
            )
            patterns.append(pattern)
        
        return patterns


class TopicPatternMatcher:
    """Matches topic patterns from context data"""
    
    async def identify_patterns(
        self, 
        user_id: str, 
        contexts: List[ContextSnapshot]
    ) -> List[LearningPattern]:
        """Identify topic patterns"""
        
        patterns = []
        
        # Mock pattern identification - replace with real analysis
        if len(contexts) >= 5:
            pattern = LearningPattern(
                pattern_id=f"topic_{user_id}_{int(time.time())}",
                user_id=user_id,
                pattern_type=LearningPatternType.TOPIC_PATTERN,
                pattern_data={
                    "topics": ["work", "learning", "personal"],
                    "topic_preferences": {"work": 0.6, "learning": 0.3, "personal": 0.1}
                },
                confidence_score=0.7,
                sample_size=len(contexts),
                first_observed=contexts[0].timestamp,
                last_observed=contexts[-1].timestamp,
                frequency=len(contexts) / 7,
                quality=LearningQuality.MEDIUM
            )
            patterns.append(pattern)
        
        return patterns


class BehaviorPatternMatcher:
    """Matches behavior patterns from context data"""
    
    async def identify_patterns(
        self, 
        user_id: str, 
        contexts: List[ContextSnapshot]
    ) -> List[LearningPattern]:
        """Identify behavior patterns"""
        
        patterns = []
        
        # Mock pattern identification - replace with real analysis
        if len(contexts) >= 8:
            pattern = LearningPattern(
                pattern_id=f"behavior_{user_id}_{int(time.time())}",
                user_id=user_id,
                pattern_type=LearningPatternType.BEHAVIOR_PATTERN,
                pattern_data={
                    "response_time": "immediate",
                    "preference_consistency": "high"
                },
                confidence_score=0.8,
                sample_size=len(contexts),
                first_observed=contexts[0].timestamp,
                last_observed=contexts[-1].timestamp,
                frequency=len(contexts) / 7,
                quality=LearningQuality.HIGH
            )
            patterns.append(pattern)
        
        return patterns


class PreferencePatternMatcher:
    """Matches preference patterns from context data"""
    
    async def identify_patterns(
        self, 
        user_id: str, 
        contexts: List[ContextSnapshot]
    ) -> List[LearningPattern]:
        """Identify preference patterns"""
        
        patterns = []
        
        # Mock pattern identification - replace with real analysis
        if len(contexts) >= 6:
            pattern = LearningPattern(
                pattern_id=f"preference_{user_id}_{int(time.time())}",
                user_id=user_id,
                pattern_type=LearningPatternType.PREFERENCE_PATTERN,
                pattern_data={
                    "response_style": "detailed",
                    "interaction_frequency": "high"
                },
                confidence_score=0.7,
                sample_size=len(contexts),
                first_observed=contexts[0].timestamp,
                last_observed=contexts[-1].timestamp,
                frequency=len(contexts) / 7,
                quality=LearningQuality.MEDIUM
            )
            patterns.append(pattern)
        
        return patterns


# Factory function for creating context learners
def create_context_learner(config: Optional[ContextLearnerConfig] = None) -> ContextLearner:
    """Create a new context learner instance"""
    
    if config is None:
        config = ContextLearnerConfig()
    
    return ContextLearner(config)
