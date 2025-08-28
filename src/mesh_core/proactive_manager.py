#!/usr/bin/env python3
"""
Proactive Manager - Enhanced with Sentient's Proactive Management Concepts

This module integrates Sentient's proactive capabilities into The Mesh,
providing anticipatory actions, intelligent suggestions, and automated assistance.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import sys

# Try to import Sentient's proactive concepts
try:
    sys.path.append('/Users/admin/AI/Sentient/src/server/main')
    from proactivity.learning import record_user_feedback
    SENTIENT_PROACTIVE_AVAILABLE = True
except ImportError:
    SENTIENT_PROACTIVE_AVAILABLE = False

# Enums for proactive management
class SuggestionType(Enum):
    """Types of proactive suggestions"""
    TASK_REMINDER = "task_reminder"
    SCHEDULE_OPTIMIZATION = "schedule_optimization"
    INFORMATION_UPDATE = "information_update"
    WELLNESS_CHECK = "wellness_check"
    RELATIONSHIP_MAINTENANCE = "relationship_maintenance"
    LEARNING_OPPORTUNITY = "learning_opportunity"
    FINANCIAL_REMINDER = "financial_reminder"
    GENERIC = "generic"

class SuggestionStatus(Enum):
    """Status of proactive suggestions"""
    PENDING = "pending"
    APPROVED = "approved"
    DISMISSED = "dismissed"
    EXECUTED = "executed"
    EXPIRED = "expired"

class ProactivityLevel(Enum):
    """Levels of proactive behavior"""
    CONSERVATIVE = "conservative"  # Minimal suggestions
    BALANCED = "balanced"          # Moderate suggestions
    AGGRESSIVE = "aggressive"      # Frequent suggestions

# Dataclasses for proactive management
@dataclass
class ProactiveSuggestion:
    """A proactive suggestion for the user"""
    suggestion_id: str
    user_id: str
    suggestion_type: SuggestionType
    title: str
    description: str
    reasoning: str
    suggested_actions: List[str] = field(default_factory=list)
    priority: int = 1  # 0=high, 1=medium, 2=low
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    status: SuggestionStatus = SuggestionStatus.PENDING
    user_feedback: Optional[str] = None  # "positive", "negative", or None
    execution_result: Optional[Dict[str, Any]] = None

@dataclass
class ContextSnapshot:
    """Snapshot of current context for proactive analysis"""
    timestamp: float
    user_id: str
    current_time: str
    location: Optional[str] = None
    current_activity: Optional[str] = None
    active_tasks: List[str] = field(default_factory=list)
    recent_interactions: List[str] = field(default_factory=list)
    available_tools: List[str] = field(default_factory=list)
    user_mood: Optional[str] = None
    environmental_factors: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProactiveConfig:
    """Configuration for proactive behavior"""
    enabled: bool = True
    proactivity_level: ProactivityLevel = ProactivityLevel.BALANCED
    max_suggestions_per_day: int = 10
    suggestion_lifetime_hours: int = 24
    learning_enabled: bool = True
    feedback_threshold: float = 0.7  # Minimum confidence for suggestions
    context_gathering_interval: int = 300  # 5 minutes in seconds

@dataclass
class ProactiveResult:
    """Result of proactive analysis"""
    suggestions_generated: int
    suggestions_executed: int
    user_satisfaction_score: float
    context_quality_score: float
    learning_insights: List[str] = field(default_factory=list)

class ProactiveManager:
    """
    Enhanced Proactive Manager with Sentient's Proactive Management Concepts
    
    Provides anticipatory actions, intelligent suggestions, and automated assistance
    while maintaining Mesh's local-first architecture.
    """
    
    def __init__(self, config: ProactiveConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # State management
        self.suggestions: Dict[str, ProactiveSuggestion] = {}
        self.context_history: List[ContextSnapshot] = []
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        self.learning_data: Dict[str, Any] = {}
        
        # Sentient integration status
        self.sentient_available = SENTIENT_PROACTIVE_AVAILABLE
        
        # Proactive engine state
        self._is_running = False
        self._engine_task: Optional[asyncio.Task] = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize proactive management components"""
        
        if self.sentient_available:
            self.logger.info("Sentient proactive concepts available")
        else:
            self.logger.info("Using mock proactive concepts for development")
    
    async def start_proactive_engine(self):
        """Start the proactive management engine"""
        
        if self._is_running:
            self.logger.warning("Proactive engine is already running")
            return
        
        self.logger.info("Starting proactive management engine...")
        self._is_running = True
        self._engine_task = asyncio.create_task(self._run_proactive_loop())
    
    async def stop_proactive_engine(self):
        """Stop the proactive management engine"""
        
        if not self._is_running:
            self.logger.warning("Proactive engine is not running")
            return
        
        self.logger.info("Stopping proactive management engine...")
        self._is_running = False
        
        if self._engine_task:
            self._engine_task.cancel()
            try:
                await self._engine_task
            except asyncio.CancelledError:
                self.logger.info("Proactive engine task cancelled successfully")
            except Exception as e:
                self.logger.error(f"Error while stopping proactive engine: {e}")
        
        self._engine_task = None
    
    async def _run_proactive_loop(self):
        """Main loop for proactive management"""
        
        self.logger.info("Proactive engine started")
        
        while self._is_running:
            try:
                # Gather context from all active users
                await self._gather_context_for_all_users()
                
                # Analyze context and generate suggestions
                await self._analyze_context_and_suggest()
                
                # Execute approved suggestions
                await self._execute_approved_suggestions()
                
                # Learn from feedback and results
                await self._learn_from_feedback()
                
                # Clean up expired suggestions
                await self._cleanup_expired_suggestions()
                
                # Wait for next cycle
                await asyncio.sleep(self.config.context_gathering_interval)
                
            except Exception as e:
                self.logger.error(f"Error in proactive loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _gather_context_for_all_users(self):
        """Gather context for all active users"""
        
        # This would integrate with user management system
        # For now, use mock users
        mock_users = ["user1", "user2", "admin"]
        
        for user_id in mock_users:
            try:
                context = await self._gather_user_context(user_id)
                if context:
                    self.context_history.append(context)
                    
                    # Maintain history size
                    if len(self.context_history) > 100:
                        self.context_history = self.context_history[-100:]
                        
            except Exception as e:
                self.logger.error(f"Error gathering context for user {user_id}: {e}")
    
    async def _gather_user_context(self, user_id: str) -> Optional[ContextSnapshot]:
        """Gather context for a specific user"""
        
        try:
            # This would integrate with various Mesh systems
            current_time = time.strftime("%Y-%m-%d %H:%M:%S UTC")
            
            # Mock context gathering - replace with real integrations
            context = ContextSnapshot(
                timestamp=time.time(),
                user_id=user_id,
                current_time=current_time,
                location="Home",  # Mock location
                current_activity="Working",  # Mock activity
                active_tasks=self._get_mock_active_tasks(user_id),
                recent_interactions=self._get_mock_recent_interactions(user_id),
                available_tools=self._get_mock_available_tools(user_id),
                user_mood="focused",  # Mock mood
                environmental_factors={
                    "time_of_day": "afternoon",
                    "day_of_week": "wednesday",
                    "weather": "sunny"
                }
            )
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error gathering context for user {user_id}: {e}")
            return None
    
    def _get_mock_active_tasks(self, user_id: str) -> List[str]:
        """Get mock active tasks for development"""
        
        mock_tasks = {
            "user1": ["Project planning", "Email review"],
            "user2": ["Meeting preparation", "Document review"],
            "admin": ["System maintenance", "User support"]
        }
        
        return mock_tasks.get(user_id, [])
    
    def _get_mock_recent_interactions(self, user_id: str) -> List[str]:
        """Get mock recent interactions for development"""
        
        mock_interactions = {
            "user1": ["Asked about project status", "Requested meeting schedule"],
            "user2": ["Searched for documents", "Created new task"],
            "admin": ["Reviewed system logs", "Updated user permissions"]
        }
        
        return mock_interactions.get(user_id, [])
    
    def _get_mock_available_tools(self, user_id: str) -> List[str]:
        """Get mock available tools for development"""
        
        return ["memory", "tasks", "voice", "search", "files"]
    
    async def _analyze_context_and_suggest(self):
        """Analyze context and generate proactive suggestions"""
        
        if not self.config.enabled:
            return
        
        # Analyze recent context for each user
        recent_contexts = self.context_history[-20:]  # Last 20 context snapshots
        
        for context in recent_contexts:
            try:
                # Check if we should generate suggestions for this user
                if self._should_generate_suggestions(context):
                    suggestions = await self._generate_suggestions_for_context(context)
                    
                    # Add suggestions to the system
                    for suggestion in suggestions:
                        self.suggestions[suggestion.suggestion_id] = suggestion
                        
                        self.logger.info(
                            f"Generated suggestion for user {context.user_id}: "
                            f"{suggestion.title}"
                        )
                        
            except Exception as e:
                self.logger.error(f"Error analyzing context for user {context.user_id}: {e}")
    
    def _should_generate_suggestions(self, context: ContextSnapshot) -> bool:
        """Determine if suggestions should be generated for this context"""
        
        # Check daily limit
        user_suggestions_today = len([
            s for s in self.suggestions.values()
            if s.user_id == context.user_id and 
            s.created_at > time.time() - 86400  # 24 hours
        ])
        
        if user_suggestions_today >= self.config.max_suggestions_per_day:
            return False
        
        # Check if user is receptive to suggestions
        user_prefs = self.user_preferences.get(context.user_id, {})
        if user_prefs.get("proactivity_enabled", True) is False:
            return False
        
        # Check context quality
        if self._calculate_context_quality(context) < self.config.feedback_threshold:
            return False
        
        return True
    
    def _calculate_context_quality(self, context: ContextSnapshot) -> float:
        """Calculate the quality of context for proactive analysis"""
        
        score = 0.5  # Base score
        
        # Boost for recent context
        if time.time() - context.timestamp < 300:  # 5 minutes
            score += 0.2
        
        # Boost for rich context
        if context.current_activity:
            score += 0.1
        if context.active_tasks:
            score += 0.1
        if context.recent_interactions:
            score += 0.1
        
        # Boost for environmental factors
        if context.environmental_factors:
            score += 0.1
        
        return min(score, 1.0)
    
    async def _generate_suggestions_for_context(
        self, 
        context: ContextSnapshot
    ) -> List[ProactiveSuggestion]:
        """Generate proactive suggestions based on context"""
        
        suggestions = []
        
        # Use Sentient's approach if available
        if self.sentient_available:
            sentient_suggestions = await self._generate_sentient_suggestions(context)
            suggestions.extend(sentient_suggestions)
        else:
            # Generate basic suggestions
            basic_suggestions = self._generate_basic_suggestions(context)
            suggestions.extend(basic_suggestions)
        
        # Limit suggestions based on proactivity level
        max_suggestions = {
            ProactivityLevel.CONSERVATIVE: 1,
            ProactivityLevel.BALANCED: 2,
            ProactivityLevel.AGGRESSIVE: 3
        }.get(self.config.proactivity_level, 2)
        
        return suggestions[:max_suggestions]
    
    async def _generate_sentient_suggestions(
        self, 
        context: ContextSnapshot
    ) -> List[ProactiveSuggestion]:
        """Generate suggestions using Sentient's proactive concepts"""
        
        suggestions = []
        
        try:
            # Analyze context for different suggestion types
            if self._should_suggest_task_reminder(context):
                suggestion = await self._create_task_reminder_suggestion(context)
                if suggestion:
                    suggestions.append(suggestion)
            
            if self._should_suggest_schedule_optimization(context):
                suggestion = await self._create_schedule_optimization_suggestion(context)
                if suggestion:
                    suggestions.append(suggestion)
            
            if self._should_suggest_wellness_check(context):
                suggestion = await self._create_wellness_check_suggestion(context)
                if suggestion:
                    suggestions.append(suggestion)
            
            if self._should_suggest_learning_opportunity(context):
                suggestion = await self._create_learning_opportunity_suggestion(context)
                if suggestion:
                    suggestions.append(suggestion)
            
        except Exception as e:
            self.logger.warning(f"Sentient suggestion generation failed: {e}")
        
        return suggestions
    
    def _should_suggest_task_reminder(self, context: ContextSnapshot) -> bool:
        """Determine if task reminder suggestion should be made"""
        
        # Suggest if user has many active tasks
        if len(context.active_tasks) > 3:
            return True
        
        # Suggest if user has been inactive for a while
        if context.recent_interactions and len(context.recent_interactions) < 2:
            return True
        
        return False
    
    def _should_suggest_schedule_optimization(self, context: ContextSnapshot) -> bool:
        """Determine if schedule optimization suggestion should be made"""
        
        # Suggest during planning times
        time_of_day = context.environmental_factors.get("time_of_day", "")
        if time_of_day in ["morning", "evening"]:
            return True
        
        # Suggest if user has conflicting tasks
        if len(context.active_tasks) > 5:
            return True
        
        return False
    
    def _should_suggest_wellness_check(self, context: ContextSnapshot) -> bool:
        """Determine if wellness check suggestion should be made"""
        
        # Suggest during wellness-appropriate times
        time_of_day = context.environmental_factors.get("time_of_day", "")
        if time_of_day in ["morning", "afternoon"]:
            return True
        
        # Suggest if user has been working for a long time
        if context.current_activity == "Working" and len(context.active_tasks) > 4:
            return True
        
        return False
    
    def _should_suggest_learning_opportunity(self, context: ContextSnapshot) -> bool:
        """Determine if learning opportunity suggestion should be made"""
        
        # Suggest if user has been focused on similar tasks
        if context.recent_interactions and len(context.recent_interactions) > 3:
            return True
        
        # Suggest during learning-appropriate times
        time_of_day = context.environmental_factors.get("time_of_day", "")
        if time_of_day in ["morning", "afternoon"]:
            return True
        
        return False
    
    async def _create_task_reminder_suggestion(
        self, 
        context: ContextSnapshot
    ) -> Optional[ProactiveSuggestion]:
        """Create a task reminder suggestion"""
        
        suggestion_id = f"task_reminder_{context.user_id}_{int(time.time())}"
        
        suggestion = ProactiveSuggestion(
            suggestion_id=suggestion_id,
            user_id=context.user_id,
            suggestion_type=SuggestionType.TASK_REMINDER,
            title="Task Management Reminder",
            description=f"You have {len(context.active_tasks)} active tasks. Would you like me to help you prioritize or organize them?",
            reasoning="Multiple active tasks detected, suggesting task organization assistance",
            suggested_actions=[
                "Review and prioritize active tasks",
                "Break down complex tasks into smaller steps",
                "Schedule dedicated time for task completion"
            ],
            priority=1,
            expires_at=time.time() + (self.config.suggestion_lifetime_hours * 3600)
        )
        
        return suggestion
    
    async def _create_schedule_optimization_suggestion(
        self, 
        context: ContextSnapshot
    ) -> Optional[ProactiveSuggestion]:
        """Create a schedule optimization suggestion"""
        
        suggestion_id = f"schedule_opt_{context.user_id}_{int(time.time())}"
        
        suggestion = ProactiveSuggestion(
            suggestion_id=suggestion_id,
            user_id=context.user_id,
            suggestion_type=SuggestionType.SCHEDULE_OPTIMIZATION,
            title="Schedule Optimization Opportunity",
            description="I notice you have several tasks scheduled. Would you like me to help optimize your schedule for better productivity?",
            reasoning="Multiple tasks detected, suggesting schedule optimization",
            suggested_actions=[
                "Analyze task dependencies and conflicts",
                "Suggest optimal task ordering",
                "Identify time blocks for focused work"
            ],
            priority=2,
            expires_at=time.time() + (self.config.suggestion_lifetime_hours * 3600)
        )
        
        return suggestion
    
    async def _create_wellness_check_suggestion(
        self, 
        context: ContextSnapshot
    ) -> Optional[ProactiveSuggestion]:
        """Create a wellness check suggestion"""
        
        suggestion_id = f"wellness_{context.user_id}_{int(time.time())}"
        
        suggestion = ProactiveSuggestion(
            suggestion_id=suggestion_id,
            user_id=context.user_id,
            suggestion_type=SuggestionType.WELLNESS_CHECK,
            title="Wellness Check-in",
            description="You've been working on several tasks. How are you feeling? Would you like to take a short break?",
            reasoning="Extended work session detected, suggesting wellness check",
            suggested_actions=[
                "Take a 5-minute break",
                "Practice deep breathing exercises",
                "Review your current energy level"
            ],
            priority=2,
            expires_at=time.time() + (self.config.suggestion_lifetime_hours * 3600)
        )
        
        return suggestion
    
    async def _create_learning_opportunity_suggestion(
        self, 
        context: ContextSnapshot
    ) -> Optional[ProactiveSuggestion]:
        """Create a learning opportunity suggestion"""
        
        suggestion_id = f"learning_{context.user_id}_{int(time.time())}"
        
        suggestion = ProactiveSuggestion(
            suggestion_id=suggestion_id,
            user_id=context.user_id,
            suggestion_type=SuggestionType.LEARNING_OPPORTUNITY,
            title="Learning Opportunity Detected",
            description="I notice you've been working on similar tasks. Would you like me to suggest some learning resources to improve your efficiency?",
            reasoning="Pattern of similar tasks detected, suggesting learning opportunity",
            suggested_actions=[
                "Review task patterns and identify learning areas",
                "Suggest relevant tutorials or documentation",
                "Create a learning plan for skill improvement"
            ],
            priority=2,
            expires_at=time.time() + (self.config.suggestion_lifetime_hours * 3600)
        )
        
        return suggestion
    
    def _generate_basic_suggestions(
        self, 
        context: ContextSnapshot
    ) -> List[ProactiveSuggestion]:
        """Generate basic suggestions when Sentient is not available"""
        
        suggestions = []
        
        # Basic task reminder
        if len(context.active_tasks) > 2:
            suggestion = ProactiveSuggestion(
                suggestion_id=f"basic_task_{context.user_id}_{int(time.time())}",
                user_id=context.user_id,
                suggestion_type=SuggestionType.TASK_REMINDER,
                title="Task Overview",
                description=f"You have {len(context.active_tasks)} active tasks. Consider reviewing your priorities.",
                reasoning="Multiple active tasks detected",
                suggested_actions=["Review task list", "Update priorities"],
                priority=1,
                expires_at=time.time() + (self.config.suggestion_lifetime_hours * 3600)
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    async def _execute_approved_suggestions(self):
        """Execute suggestions that have been approved by users"""
        
        approved_suggestions = [
            s for s in self.suggestions.values()
            if s.status == SuggestionStatus.APPROVED
        ]
        
        for suggestion in approved_suggestions:
            try:
                result = await self._execute_suggestion(suggestion)
                suggestion.execution_result = result
                suggestion.status = SuggestionStatus.EXECUTED
                
                self.logger.info(f"Executed suggestion {suggestion.suggestion_id}")
                
            except Exception as e:
                self.logger.error(f"Error executing suggestion {suggestion.suggestion_id}: {e}")
                suggestion.status = SuggestionStatus.EXPIRED
    
    async def _execute_suggestion(self, suggestion: ProactiveSuggestion) -> Dict[str, Any]:
        """Execute a specific suggestion"""
        
        try:
            # This would integrate with various Mesh systems
            execution_result = {
                "executed_at": time.time(),
                "success": True,
                "actions_taken": [],
                "results": {}
            }
            
            # Execute suggested actions based on suggestion type
            if suggestion.suggestion_type == SuggestionType.TASK_REMINDER:
                result = await self._execute_task_reminder(suggestion)
                execution_result["actions_taken"].append("task_reminder")
                execution_result["results"]["task_reminder"] = result
            
            elif suggestion.suggestion_type == SuggestionType.SCHEDULE_OPTIMIZATION:
                result = await self._execute_schedule_optimization(suggestion)
                execution_result["actions_taken"].append("schedule_optimization")
                execution_result["results"]["schedule_optimization"] = result
            
            elif suggestion.suggestion_type == SuggestionType.WELLNESS_CHECK:
                result = await self._execute_wellness_check(suggestion)
                execution_result["actions_taken"].append("wellness_check")
                execution_result["results"]["wellness_check"] = result
            
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Error executing suggestion: {e}")
            return {
                "executed_at": time.time(),
                "success": False,
                "error": str(e)
            }
    
    async def _execute_task_reminder(self, suggestion: ProactiveSuggestion) -> Dict[str, Any]:
        """Execute task reminder suggestion"""
        
        # This would integrate with the task system
        return {
            "action": "task_reminder_sent",
            "timestamp": time.time(),
            "details": "Task reminder notification sent to user"
        }
    
    async def _execute_schedule_optimization(self, suggestion: ProactiveSuggestion) -> Dict[str, Any]:
        """Execute schedule optimization suggestion"""
        
        # This would integrate with the scheduling system
        return {
            "action": "schedule_analysis_completed",
            "timestamp": time.time(),
            "details": "Schedule analysis completed and optimization suggestions generated"
        }
    
    async def _execute_wellness_check(self, suggestion: ProactiveSuggestion) -> Dict[str, Any]:
        """Execute wellness check suggestion"""
        
        # This would integrate with wellness/health systems
        return {
            "action": "wellness_check_initiated",
            "timestamp": time.time(),
            "details": "Wellness check initiated for user"
        }
    
    async def _learn_from_feedback(self):
        """Learn from user feedback and execution results"""
        
        if not self.config.learning_enabled:
            return
        
        # Analyze feedback patterns
        for suggestion in self.suggestions.values():
            if suggestion.user_feedback:
                await self._process_user_feedback(suggestion)
        
        # Analyze execution results
        executed_suggestions = [
            s for s in self.suggestions.values()
            if s.status == SuggestionStatus.EXECUTED
        ]
        
        for suggestion in executed_suggestions:
            await self._process_execution_result(suggestion)
    
    async def _process_user_feedback(self, suggestion: ProactiveSuggestion):
        """Process user feedback for learning"""
        
        try:
            if self.sentient_available:
                # Use Sentient's feedback recording if available
                await record_user_feedback(
                    suggestion.user_id,
                    suggestion.suggestion_type.value,
                    suggestion.user_feedback
                )
            else:
                # Local feedback processing
                await self._record_local_feedback(suggestion)
                
        except Exception as e:
            self.logger.warning(f"Error processing feedback: {e}")
    
    async def _record_local_feedback(self, suggestion: ProactiveSuggestion):
        """Record feedback locally for learning"""
        
        user_id = suggestion.user_id
        suggestion_type = suggestion.suggestion_type.value
        feedback = suggestion.user_feedback
        
        if user_id not in self.learning_data:
            self.learning_data[user_id] = {}
        
        if suggestion_type not in self.learning_data[user_id]:
            self.learning_data[user_id][suggestion_type] = {
                "positive": 0,
                "negative": 0,
                "total": 0
            }
        
        learning_entry = self.learning_data[user_id][suggestion_type]
        learning_entry["total"] += 1
        
        if feedback == "positive":
            learning_entry["positive"] += 1
        elif feedback == "negative":
            learning_entry["negative"] += 1
    
    async def _process_execution_result(self, suggestion: ProactiveSuggestion):
        """Process execution results for learning"""
        
        if not suggestion.execution_result:
            return
        
        # Learn from successful executions
        if suggestion.execution_result.get("success", False):
            await self._learn_from_success(suggestion)
        else:
            await self._learn_from_failure(suggestion)
    
    async def _learn_from_success(self, suggestion: ProactiveSuggestion):
        """Learn from successful suggestion execution"""
        
        # This would update learning models and improve future suggestions
        self.logger.debug(f"Learning from successful execution of suggestion {suggestion.suggestion_id}")
    
    async def _learn_from_failure(self, suggestion: ProactiveSuggestion):
        """Learn from failed suggestion execution"""
        
        # This would update learning models to avoid similar failures
        self.logger.debug(f"Learning from failed execution of suggestion {suggestion.suggestion_id}")
    
    async def _cleanup_expired_suggestions(self):
        """Clean up expired suggestions"""
        
        current_time = time.time()
        expired_suggestions = [
            s_id for s_id, suggestion in self.suggestions.items()
            if suggestion.expires_at and suggestion.expires_at < current_time
        ]
        
        for suggestion_id in expired_suggestions:
            suggestion = self.suggestions[suggestion_id]
            if suggestion.status == SuggestionStatus.PENDING:
                suggestion.status = SuggestionStatus.EXPIRED
                self.logger.info(f"Expired suggestion {suggestion_id}")
    
    # Public API methods
    async def get_user_suggestions(
        self, 
        user_id: str, 
        status: Optional[SuggestionStatus] = None
    ) -> List[ProactiveSuggestion]:
        """Get suggestions for a specific user"""
        
        user_suggestions = [
            s for s in self.suggestions.values()
            if s.user_id == user_id
        ]
        
        if status:
            user_suggestions = [s for s in user_suggestions if s.status == status]
        
        return user_suggestions
    
    async def approve_suggestion(self, suggestion_id: str, user_id: str) -> bool:
        """Approve a suggestion for execution"""
        
        if suggestion_id not in self.suggestions:
            return False
        
        suggestion = self.suggestions[suggestion_id]
        if suggestion.user_id != user_id:
            return False
        
        suggestion.status = SuggestionStatus.APPROVED
        self.logger.info(f"User {user_id} approved suggestion {suggestion_id}")
        
        return True
    
    async def dismiss_suggestion(self, suggestion_id: str, user_id: str) -> bool:
        """Dismiss a suggestion"""
        
        if suggestion_id not in self.suggestions:
            return False
        
        suggestion = self.suggestions[suggestion_id]
        if suggestion.user_id != user_id:
            return False
        
        suggestion.status = SuggestionStatus.DISMISSED
        suggestion.user_feedback = "negative"
        self.logger.info(f"User {user_id} dismissed suggestion {suggestion_id}")
        
        return True
    
    async def provide_feedback(
        self, 
        suggestion_id: str, 
        user_id: str, 
        feedback: str
    ) -> bool:
        """Provide feedback on a suggestion"""
        
        if suggestion_id not in self.suggestions:
            return False
        
        suggestion = self.suggestions[suggestion_id]
        if suggestion.user_id != user_id:
            return False
        
        if feedback not in ["positive", "negative"]:
            return False
        
        suggestion.user_feedback = feedback
        self.logger.info(f"User {user_id} provided {feedback} feedback on suggestion {suggestion_id}")
        
        return True
    
    async def get_proactive_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about proactive behavior for a user"""
        
        user_suggestions = await self.get_user_suggestions(user_id)
        
        insights = {
            "total_suggestions": len(user_suggestions),
            "pending_suggestions": len([s for s in user_suggestions if s.status == SuggestionStatus.PENDING]),
            "approved_suggestions": len([s for s in user_suggestions if s.status == SuggestionStatus.APPROVED]),
            "dismissed_suggestions": len([s for s in user_suggestions if s.status == SuggestionStatus.DISMISSED]),
            "executed_suggestions": len([s for s in user_suggestions if s.status == SuggestionStatus.EXECUTED]),
            "feedback_distribution": self._calculate_feedback_distribution(user_suggestions),
            "suggestion_type_distribution": self._calculate_type_distribution(user_suggestions),
            "learning_insights": self.learning_data.get(user_id, {})
        }
        
        return insights
    
    def _calculate_feedback_distribution(self, suggestions: List[ProactiveSuggestion]) -> Dict[str, int]:
        """Calculate distribution of user feedback"""
        
        distribution = {"positive": 0, "negative": 0, "none": 0}
        
        for suggestion in suggestions:
            feedback = suggestion.user_feedback or "none"
            distribution[feedback] += 1
        
        return distribution
    
    def _calculate_type_distribution(self, suggestions: List[ProactiveSuggestion]) -> Dict[str, int]:
        """Calculate distribution of suggestion types"""
        
        distribution = {}
        
        for suggestion in suggestions:
            suggestion_type = suggestion.suggestion_type.value
            distribution[suggestion_type] = distribution.get(suggestion_type, 0) + 1
        
        return distribution
    
    async def cleanup(self):
        """Clean up resources"""
        
        # Stop the proactive engine
        await self.stop_proactive_engine()
        
        # Save learning data
        await self._save_learning_data()
    
    async def _save_learning_data(self):
        """Save learning data to persistent storage"""
        
        # This would integrate with persistent storage
        self.logger.info("Saving proactive learning data")


# Factory function for creating proactive managers
def create_proactive_manager(config: Optional[ProactiveConfig] = None) -> ProactiveManager:
    """Create a new proactive manager instance"""
    
    if config is None:
        config = ProactiveConfig()
    
    return ProactiveManager(config)
