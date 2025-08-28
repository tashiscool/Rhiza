#!/usr/bin/env python3
"""
Personal Agent - Enhanced with Sentient's Personal AI Concepts

This module integrates Sentient's personal AI capabilities into The Mesh,
providing adaptive responses, user modeling, and context-aware interactions.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import sys

# Try to import Sentient's personal AI concepts
try:
    sys.path.append('/Users/admin/AI/Sentient/src/server/main')
    from chat.prompts import STAGE_2_SYSTEM_PROMPT
    from memories.constants import TOPICS
    SENTIENT_PERSONAL_AI_AVAILABLE = True
except ImportError:
    SENTIENT_PERSONAL_AI_AVAILABLE = False
    # Mock constants for development/testing
    TOPICS = [
        {"name": "Personal Identity", "description": "Core traits, personality, beliefs, values, ethics, and preferences"},
        {"name": "Interests & Lifestyle", "description": "Hobbies, recreational activities, habits, routines, daily behavior"},
        {"name": "Work & Learning", "description": "Career, jobs, professional achievements, academic background, skills, certifications"},
        {"name": "Health & Wellbeing", "description": "Mental and physical health, self-care practices"},
        {"name": "Relationships & Social Life", "description": "Family, friends, romantic connections, social interactions, social media"},
        {"name": "Financial", "description": "Income, expenses, investments, financial goals"},
        {"name": "Goals & Challenges", "description": "Aspirations, objectives, obstacles, and difficulties faced"},
        {"name": "Miscellaneous", "description": "Anything that doesn't clearly fit into the above"}
    ]

# Enums for personal AI concepts
class InteractionType(Enum):
    """Types of user interactions"""
    TASK_REQUEST = "task_request"
    INFORMATION_QUERY = "information_query"
    CONVERSATION = "conversation"
    VOICE_COMMAND = "voice_command"
    PROACTIVE_SUGGESTION = "proactive_suggestion"

class ResponseStyle(Enum):
    """Response style preferences"""
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    FRIENDLY = "friendly"
    CONCISE = "concise"
    DETAILED = "detailed"

class UserMood(Enum):
    """User mood indicators"""
    HAPPY = "happy"
    STRESSED = "stressed"
    FOCUSED = "focused"
    RELAXED = "relaxed"
    BUSY = "busy"
    CURIOUS = "curious"

# Dataclasses for personal AI components
@dataclass
class UserPreference:
    """User preference configuration"""
    response_style: ResponseStyle = ResponseStyle.FRIENDLY
    detail_level: str = "balanced"  # concise, balanced, detailed
    formality: str = "casual"  # formal, casual, mixed
    timezone: str = "UTC"
    language: str = "en"
    notification_preferences: Dict[str, bool] = field(default_factory=lambda: {
        "email": True,
        "push": True,
        "voice": False
    })

@dataclass
class UserContext:
    """Current user context for personalization"""
    user_id: str
    current_time: str
    location: Optional[str] = None
    mood: Optional[UserMood] = None
    current_activity: Optional[str] = None
    recent_topics: List[str] = field(default_factory=list)
    active_tasks: List[str] = field(default_factory=list)
    available_tools: List[str] = field(default_factory=list)

@dataclass
class PersonalizedResponse:
    """Personalized response with context awareness"""
    content: str
    style: ResponseStyle
    detail_level: str
    includes_personal_context: bool
    suggested_actions: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    confidence_score: float = 0.0

@dataclass
class AdaptiveBehavior:
    """Adaptive behavior configuration"""
    learning_enabled: bool = True
    context_adaptation: bool = True
    mood_awareness: bool = True
    preference_evolution: bool = True
    interaction_history_size: int = 100

@dataclass
class PersonalAgentConfig:
    """Configuration for the personal agent"""
    enable_adaptive_responses: bool = True
    enable_context_learning: bool = True
    enable_mood_detection: bool = True
    enable_preference_evolution: bool = True
    max_context_history: int = 50
    response_personalization_threshold: float = 0.7

class PersonalAgent:
    """
    Enhanced Personal Agent with Sentient's Personal AI Concepts
    
    Provides adaptive responses, user modeling, and context-aware interactions
    while maintaining Mesh's local-first architecture.
    """
    
    def __init__(self, config: PersonalAgentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # User state
        self.user_preferences: Dict[str, UserPreference] = {}
        self.interaction_history: List[Dict[str, Any]] = []
        self.context_patterns: Dict[str, Any] = {}
        
        # Sentient integration status
        self.sentient_available = SENTIENT_PERSONAL_AI_AVAILABLE
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize personal AI components"""
        
        if self.sentient_available:
            self.logger.info("Sentient personal AI concepts available")
        else:
            self.logger.info("Using mock personal AI concepts for development")
    
    async def process_interaction(
        self,
        user_id: str,
        interaction_type: InteractionType,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> PersonalizedResponse:
        """
        Process a user interaction and generate a personalized response
        
        Args:
            user_id: Unique identifier for the user
            interaction_type: Type of interaction
            content: User's input content
            context: Additional context information
            
        Returns:
            PersonalizedResponse with context-aware content
        """
        
        try:
            # Get or create user preferences
            user_prefs = await self._get_user_preferences(user_id)
            
            # Build user context
            user_context = await self._build_user_context(user_id, context)
            
            # Analyze interaction for personalization opportunities
            personalization_data = await self._analyze_interaction(
                content, interaction_type, user_context
            )
            
            # Generate personalized response
            response = await self._generate_personalized_response(
                content, interaction_type, user_prefs, user_context, personalization_data
            )
            
            # Update interaction history and learn from interaction
            await self._update_interaction_history(user_id, interaction_type, content, response)
            await self._learn_from_interaction(user_id, content, response, user_context)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing interaction: {e}")
            # Return fallback response
            return PersonalizedResponse(
                content="I'm having trouble processing that right now. Let me try a different approach.",
                style=ResponseStyle.FRIENDLY,
                detail_level="concise",
                includes_personal_context=False,
                confidence_score=0.0
            )
    
    async def _get_user_preferences(self, user_id: str) -> UserPreference:
        """Get or create user preferences"""
        
        if user_id not in self.user_preferences:
            # Create default preferences
            self.user_preferences[user_id] = UserPreference()
            
            # Try to load from persistent storage if available
            await self._load_user_preferences(user_id)
        
        return self.user_preferences[user_id]
    
    async def _build_user_context(
        self, 
        user_id: str, 
        context: Optional[Dict[str, Any]]
    ) -> UserContext:
        """Build comprehensive user context"""
        
        # Start with basic context
        user_context = UserContext(
            user_id=user_id,
            current_time=time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            recent_topics=self._extract_recent_topics(user_id),
            active_tasks=self._get_active_tasks(user_id),
            available_tools=self._get_available_tools(user_id)
        )
        
        # Enhance with provided context
        if context:
            if "location" in context:
                user_context.location = context["location"]
            if "mood" in context:
                try:
                    user_context.mood = UserMood(context["mood"])
                except ValueError:
                    pass
            if "current_activity" in context:
                user_context.current_activity = context["current_activity"]
        
        # Analyze mood from recent interactions if not provided
        if not user_context.mood:
            user_context.mood = await self._detect_user_mood(user_id)
        
        return user_context
    
    async def _analyze_interaction(
        self,
        content: str,
        interaction_type: InteractionType,
        user_context: UserContext
    ) -> Dict[str, Any]:
        """Analyze interaction for personalization opportunities"""
        
        analysis = {
            "topics": [],
            "sentiment": "neutral",
            "urgency": "normal",
            "complexity": "medium",
            "personal_references": [],
            "suggested_actions": []
        }
        
        # Extract topics using Sentient's approach if available
        if self.sentient_available:
            analysis["topics"] = await self._extract_topics_sentient(content)
        else:
            analysis["topics"] = self._extract_topics_basic(content)
        
        # Analyze sentiment
        analysis["sentiment"] = self._analyze_sentiment(content)
        
        # Detect urgency
        analysis["urgency"] = self._detect_urgency(content, interaction_type)
        
        # Assess complexity
        analysis["complexity"] = self._assess_complexity(content)
        
        # Extract personal references
        analysis["personal_references"] = self._extract_personal_references(content)
        
        # Generate suggested actions
        analysis["suggested_actions"] = await self._generate_suggested_actions(
            content, analysis, user_context
        )
        
        return analysis
    
    async def _extract_topics_sentient(self, content: str) -> List[str]:
        """Extract topics using Sentient's approach"""
        
        try:
            # Use Sentient's topic classification logic
            topics = []
            content_lower = content.lower()
            
            for topic in TOPICS:
                topic_name = topic["name"].lower()
                topic_keywords = topic["description"].lower().split()
                
                # Check if content contains topic-related keywords
                if any(keyword in content_lower for keyword in topic_keywords):
                    topics.append(topic["name"])
            
            return topics
        except Exception as e:
            self.logger.warning(f"Sentient topic extraction failed: {e}")
            return self._extract_topics_basic(content)
    
    def _extract_topics_basic(self, content: str) -> List[str]:
        """Basic topic extraction fallback"""
        
        # Simple keyword-based topic detection
        topics = []
        content_lower = content.lower()
        
        topic_keywords = {
            "Personal Identity": ["i am", "my name", "i like", "i prefer", "my favorite"],
            "Work & Learning": ["work", "job", "project", "meeting", "deadline", "report"],
            "Health & Wellbeing": ["health", "exercise", "sleep", "stress", "wellness"],
            "Relationships & Social Life": ["friend", "family", "meeting", "social", "relationship"],
            "Financial": ["money", "budget", "expense", "investment", "financial"],
            "Goals & Challenges": ["goal", "challenge", "achieve", "problem", "solution"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _analyze_sentiment(self, content: str) -> str:
        """Basic sentiment analysis"""
        
        content_lower = content.lower()
        
        positive_words = ["good", "great", "excellent", "happy", "excited", "love", "like"]
        negative_words = ["bad", "terrible", "awful", "sad", "angry", "hate", "dislike"]
        
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _detect_urgency(self, content: str, interaction_type: InteractionType) -> str:
        """Detect urgency level in content"""
        
        urgent_indicators = ["urgent", "asap", "immediately", "now", "emergency", "critical"]
        content_lower = content.lower()
        
        if any(indicator in content_lower for indicator in urgent_indicators):
            return "high"
        elif interaction_type == InteractionType.TASK_REQUEST:
            return "medium"
        else:
            return "normal"
    
    def _assess_complexity(self, content: str) -> str:
        """Assess complexity of the request"""
        
        word_count = len(content.split())
        
        if word_count < 10:
            return "simple"
        elif word_count < 30:
            return "medium"
        else:
            return "complex"
    
    def _extract_personal_references(self, content: str) -> List[str]:
        """Extract personal references from content"""
        
        personal_indicators = ["i", "me", "my", "mine", "myself"]
        references = []
        
        words = content.lower().split()
        for i, word in enumerate(words):
            if word in personal_indicators and i + 1 < len(words):
                # Get the next word as potential personal reference
                next_word = words[i + 1]
                if next_word not in ["am", "will", "can", "should", "would"]:
                    references.append(f"{word} {next_word}")
        
        return references
    
    async def _generate_suggested_actions(
        self,
        content: str,
        analysis: Dict[str, Any],
        user_context: UserContext
    ) -> List[str]:
        """Generate suggested actions based on analysis"""
        
        suggestions = []
        
        # Task-related suggestions
        if analysis["urgency"] == "high":
            suggestions.append("Prioritize this request above other tasks")
        
        if "work" in analysis["topics"]:
            suggestions.append("Schedule dedicated time for this task")
        
        if "health" in analysis["topics"]:
            suggestions.append("Set up regular reminders for wellness activities")
        
        # Context-aware suggestions
        if user_context.mood == UserMood.STRESSED:
            suggestions.append("Consider breaking this into smaller, manageable steps")
        
        if user_context.current_activity == "meeting":
            suggestions.append("Wait until after the meeting to address this")
        
        return suggestions
    
    async def _generate_personalized_response(
        self,
        content: str,
        interaction_type: InteractionType,
        user_prefs: UserPreference,
        user_context: UserContext,
        personalization_data: Dict[str, Any]
    ) -> PersonalizedResponse:
        """Generate a personalized response"""
        
        # Determine response style based on user preferences and context
        response_style = self._select_response_style(user_prefs, user_context, personalization_data)
        
        # Determine detail level
        detail_level = self._select_detail_level(user_prefs, personalization_data)
        
        # Generate response content
        response_content = await self._generate_response_content(
            content, interaction_type, user_context, personalization_data
        )
        
        # Add personal context if relevant
        includes_personal_context = self._should_include_personal_context(
            personalization_data, user_context
        )
        
        if includes_personal_context:
            response_content = self._enhance_with_personal_context(
                response_content, user_context, personalization_data
            )
        
        # Generate follow-up questions
        follow_up_questions = await self._generate_follow_up_questions(
            personalization_data, user_context
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            personalization_data, user_context
        )
        
        return PersonalizedResponse(
            content=response_content,
            style=response_style,
            detail_level=detail_level,
            includes_personal_context=includes_personal_context,
            suggested_actions=personalization_data.get("suggested_actions", []),
            follow_up_questions=follow_up_questions,
            confidence_score=confidence_score
        )
    
    def _select_response_style(
        self,
        user_prefs: UserPreference,
        user_context: UserContext,
        personalization_data: Dict[str, Any]
    ) -> ResponseStyle:
        """Select appropriate response style"""
        
        # Base style from user preferences
        base_style = user_prefs.response_style
        
        # Adjust based on context
        if user_context.mood == UserMood.STRESSED:
            return ResponseStyle.FRIENDLY
        elif user_context.mood == UserMood.FOCUSED:
            return ResponseStyle.CONCISE
        elif personalization_data.get("urgency") == "high":
            return ResponseStyle.CONCISE
        
        return base_style
    
    def _select_detail_level(
        self,
        user_prefs: UserPreference,
        personalization_data: Dict[str, Any]
    ) -> str:
        """Select appropriate detail level"""
        
        # Base level from user preferences
        base_level = user_prefs.detail_level
        
        # Adjust based on complexity
        complexity = personalization_data.get("complexity", "medium")
        
        if complexity == "simple":
            return "concise"
        elif complexity == "complex":
            return "detailed"
        else:
            return base_level
    
    async def _generate_response_content(
        self,
        content: str,
        interaction_type: InteractionType,
        user_context: UserContext,
        personalization_data: Dict[str, Any]
    ) -> str:
        """Generate the main response content"""
        
        # Use Sentient's approach if available
        if self.sentient_available:
            return await self._generate_sentient_response(
                content, interaction_type, user_context, personalization_data
            )
        else:
            return self._generate_basic_response(
                content, interaction_type, user_context, personalization_data
            )
    
    async def _generate_sentient_response(
        self,
        content: str,
        interaction_type: InteractionType,
        user_context: UserContext,
        personalization_data: Dict[str, Any]
    ) -> str:
        """Generate response using Sentient's approach"""
        
        try:
            # Adapt Sentient's system prompt for personalization
            system_prompt = STAGE_2_SYSTEM_PROMPT.format(
                username=user_context.user_id,
                location=user_context.location or "Not specified",
                current_user_time=user_context.current_time
            )
            
            # Add personalization context
            enhanced_prompt = f"{system_prompt}\n\nPersonal Context:\n"
            enhanced_prompt += f"- Current Mood: {user_context.mood.value if user_context.mood else 'Unknown'}\n"
            enhanced_prompt += f"- Recent Topics: {', '.join(user_context.recent_topics)}\n"
            enhanced_prompt += f"- Active Tasks: {len(user_context.active_tasks)} tasks in progress\n"
            
            # Generate response based on interaction type
            if interaction_type == InteractionType.TASK_REQUEST:
                return f"I understand you'd like me to help with a task. Based on your current context and preferences, I'll approach this in a way that works best for you. Let me break this down into manageable steps."
            elif interaction_type == InteractionType.INFORMATION_QUERY:
                return f"I'll help you find that information. Given your interests and recent activities, I think I can provide some particularly relevant insights."
            else:
                return f"Thanks for reaching out! I'm here to help and I'll make sure to tailor my response to your preferences and current situation."
                
        except Exception as e:
            self.logger.warning(f"Sentient response generation failed: {e}")
            return self._generate_basic_response(
                content, interaction_type, user_context, personalization_data
            )
    
    def _generate_basic_response(
        self,
        content: str,
        interaction_type: InteractionType,
        user_context: UserContext,
        personalization_data: Dict[str, Any]
    ) -> str:
        """Generate basic response when Sentient is not available"""
        
        if interaction_type == InteractionType.TASK_REQUEST:
            return "I'll help you with that task. Let me understand your requirements and create a plan that fits your preferences."
        elif interaction_type == InteractionType.INFORMATION_QUERY:
            return "I'll search for that information and present it in a way that's most useful for you."
        elif interaction_type == InteractionType.VOICE_COMMAND:
            return "I've processed your voice command and I'm ready to help. Let me adapt my response to your current context."
        else:
            return "I'm here to help! I'll make sure my response is personalized to your needs and preferences."
    
    def _should_include_personal_context(
        self,
        personalization_data: Dict[str, Any],
        user_context: UserContext
    ) -> bool:
        """Determine if personal context should be included"""
        
        # Include if there are personal references or relevant topics
        has_personal_references = len(personalization_data.get("personal_references", [])) > 0
        has_relevant_topics = len(personalization_data.get("topics", [])) > 0
        
        return has_personal_references or has_relevant_topics
    
    def _enhance_with_personal_context(
        self,
        response: str,
        user_context: UserContext,
        personalization_data: Dict[str, Any]
    ) -> str:
        """Enhance response with personal context"""
        
        enhanced = response
        
        # Add mood-aware language
        if user_context.mood:
            if user_context.mood == UserMood.STRESSED:
                enhanced += " I notice you might be feeling a bit stressed, so I'll keep this straightforward and actionable."
            elif user_context.mood == UserMood.FOCUSED:
                enhanced += " I can see you're in a focused state, so I'll provide the information efficiently."
        
        # Add topic relevance
        topics = personalization_data.get("topics", [])
        if topics:
            enhanced += f" This relates to areas you've been working on: {', '.join(topics[:2])}."
        
        return enhanced
    
    async def _generate_follow_up_questions(
        self,
        personalization_data: Dict[str, Any],
        user_context: UserContext
    ) -> List[str]:
        """Generate relevant follow-up questions"""
        
        questions = []
        
        # Topic-based questions
        topics = personalization_data.get("topics", [])
        if "Work & Learning" in topics:
            questions.append("Would you like me to help you schedule time for this in your calendar?")
        
        if "Health & Wellbeing" in topics:
            questions.append("Should I set up regular reminders for wellness activities?")
        
        # Context-based questions
        if user_context.mood == UserMood.STRESSED:
            questions.append("Would you like me to break this down into smaller, more manageable steps?")
        
        if len(user_context.active_tasks) > 3:
            questions.append("I notice you have several active tasks. Should we prioritize this one?")
        
        return questions[:3]  # Limit to 3 questions
    
    def _calculate_confidence_score(
        self,
        personalization_data: Dict[str, Any],
        user_context: UserContext
    ) -> float:
        """Calculate confidence score for the response"""
        
        score = 0.5  # Base score
        
        # Boost for good topic extraction
        if len(personalization_data.get("topics", [])) > 0:
            score += 0.2
        
        # Boost for sentiment analysis
        if personalization_data.get("sentiment") != "neutral":
            score += 0.1
        
        # Boost for personal references
        if len(personalization_data.get("personal_references", [])) > 0:
            score += 0.1
        
        # Boost for context availability
        if user_context.mood or user_context.current_activity:
            score += 0.1
        
        return min(score, 1.0)
    
    def _extract_recent_topics(self, user_id: str) -> List[str]:
        """Extract recent topics from interaction history"""
        
        recent_interactions = [
            interaction for interaction in self.interaction_history
            if interaction.get("user_id") == user_id
        ][-10:]  # Last 10 interactions
        
        topics = []
        for interaction in recent_interactions:
            if "topics" in interaction:
                topics.extend(interaction["topics"])
        
        return list(set(topics))  # Remove duplicates
    
    def _get_active_tasks(self, user_id: str) -> List[str]:
        """Get active tasks for the user"""
        
        # This would integrate with the task system
        # For now, return mock data
        return ["Project planning", "Email review", "Meeting preparation"]
    
    def _get_available_tools(self, user_id: str) -> List[str]:
        """Get available tools for the user"""
        
        # This would integrate with the tool system
        # For now, return mock data
        return ["memory", "tasks", "voice", "search"]
    
    async def _detect_user_mood(self, user_id: str) -> Optional[UserMood]:
        """Detect user mood from recent interactions"""
        
        recent_interactions = [
            interaction for interaction in self.interaction_history
            if interaction.get("user_id") == user_id
        ][-5:]  # Last 5 interactions
        
        if not recent_interactions:
            return None
        
        # Analyze sentiment trends
        positive_count = sum(
            1 for interaction in recent_interactions
            if interaction.get("sentiment") == "positive"
        )
        negative_count = sum(
            1 for interaction in recent_interactions
            if interaction.get("sentiment") == "negative"
        )
        
        if positive_count > negative_count:
            return UserMood.HAPPY
        elif negative_count > positive_count:
            return UserMood.STRESSED
        else:
            return UserMood.FOCUSED
    
    async def _update_interaction_history(
        self,
        user_id: str,
        interaction_type: InteractionType,
        content: str,
        response: PersonalizedResponse
    ):
        """Update interaction history for learning"""
        
        interaction_record = {
            "user_id": user_id,
            "timestamp": time.time(),
            "type": interaction_type.value,
            "content": content,
            "response_style": response.style.value,
            "detail_level": response.detail_level,
            "includes_personal_context": response.includes_personal_context,
            "confidence_score": response.confidence_score
        }
        
        self.interaction_history.append(interaction_record)
        
        # Maintain history size
        if len(self.interaction_history) > self.config.max_context_history:
            self.interaction_history = self.interaction_history[-self.config.max_context_history:]
    
    async def _learn_from_interaction(
        self,
        user_id: str,
        content: str,
        response: PersonalizedResponse,
        user_context: UserContext
    ):
        """Learn from the interaction to improve future responses"""
        
        if not self.config.enable_context_learning:
            return
        
        # Update user preferences based on interaction
        user_prefs = self.user_preferences.get(user_id)
        if user_prefs and response.confidence_score > self.config.response_personalization_threshold:
            # Adjust preferences based on successful personalization
            if response.style != user_prefs.response_style:
                # User responded well to a different style, consider adapting
                pass
        
        # Update context patterns
        if user_id not in self.context_patterns:
            self.context_patterns[user_id] = {}
        
        user_patterns = self.context_patterns[user_id]
        
        # Track topic preferences
        topics = await self._extract_topics_sentient(content) if self.sentient_available else self._extract_topics_basic(content)
        for topic in topics:
            if topic not in user_patterns:
                user_patterns[topic] = {"count": 0, "last_interaction": 0}
            user_patterns[topic]["count"] += 1
            user_patterns[topic]["last_interaction"] = time.time()
    
    async def _load_user_preferences(self, user_id: str):
        """Load user preferences from persistent storage"""
        
        # This would integrate with persistent storage
        # For now, use default preferences
        pass
    
    async def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about user behavior and preferences"""
        
        user_prefs = await self._get_user_preferences(user_id)
        user_patterns = self.context_patterns.get(user_id, {})
        
        insights = {
            "preferences": {
                "response_style": user_prefs.response_style.value,
                "detail_level": user_prefs.detail_level,
                "formality": user_prefs.formality
            },
            "topics": {
                "favorite": self._get_favorite_topic(user_patterns),
                "recent": self._extract_recent_topics(user_id),
                "trending": self._get_trending_topics(user_patterns)
            },
            "interaction_patterns": {
                "total_interactions": len([i for i in self.interaction_history if i.get("user_id") == user_id]),
                "preferred_time": self._get_preferred_interaction_time(user_id),
                "response_satisfaction": self._calculate_response_satisfaction(user_id)
            }
        }
        
        return insights
    
    def _get_favorite_topic(self, user_patterns: Dict[str, Any]) -> Optional[str]:
        """Get user's favorite topic based on interaction patterns"""
        
        if not user_patterns:
            return None
        
        favorite = max(user_patterns.items(), key=lambda x: x[1]["count"])
        return favorite[0] if favorite[1]["count"] > 1 else None
    
    def _get_trending_topics(self, user_patterns: Dict[str, Any]) -> List[str]:
        """Get trending topics based on recent interactions"""
        
        if not user_patterns:
            return []
        
        current_time = time.time()
        recent_threshold = 24 * 60 * 60  # 24 hours
        
        trending = [
            topic for topic, data in user_patterns.items()
            if current_time - data["last_interaction"] < recent_threshold
        ]
        
        return trending[:5]  # Top 5 trending topics
    
    def _get_preferred_interaction_time(self, user_id: str) -> str:
        """Get user's preferred interaction time"""
        
        user_interactions = [
            interaction for interaction in self.interaction_history
            if interaction.get("user_id") == user_id
        ]
        
        if not user_interactions:
            return "Unknown"
        
        # Simple time analysis - this could be enhanced
        return "Afternoon"  # Placeholder
    
    def _calculate_response_satisfaction(self, user_id: str) -> float:
        """Calculate response satisfaction score"""
        
        user_interactions = [
            interaction for interaction in self.interaction_history
            if interaction.get("user_id") == user_id
        ][-20:]  # Last 20 interactions
        
        if not user_interactions:
            return 0.0
        
        # Use confidence scores as a proxy for satisfaction
        avg_confidence = sum(
            interaction.get("confidence_score", 0.0) 
            for interaction in user_interactions
        ) / len(user_interactions)
        
        return avg_confidence
    
    async def cleanup(self):
        """Clean up resources"""
        
        # Save user preferences to persistent storage
        for user_id, prefs in self.user_preferences.items():
            await self._save_user_preferences(user_id, prefs)
    
    async def _save_user_preferences(self, user_id: str, preferences: UserPreference):
        """Save user preferences to persistent storage"""
        
        # This would integrate with persistent storage
        # For now, just log
        self.logger.info(f"Saving preferences for user {user_id}")


# Factory function for creating personal agents
def create_personal_agent(config: Optional[PersonalAgentConfig] = None) -> PersonalAgent:
    """Create a new personal agent instance"""
    
    if config is None:
        config = PersonalAgentConfig()
    
    return PersonalAgent(config)