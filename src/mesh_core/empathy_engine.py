"""
Empathy Engine - Integration with focused-empathy system
Social Repair Algorithms for conflict resolution and perspective-taking
"""

import asyncio
import sys
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
import torch

# Add focused-empathy to path
# Try to import focused-empathy components
try:
    sys.path.append('/Users/admin/AI/focused-empathy')
    from agents.gee_agent import GeeCauseInferenceAgent
    from modules.empathy_scorer import EmpathyScorer
    from utils.etc_utils import EMOTION_LABELS
    FOCUSED_EMPATHY_AVAILABLE = True
except ImportError:
    # Mock classes for development/testing
    class GeeCauseInferenceAgent:
        """Mock GEE agent for development"""
        def __init__(self, **kwargs):
            pass
        
        def inference(self, *args, **kwargs):
            return {"emotion": "neutral", "cause": "unknown", "confidence": 0.5}
    
    class EmpathyScorer:
        """Mock empathy scorer for development"""
        def __init__(self, **kwargs):
            pass
        
        def score(self, *args, **kwargs):
            return 0.7
    
    EMOTION_LABELS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
    FOCUSED_EMPATHY_AVAILABLE = False


@dataclass
class EmpathyResponse:
    """Response from empathy processing"""
    mediation_guidance: str
    empathy_score: float
    emotion_causes: List[str]
    perspective_analysis: Dict[str, Any]
    resolution_steps: List[str]
    processing_time: float


@dataclass
class ConflictAnalysis:
    """Analysis of conflict between users"""
    shared_truths: List[str]
    perspective_gaps: List[Dict[str, Any]]
    emotional_triggers: List[str]
    mediation_strategy: str
    success_probability: float


class EmpathyEngine:
    """
    Empathy Engine for social repair and conflict resolution
    
    Integrates focused-empathy system for:
    - Emotion cause recognition using GEE (Generative Emotion Estimator)
    - Empathetic response generation focused on emotion causes
    - Perspective-taking algorithms for conflict mediation
    - Social repair when users disagree
    
    Core Philosophy: "The Mesh doesn't suppress—it diagnoses the misalignment 
    and proposes shared context or empathy-based reintroduction"
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize focused-empathy components
        self.gee_agent = None
        self.empathy_scorer = None
        
        # Configuration
        self.emotion_model = config.get('emotion_model', 'focused-empathy')
        self.empathy_threshold = config.get('empathy_threshold', 0.6)
        self.mediation_style = config.get('mediation_style', 'collaborative')
        
        # Initialize empathy components
        self._initialize_empathy_components()
        
        self.logger.info("Empathy Engine initialized - Ready for social repair")
    
    def _initialize_empathy_components(self):
        """Initialize focused-empathy system components"""
        
        try:
            # Initialize GEE agent for emotion cause recognition
            self._initialize_gee_agent()
            
            # Initialize empathy scorer
            self._initialize_empathy_scorer()
            
            self.logger.info("Focused-empathy components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize empathy components: {e}")
            self.logger.warning("Falling back to basic empathy analysis")
            self._use_basic_empathy = True
    
    def _initialize_gee_agent(self):
        """Initialize Generative Emotion Estimator"""
        
        # Note: This requires the focused-empathy environment and models
        # For now, we'll create a placeholder that can be enhanced
        
        try:
            # This would initialize the actual GEE agent
            # gee_opt = {...}  # Configuration for GEE
            # self.gee_agent = GeeCauseInferenceAgent(gee_opt)
            
            # Placeholder for development
            self.gee_agent = None
            self.logger.info("GEE agent placeholder ready")
            
        except Exception as e:
            self.logger.warning(f"GEE agent initialization failed: {e}")
    
    def _initialize_empathy_scorer(self):
        """Initialize empathy scoring system"""
        
        try:
            # This would initialize the actual empathy scorer
            # self.empathy_scorer = EmpathyScorer()
            
            # Placeholder for development  
            self.empathy_scorer = None
            self.logger.info("Empathy scorer placeholder ready")
            
        except Exception as e:
            self.logger.warning(f"Empathy scorer initialization failed: {e}")
    
    async def generate_mediation(
        self,
        conflict_query: str,
        context: Dict[str, Any],
        truth_facts: List[Dict[str, Any]]
    ) -> EmpathyResponse:
        """
        Generate mediation guidance for conflicts
        
        When two users disagree, this generates:
        - Shared objective truths they can both accept
        - Perspective-taking to understand each viewpoint  
        - Collaborative resolution steps
        - Empathetic reintroduction pathway
        """
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Analyze emotional context
            emotion_analysis = await self._analyze_emotions(conflict_query, context)
            
            # Find emotion causes using GEE
            emotion_causes = await self._identify_emotion_causes(
                conflict_query, 
                emotion_analysis
            )
            
            # Generate perspective analysis
            perspective_analysis = await self._analyze_perspectives(
                conflict_query,
                context,
                truth_facts
            )
            
            # Create mediation strategy
            mediation_guidance = await self._generate_mediation_strategy(
                emotion_analysis,
                emotion_causes, 
                perspective_analysis,
                truth_facts
            )
            
            # Calculate empathy score
            empathy_score = self._calculate_empathy_score(
                mediation_guidance,
                emotion_analysis
            )
            
            # Generate resolution steps
            resolution_steps = self._create_resolution_steps(
                perspective_analysis,
                truth_facts
            )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return EmpathyResponse(
                mediation_guidance=mediation_guidance,
                empathy_score=empathy_score,
                emotion_causes=emotion_causes,
                perspective_analysis=perspective_analysis,
                resolution_steps=resolution_steps,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Mediation generation error: {e}")
            return self._create_basic_mediation_response(conflict_query, start_time)
    
    async def _analyze_emotions(
        self, 
        text: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze emotions in conflict text"""
        
        # Extract emotional indicators
        emotional_words = self._extract_emotional_words(text)
        
        # Determine primary emotions
        primary_emotions = self._classify_emotions(text, emotional_words)
        
        # Assess emotional intensity
        intensity = self._calculate_emotional_intensity(text, emotional_words)
        
        return {
            'primary_emotions': primary_emotions,
            'emotional_words': emotional_words,
            'intensity': intensity,
            'context_emotions': context.get('emotions', [])
        }
    
    def _extract_emotional_words(self, text: str) -> List[str]:
        """Extract emotionally charged words from text"""
        
        # Basic emotion word detection (can be enhanced with NLP)
        emotion_indicators = {
            'anger': ['angry', 'furious', 'mad', 'irritated', 'frustrated'],
            'sadness': ['sad', 'depressed', 'disappointed', 'upset', 'hurt'],
            'fear': ['afraid', 'scared', 'worried', 'anxious', 'concerned'],
            'joy': ['happy', 'excited', 'pleased', 'delighted', 'glad'],
            'disgust': ['disgusted', 'revolted', 'appalled', 'repulsed'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked']
        }
        
        found_emotions = []
        text_lower = text.lower()
        
        for emotion, words in emotion_indicators.items():
            for word in words:
                if word in text_lower:
                    found_emotions.append(word)
        
        return found_emotions
    
    def _classify_emotions(
        self, 
        text: str, 
        emotional_words: List[str]
    ) -> List[str]:
        """Classify primary emotions in text"""
        
        # Simple classification based on emotional words
        # In production, this would use the focused-empathy models
        
        emotion_counts = {}
        emotion_map = {
            'angry': 'anger', 'furious': 'anger', 'mad': 'anger',
            'sad': 'sadness', 'disappointed': 'sadness', 'hurt': 'sadness',
            'afraid': 'fear', 'worried': 'fear', 'anxious': 'fear',
            'happy': 'joy', 'excited': 'joy', 'pleased': 'joy'
        }
        
        for word in emotional_words:
            emotion = emotion_map.get(word)
            if emotion:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Return top emotions
        return sorted(emotion_counts.keys(), key=lambda x: emotion_counts[x], reverse=True)[:3]
    
    def _calculate_emotional_intensity(
        self, 
        text: str, 
        emotional_words: List[str]
    ) -> float:
        """Calculate overall emotional intensity"""
        
        if not emotional_words:
            return 0.0
        
        # Simple intensity based on emotional word count and text length
        intensity = len(emotional_words) / max(len(text.split()), 1)
        
        # Boost for intensifying words
        intensifiers = ['very', 'extremely', 'really', 'absolutely', 'completely']
        text_lower = text.lower()
        
        for intensifier in intensifiers:
            if intensifier in text_lower:
                intensity *= 1.2
        
        return min(1.0, intensity)
    
    async def _identify_emotion_causes(
        self,
        text: str,
        emotion_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Identify causes of emotions using GEE model
        
        This is where focused-empathy's strength shines - identifying 
        what specifically triggers emotions in the text
        """
        
        if self.gee_agent:
            try:
                # Use actual GEE model for emotion cause recognition
                # causes = await self.gee_agent.identify_causes(text, emotion_analysis)
                # return causes
                pass
            except Exception as e:
                self.logger.warning(f"GEE analysis failed: {e}")
        
        # Fallback: Simple cause identification
        causes = []
        
        # Look for common cause patterns
        cause_patterns = [
            ('because', 'causal explanation'),
            ('due to', 'causal factor'),  
            ('when', 'situational trigger'),
            ('after', 'temporal trigger'),
            ('if', 'conditional trigger')
        ]
        
        text_lower = text.lower()
        for pattern, cause_type in cause_patterns:
            if pattern in text_lower:
                # Extract text around the pattern
                pattern_index = text_lower.find(pattern)
                if pattern_index != -1:
                    cause_context = text[max(0, pattern_index-20):pattern_index+50]
                    causes.append(f"{cause_type}: {cause_context.strip()}")
        
        return causes[:5]  # Return top 5 causes
    
    async def _analyze_perspectives(
        self,
        conflict_query: str,
        context: Dict[str, Any], 
        truth_facts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze different perspectives in the conflict
        
        Implements perspective-taking to understand multiple viewpoints
        """
        
        perspectives = {}
        
        # Extract different viewpoints from context
        if 'user_positions' in context:
            for i, position in enumerate(context['user_positions']):
                perspective_key = f"user_{i+1}"
                perspectives[perspective_key] = {
                    'position': position,
                    'supporting_facts': self._find_supporting_facts(position, truth_facts),
                    'emotional_state': self._analyze_emotional_state(position),
                    'core_values': self._identify_core_values(position)
                }
        
        # Find common ground
        common_ground = self._identify_common_ground(perspectives, truth_facts)
        
        # Identify perspective gaps
        perspective_gaps = self._identify_perspective_gaps(perspectives)
        
        return {
            'perspectives': perspectives,
            'common_ground': common_ground,
            'perspective_gaps': perspective_gaps,
            'mediation_opportunities': self._find_mediation_opportunities(
                perspectives, 
                common_ground
            )
        }
    
    def _find_supporting_facts(
        self, 
        position: str, 
        truth_facts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find truth facts that support a given position"""
        
        supporting_facts = []
        position_lower = position.lower()
        
        for fact in truth_facts:
            fact_content = fact.get('content', '').lower()
            
            # Simple keyword matching (can be enhanced with semantic similarity)
            if any(word in fact_content for word in position_lower.split() if len(word) > 3):
                supporting_facts.append(fact)
        
        return supporting_facts
    
    def _analyze_emotional_state(self, position: str) -> Dict[str, Any]:
        """Analyze emotional state behind a position"""
        
        emotions = self._classify_emotions(position, self._extract_emotional_words(position))
        intensity = self._calculate_emotional_intensity(position, self._extract_emotional_words(position))
        
        return {
            'primary_emotions': emotions,
            'intensity': intensity,
            'defensive': any(word in position.lower() for word in ['wrong', 'stupid', 'ridiculous']),
            'collaborative': any(word in position.lower() for word in ['understand', 'agree', 'together'])
        }
    
    def _identify_core_values(self, position: str) -> List[str]:
        """Identify core values expressed in a position"""
        
        value_indicators = {
            'fairness': ['fair', 'equal', 'just', 'equitable'],
            'security': ['safe', 'secure', 'protected', 'stable'],
            'freedom': ['free', 'liberty', 'choice', 'independent'],
            'truth': ['true', 'honest', 'accurate', 'factual'],
            'compassion': ['care', 'help', 'support', 'empathy']
        }
        
        identified_values = []
        position_lower = position.lower()
        
        for value, indicators in value_indicators.items():
            if any(indicator in position_lower for indicator in indicators):
                identified_values.append(value)
        
        return identified_values
    
    def _identify_common_ground(
        self, 
        perspectives: Dict[str, Any], 
        truth_facts: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify shared beliefs and values between perspectives"""
        
        common_ground = []
        
        # Find shared values
        all_values = []
        for perspective in perspectives.values():
            all_values.extend(perspective.get('core_values', []))
        
        # Count value frequency
        value_counts = {}
        for value in all_values:
            value_counts[value] = value_counts.get(value, 0) + 1
        
        # Values shared by multiple perspectives
        shared_values = [
            value for value, count in value_counts.items() 
            if count > 1
        ]
        
        if shared_values:
            common_ground.append(f"Shared values: {', '.join(shared_values)}")
        
        # Find facts all perspectives can accept
        high_confidence_facts = [
            fact for fact in truth_facts 
            if fact.get('reliability_score', 0) >= 0.8
        ]
        
        if high_confidence_facts:
            common_ground.append(f"Objective facts both can accept: {len(high_confidence_facts)} verified facts")
        
        return common_ground
    
    def _identify_perspective_gaps(
        self, 
        perspectives: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify gaps between different perspectives"""
        
        gaps = []
        perspective_list = list(perspectives.items())
        
        for i in range(len(perspective_list)):
            for j in range(i + 1, len(perspective_list)):
                user1_key, user1_data = perspective_list[i]
                user2_key, user2_data = perspective_list[j]
                
                # Compare emotional states
                user1_emotions = set(user1_data['emotional_state']['primary_emotions'])
                user2_emotions = set(user2_data['emotional_state']['primary_emotions'])
                
                emotional_gap = user1_emotions.symmetric_difference(user2_emotions)
                
                # Compare values
                user1_values = set(user1_data['core_values'])
                user2_values = set(user2_data['core_values'])
                
                value_gap = user1_values.symmetric_difference(user2_values)
                
                if emotional_gap or value_gap:
                    gaps.append({
                        'users': [user1_key, user2_key],
                        'emotional_differences': list(emotional_gap),
                        'value_differences': list(value_gap),
                        'bridgeable': len(emotional_gap) + len(value_gap) <= 3
                    })
        
        return gaps
    
    def _find_mediation_opportunities(
        self,
        perspectives: Dict[str, Any],
        common_ground: List[str]
    ) -> List[str]:
        """Find opportunities for mediation"""
        
        opportunities = []
        
        if common_ground:
            opportunities.append(f"Build on shared foundation: {common_ground[0]}")
        
        # Look for collaborative emotional states
        collaborative_users = []
        for user_key, data in perspectives.items():
            if data['emotional_state'].get('collaborative', False):
                collaborative_users.append(user_key)
        
        if len(collaborative_users) >= 2:
            opportunities.append("Both parties show openness to collaboration")
        
        # Check for low defensiveness
        non_defensive_users = []
        for user_key, data in perspectives.items():
            if not data['emotional_state'].get('defensive', True):
                non_defensive_users.append(user_key)
        
        if non_defensive_users:
            opportunities.append("Reduced defensiveness creates space for dialogue")
        
        return opportunities
    
    async def _generate_mediation_strategy(
        self,
        emotion_analysis: Dict[str, Any],
        emotion_causes: List[str],
        perspective_analysis: Dict[str, Any],
        truth_facts: List[Dict[str, Any]]
    ) -> str:
        """Generate comprehensive mediation strategy"""
        
        strategy_parts = []
        
        # Opening with empathy acknowledgment
        if emotion_analysis['primary_emotions']:
            emotions_str = ', '.join(emotion_analysis['primary_emotions'])
            strategy_parts.append(
                f"I understand this situation involves strong emotions: {emotions_str}. "
                "Both perspectives deserve to be heard and understood."
            )
        
        # Address emotion causes if identified
        if emotion_causes:
            strategy_parts.append(
                "The core triggers in this situation appear to be: "
                + '; '.join(emotion_causes[:2])
            )
        
        # Present common ground
        common_ground = perspective_analysis.get('common_ground', [])
        if common_ground:
            strategy_parts.append(
                f"You both share important common ground: {common_ground[0]}"
            )
        
        # Present objective facts
        if truth_facts:
            high_confidence_facts = [
                fact for fact in truth_facts 
                if fact.get('reliability_score', 0) >= 0.8
            ]
            
            if high_confidence_facts:
                strategy_parts.append(
                    f"Here are {len(high_confidence_facts)} verified facts we can all agree on:"
                )
                
                for i, fact in enumerate(high_confidence_facts[:2], 1):
                    strategy_parts.append(f"{i}. {fact['content']}")
        
        # Suggest perspective-taking
        perspectives = perspective_analysis.get('perspectives', {})
        if len(perspectives) >= 2:
            strategy_parts.append(
                "To move forward, would you each be willing to share what you think "
                "the other person's main concern is? This can help build mutual understanding."
            )
        
        # Propose resolution pathway
        opportunities = perspective_analysis.get('mediation_opportunities', [])
        if opportunities:
            strategy_parts.append(
                f"I see an opportunity: {opportunities[0]}. "
                "How might we build on this together?"
            )
        
        return '\n\n'.join(strategy_parts)
    
    def _calculate_empathy_score(
        self,
        mediation_text: str,
        emotion_analysis: Dict[str, Any]
    ) -> float:
        """Calculate empathy score of the mediation response"""
        
        if self.empathy_scorer:
            try:
                # Use actual empathy scorer
                # score = self.empathy_scorer.score(mediation_text, emotion_analysis)
                # return score
                pass
            except Exception as e:
                self.logger.warning(f"Empathy scoring failed: {e}")
        
        # Fallback: Simple empathy scoring
        empathy_indicators = [
            'understand', 'feel', 'perspective', 'both', 'together',
            'respect', 'acknowledge', 'valid', 'concern', 'care'
        ]
        
        text_lower = mediation_text.lower()
        empathy_count = sum(1 for indicator in empathy_indicators if indicator in text_lower)
        
        # Normalize to 0-1 scale
        empathy_score = min(1.0, empathy_count / 10.0)
        
        # Boost if addresses emotions directly
        if emotion_analysis.get('primary_emotions'):
            empathy_score *= 1.2
        
        return min(1.0, empathy_score)
    
    def _create_resolution_steps(
        self,
        perspective_analysis: Dict[str, Any],
        truth_facts: List[Dict[str, Any]]
    ) -> List[str]:
        """Create concrete steps for conflict resolution"""
        
        steps = []
        
        # Step 1: Acknowledge each perspective
        perspectives = perspective_analysis.get('perspectives', {})
        if len(perspectives) >= 2:
            steps.append("Each person shares their main concern without interruption")
        
        # Step 2: Identify shared values
        common_ground = perspective_analysis.get('common_ground', [])
        if common_ground:
            steps.append(f"Acknowledge shared foundation: {common_ground[0]}")
        
        # Step 3: Review objective facts
        if truth_facts:
            steps.append("Review objective facts together")
        
        # Step 4: Generate solutions collaboratively
        steps.append("Brainstorm solutions that honor both perspectives")
        
        # Step 5: Choose next steps
        steps.append("Agree on one small step forward")
        
        return steps
    
    def _create_basic_mediation_response(
        self, 
        query: str, 
        start_time: float
    ) -> EmpathyResponse:
        """Create basic mediation response when full processing fails"""
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        basic_mediation = (
            "I understand there's a disagreement here. "
            "Let's take a step back and try to understand each perspective. "
            "What would you say is your main concern in this situation?"
        )
        
        return EmpathyResponse(
            mediation_guidance=basic_mediation,
            empathy_score=0.6,
            emotion_causes=[],
            perspective_analysis={},
            resolution_steps=[
                "Each person states their main concern",
                "Look for any shared values or goals", 
                "Focus on small steps forward"
            ],
            processing_time=processing_time
        )
    
    async def analyze_conflict(
        self,
        user_a_position: str,
        user_b_position: str,
        context: Dict[str, Any]
    ) -> ConflictAnalysis:
        """
        Comprehensive conflict analysis between two positions
        
        Used when the Mesh detects disagreement and needs to diagnose 
        the misalignment for social repair
        """
        
        # Prepare context with both positions
        conflict_context = {
            **context,
            'user_positions': [user_a_position, user_b_position]
        }
        
        # Get truth facts relevant to the conflict
        truth_facts = context.get('truth_facts', [])
        
        # Generate empathy analysis
        empathy_response = await self.generate_mediation(
            f"Conflict: {user_a_position} vs {user_b_position}",
            conflict_context,
            truth_facts
        )
        
        # Extract analysis components
        perspective_analysis = empathy_response.perspective_analysis
        
        return ConflictAnalysis(
            shared_truths=perspective_analysis.get('common_ground', []),
            perspective_gaps=perspective_analysis.get('perspective_gaps', []),
            emotional_triggers=empathy_response.emotion_causes,
            mediation_strategy=empathy_response.mediation_guidance,
            success_probability=empathy_response.empathy_score
        )
    
    def get_empathy_system_status(self) -> Dict[str, Any]:
        """Get status of empathy system components"""
        
        return {
            'gee_agent_available': self.gee_agent is not None,
            'empathy_scorer_available': self.empathy_scorer is not None,
            'emotion_labels_count': len(EMOTION_LABELS) if EMOTION_LABELS else 0,
            'mediation_style': self.mediation_style,
            'empathy_threshold': self.empathy_threshold,
            'status': 'operational' if (self.gee_agent or self.empathy_scorer) else 'basic_mode'
        }